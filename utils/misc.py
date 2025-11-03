from flax import struct
import dataclasses
from dataclasses import fields, make_dataclass
from typing import Type, List, Any
import jax.numpy as jnp
import jax
from dataclasses import is_dataclass, fields, replace
from typing import Any, Mapping, Sequence


from game.improvements import Improvements

_MAX_BIT = max(imp._value_ for imp in Improvements)          # largest bit used
_BIT_RANGE = jnp.arange(_MAX_BIT + 1, dtype=jnp.uint32)      # 0 … _MAX_BIT

def improvement_mask_for_batch(
    improvement_bitfield: jnp.ndarray,  # (T,F,E,2,R) uint32
    terrain_map: jnp.ndarray,  # (H,W)
    feature_map: jnp.ndarray,
    elevation_map: jnp.ndarray,
    freshwater_map: jnp.ndarray,  # bool (True = fresh)
    resource_map: jnp.ndarray,
    lake_map: jnp.ndarray,
    rc_batch: jnp.ndarray  # (B,2) [[row,col],…]
) -> jnp.ndarray:
    """
    Returns a boolean mask of shape (B, _MAX_BIT+1).
    Column k is True iff bit-k is set for that tile – identical logic to:

        word = improvement_bitfield[t,f,e,w,r]
        (word & (1 << k)) != 0
    """

    rows, cols = rc_batch[:, 0], rc_batch[:, 1]

    # gather per-tile indices (cast to int32 so jax advanced indexing is happy)
    terr = terrain_map[rows, cols].astype(jnp.int32)
    feat = feature_map[rows, cols].astype(jnp.int32)
    elev = elevation_map[rows, cols].astype(jnp.int32)
    fresh = freshwater_map[rows, cols].astype(jnp.int32)  # 0 / 1
    res = resource_map[rows, cols].astype(jnp.int32)

    # packed uint32 word for each tile
    bits = improvement_bitfield[terr, feat, elev, fresh, res]  # (B,) uint32

    # explode all bits 0 … _MAX_BIT in one fused op
    mask = (bits[:, None] & (1 << _BIT_RANGE)) != 0  # (B, _MAX_BIT+1)
    is_not_lake = lake_map[rows, cols] == 0
    mask = mask & is_not_lake[:, None]
    return mask

def make_projected_flax_dataclass(
    base_cls: Type[Any],
    exclude: List[str],
    name: str = None,
) -> Type[Any]:
    """
    Create a new @flax.struct.dataclass by excluding certain fields from an existing one.

    Args:
        base_cls: The original flax dataclass class (e.g., `PlayerCity`)
        exclude: List of field names to exclude
        name: Optional name for the new dataclass

    Returns:
        A new flax.struct.dataclass class with only the included fields.
    """
    included_fields = [
        (f.name, f.type, f)  # name, type, Field object (for default)
        for f in fields(base_cls)
        if f.name not in exclude
    ]

    # If no name provided, default to BaseClassFiltered
    cls_name = name or f"{base_cls.__name__}Filtered"

    # Use dataclasses.make_dataclass to build a real Python type
    base_args = [
        (name, typ, f.default if f.default is not dataclasses.MISSING else struct.field())
        for name, typ, f in included_fields
    ]

    dynamic_cls = make_dataclass(cls_name, base_args, bases=(object,))
    
    # Now wrap with @flax.struct.dataclass
    return struct.dataclass(dynamic_cls)

def project_instance(source_obj, target_cls):
    """
    Given a source dataclass and a filtered flax dataclass (target_cls),
    initialize the target by copying only matching fields from the source.
    """
    return target_cls(**{
        f.name: getattr(source_obj, f.name)
        for f in fields(target_cls)
    })


def leaves_in_instance(tree: Any) -> List[Any]:
    """
    Return a flat list of leaves that are *present* inside this one PyTree
    instance, ignoring any dataclass fields that were added later and therefore
    do **not** exist on the loaded object.

    Works with:
      • flax/standard dataclasses (frozen or not)
      • dict / flax.core.FrozenDict
      • lists, tuples, named-tuples
      • arbitrary nested mixes of the above
    """
    # Case 1 – dataclass: recurse over its *actual* attributes
    if is_dataclass(tree):
        # `vars()` only shows attributes that exist on *this* instance
        return sum((leaves_in_instance(v) for v in vars(tree).values()), [])

    # Case 2 – mapping (dict, FrozenDict, …)
    if isinstance(tree, Mapping):
        return sum((leaves_in_instance(v) for v in tree.values()), [])

    # Case 3 – sequence (list, tuple, …) but not str / bytes
    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        return sum((leaves_in_instance(v) for v in tree), [])

    # Base case – treat as a leaf
    return [vars(tree)]

def _present_field_names(obj):
    """
    Yield the names of dataclass attributes that are *present on this instance*.

    • If the dataclass has a __dict__ (common when slots=False) we can just
      look at vars(obj).keys() – that’s fast and accurate.
    • Flax dataclasses default to slots=True, so __dict__ is absent and
      vars(obj) raises TypeError.  In that case we fall back to the declared
      dataclass fields and keep only those for which hasattr(obj, name) is true
      (those that were pickled).
    """
    try:                                  # slots=False path
        yield from vars(obj).keys()
    except TypeError:                     # slots=True path
        for f in fields(obj):
            if hasattr(obj, f.name):      # attribute really present
                yield f.name


def _merge_subset(a: Any, b: Any) -> Any:
    """
    Recursively copy the contents of dataclass `b` into `a`
    **only along the structure of `b`**.

    If both `a` and `b` are dataclasses we walk their fields;
    otherwise we’ve reached a leaf and simply return `b`.
    """
    if is_dataclass(a) and is_dataclass(b):
        updates = {
            name: _merge_subset(getattr(a, name), getattr(b, name))
            for name in _present_field_names(b)   # only fields that *exist* in b
            if hasattr(a, name)                   # …and also exist in a
        }
        return replace(a, **updates)           # flax dataclasses support `replace`
    else:
        return b                              # leaf → overwrite

def set_from_subset(A, B):
    """
    Return a copy of dataclass `A` whose values are replaced by those in `B`
    wherever `B` defines a field (deep merge).
    """
    if not (is_dataclass(A) and is_dataclass(B)):
        raise TypeError("Both inputs must be dataclass instances.")
    return _merge_subset(A, B)


import numpy as np
def log_to_disk(format_str, *args):
    # args arrive as NumPy scalars/arrays on the host
    py_args = []
    for a in args:
        a = np.asarray(a)
        py_args.append(a.item() if a.shape == () else a.tolist())
    with open('game_log.txt', 'a') as f:
        f.write(format_str.format(*py_args) + "\n")


from functools import partial
import dataclasses
import numpy as np
import jax.numpy as jnp
from jax.experimental import io_callback  # JAX host callback

# -------------------------- helpers --------------------------

def _indent_block(s: str, indent: int = 4) -> str:
    pad = " " * indent
    return "\n".join(pad + line for line in s.splitlines())

def _array_to_full_string(a: np.ndarray) -> str:
    """
    Convert array/scalar to a fully expanded string (no '...').
    """
    a = np.asarray(a)
    if a.shape == ():  # scalar
        return str(a.item())
    # Ensure NO summarization: set threshold > total element count
    thr = int(a.size) + 1
    return np.array2string(
        a,
        separator=", ",
        max_line_width=200,  # wrap reasonably
        threshold=thr,       # <- disables '...'
        edgeitems=0,
        suppress_small=False
    )

def _flatten_first_batch(obj, prefix=""):
    """
    Recursively traverse (nested) dataclasses; collect (name, value[0]) pairs.
    For non-dataclass leaves (arrays), take [0] if possible, else the value.
    Returns: list[(full_name, leaf_value_slice0)]
    """
    out = []

    def rec(o, pfx):
        if dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                child = getattr(o, f.name)
                name = f"{pfx}.{f.name}" if pfx else f.name
                rec(child, name)
        else:
            try:
                v0 = o[0]  # take first parallel game
            except Exception:
                v0 = o
            out.append((pfx, v0))

    rec(obj, prefix)
    return out

# ----------------------- host-side logger -----------------------

def _log_blocks_to_disk(format_str, *args):
    """
    Host-only function called by io_callback.
    The `format_str` alternates placeholders: (player_id, array), (player_id, array), ...
    We therefore treat even-indexed args as player_id, odd-indexed as arrays.
    """
    formatted = []
    for i, a in enumerate(args):
        a = np.asarray(a)
        if i % 2 == 0:
            # player_id placeholder
            formatted.append(str(int(a.item())))
        else:
            # observation array/scalar block
            formatted.append(_indent_block(_array_to_full_string(a), indent=4))

    with open("game_log.txt", "a") as f:
        f.write(format_str.format(*formatted))
        f.write("\n")

# ------------------------- public API --------------------------

def io_log_full_observation(obs, player_id):
    """
    Log every array in TerraNovaObservation (including nested dataclasses),
    taking the first element along axis 0. Each block is labeled:

    FIELD.NAME (player_id):
        [[full array, no ellipses], ...]
    
    `player_id` can be a Python int or a JAX scalar; it will be passed through
    the callback so it’s available on the host.
    """
    parts = _flatten_first_batch(obs)  # [(name, arr0), ...]
    pid_arg = jnp.asarray(player_id)   # ensure a JAX-compatible scalar

    fmt_lines = []
    args = []
    for name, arr in parts:
        # Two placeholders per field: one for (player_id), one for the array block
        fmt_lines.append(f"{name.upper()} ({{}}):\n{{}}\n")
        args.extend([pid_arg, arr])

    format_str = "".join(fmt_lines)

    # Send everything in one callback. The host function will format labels + arrays.
    io_callback(partial(_log_blocks_to_disk, format_str), None, *args)

