import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import fields
from game.religion import MAX_IDX_PANTHEON
from game.resources import ALL_RESOURCES_TECH, RESOURCE_TO_IDX, RESOURCE_YIELDS
from game.social_policies import SocialPolicies

def rowcol_to_hex(rowcol):
    row = rowcol[0]
    col = rowcol[1]
    return row * 66 + col

def unique_rows(arr):
    # Sort rows lexicographically
    arr = arr[jnp.lexsort((arr[:, 1], arr[:, 0]))]

    # Compare each row to the previous one
    diffs = jnp.any(arr[1:] != arr[:-1], axis=1)

    # Always keep the first row + any row that differs from the one before it
    keep = jnp.concatenate([jnp.array([True]), diffs])

    return arr[keep]


def get_hexspace_deltas_from_gamestate_row(row):
    """
    All we care about is the gamestate row, as our hexgrid board is 
    row-based offset, which makes the "jagged" line the board's columns

    Returns:
        array of neighbors (row, col) in gamestate array space, starting with the hex to
        the direct west and going clockwise
    """
    deltas_even_row = [
        (0, -1), # west
        (-1, -1), # northwest
        (-1, 0), # northeast
        (0, 1), # east
        (1, 0), # southeast
        (1, -1), # southwest
    ]
    deltas_odd_row = [
        (0, -1), # west
        (-1, 0), # northwest
        (-1, 1), # northeast
        (0, 1), # east
        (1, 1), # southeast
        (1, 0), # southwest
    ]
    
    even_row_bool = row % 2 == 0
    return jnp.array(deltas_even_row) * even_row_bool + (1 - even_row_bool) * jnp.array(deltas_odd_row)


def get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row, col, max_rows, max_cols):
    """
    The row, col here are in matrix [x, y] --> matrix[row, col] = value centered on

    """
    # Here, the col in the matrix form gives us the row in the hex grid, as the hex grid
    # works on an inverted cartesian plane where (0, 0) is top-left
    deltas = get_hexspace_deltas_from_gamestate_row(row)
    neighbors = jnp.array([row, col]) + deltas

    # Now we need to handle the toroidal west/east and bounded north/south
    # Wrap columns (east-west)
    # TODO: double check that we want to do (max_cols + 1)!!!
    neighbors = neighbors.at[:, 1].set(neighbors[:, 1] % (max_cols))
    neighbors = neighbors.at[:, 0].set(jnp.clip(neighbors[:, 0], 0, max_rows))

    return neighbors

### PRECOMPUTING 3 TILE RING ###
# Ring 1: 6 hexes (immediate neighbors)
RING_1_OFFSETS_EVEN = jnp.array([
    [0, -1],   # west
    [-1, -1],  # northwest  
    [-1, 0],   # northeast
    [0, 1],    # east
    [1, 0],    # southeast
    [1, -1],   # southwest
])

RING_1_OFFSETS_ODD = jnp.array([
    [0, -1],   # west
    [-1, 0],   # northwest
    [-1, 1],   # northeast  
    [0, 1],    # east
    [1, 1],    # southeast
    [1, 0],    # southwest
])

# Ring 2: 12 hexes (precomputed by hand or script)
RING_2_OFFSETS_EVEN = jnp.array([
    [0, -2],   # [10, 12] = [10, 14] + [0, -2]
    [-1, -2],  # [9, 12] = [10, 14] + [-1, -2]
    [-2, -1],  # [8, 13] = [10, 14] + [-2, -1]
    [-2, 0],   # [8, 14] = [10, 14] + [-2, 0]
    [-2, 1],   # [8, 15] = [10, 14] + [-2, 1]
    [-1, 1],   # [9, 15] = [10, 14] + [-1, 1]
    [0, 2],    # [10, 16] = [10, 14] + [0, 2]
    [1, 1],    # [11, 15] = [10, 14] + [1, 1]
    [2, 1],    # [12, 15] = [10, 14] + [2, 1]
    [2, 0],    # [12, 14] = [10, 14] + [2, 0]
    [2, -1],   # [12, 13] = [10, 14] + [2, -1]
    [1, -2],   # [11, 12] = [10, 14] + [1, -2]
])

RING_2_OFFSETS_ODD = jnp.array([
    [0, -2],   # 2 steps west
    [-1, -1],  # adjusted for odd row offset
    [-2, -1],   # adjusted for odd row offset
    [-2, 0],   # adjusted for odd row offset
    [-2, 1],   # adjusted for odd row offset
    [-1, 2],   # adjusted for odd row offset
    [0, 2],    # 2 steps east
    [1, 2],    # adjusted for odd row offset
    [2, 1],    # adjusted for odd row offset
    [2, 0],    # adjusted for odd row offset
    [2, -1],    # adjusted for odd row offset
    [1, -1],   # adjusted for odd row offset
])

# Ring 3: 18 hexes (distance 3 from center)
RING_3_OFFSETS_EVEN = jnp.array([
    [0, -3],   # [10, 11] = [10, 14] + [0, -3]
    [-1, -3],  # [9, 11] = [10, 14] + [-1, -3]
    [-2, -2],  # [8, 12] = [10, 14] + [-2, -2]
    [-3, -2],  # [7, 12] = [10, 14] + [-3, -2]
    [-3, -1],  # [7, 13] = [10, 14] + [-3, -1]
    [-3, 0],   # [7, 14] = [10, 14] + [-3, 0]
    [-3, 1],   # [7, 15] = [10, 14] + [-3, 1]
    [-2, 2],   # [8, 16] = [10, 14] + [-2, 2]
    [-1, 2],   # [9, 16] = [10, 14] + [-1, 2]
    [0, 3],    # [10, 17] = [10, 14] + [0, 3]
    [1, 2],    # [11, 16] = [10, 14] + [1, 2]
    [2, 2],    # [12, 16] = [10, 14] + [2, 2]
    [3, 1],    # [13, 15] = [10, 14] + [3, 1]
    [3, 0],    # [13, 14] = [10, 14] + [3, 0]
    [3, -1],   # [13, 13] = [10, 14] + [3, -1]
    [3, -2],   # [13, 12] = [10, 14] + [3, -2]
    [2, -2],   # [12, 12] = [10, 14] + [2, -2]
    [1, -3],   # [11, 11] = [10, 14] + [1, -3]
])

RING_3_OFFSETS_ODD = jnp.array([
    [0, -3],   # 3 steps west
    [-1, -2],  # adjusted for odd row
    [-2, -2],  # adjusted for odd row
    [-3, -1],  # adjusted for odd row
    [-3, 0],   # adjusted for odd row
    [-3, 1],   # adjusted for odd row
    [-3, 2],   # adjusted for odd row
    [-2, 2],   # adjusted for odd row
    [-1, 3],   # adjusted for odd row
    [0, 3],    # 3 steps east
    [1, 3],    # adjusted for odd row
    [2, 2],    # adjusted for odd row
    [3, 2],    # adjusted for odd row
    [3, 1],    # adjusted for odd row
    [3, 0],    # adjusted for odd row
    [3, -1],   # adjusted for odd row
    [2, -2],   # adjusted for odd row
    [1, -2],   # adjusted for odd row
])

def get_hex_rings_vectorized(center_row, center_col, max_rows, max_cols):
    """
    Ultra-fast hex ring computation using precomputed offsets.
    Returns rings 1, 2, and 3 around the center position.
    """
    center_pos = jnp.array([center_row, center_col])
    center_is_even = center_row % 2 == 0
    
    # Ring 1 (6 hexes)
    ring_1_offsets = (center_is_even * RING_1_OFFSETS_EVEN + 
                      (1 - center_is_even) * RING_1_OFFSETS_ODD)
    ring_1 = center_pos + ring_1_offsets
    
    # Ring 2 (12 hexes)
    ring_2_offsets = (center_is_even * RING_2_OFFSETS_EVEN +
                      (1 - center_is_even) * RING_2_OFFSETS_ODD)
    ring_2 = center_pos + ring_2_offsets
    
    # Ring 3 (18 hexes)
    ring_3_offsets = (center_is_even * RING_3_OFFSETS_EVEN +
                      (1 - center_is_even) * RING_3_OFFSETS_ODD)
    ring_3 = center_pos + ring_3_offsets
    
    # Apply boundary conditions
    def apply_boundaries(positions):
        wrapped_cols = positions[:, 1] % max_cols
        clipped_rows = jnp.clip(positions[:, 0], 0, max_rows)
        return jnp.stack([clipped_rows, wrapped_cols], axis=1)
    
    ring_1 = apply_boundaries(ring_1)
    ring_2 = apply_boundaries(ring_2)
    ring_3 = apply_boundaries(ring_3)
    
    return ring_1, ring_2, ring_3




def generate_6d_border_vector_from_ownership_matrix(ownership_matrix):
    num_owners = ownership_matrix.max()
    H, W = ownership_matrix.shape
    ownership_borders = jnp.zeros(shape=(num_owners, H, W, 6), dtype=jnp.uint8)

    for i in range(1, num_owners + 1):
        for row in range(H):
            for col in range(W):
                if ownership_matrix[row, col] == i:
                    # If the current entry belongs to player i, then we look at all of the surrounding
                    # hexes. If there is no owned hex by this player in one of the surrounding hexes, 
                    # then we know that the current [row, col] hex is at the border.
                    surround_hexes = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=row, col=col, max_cols=W, max_rows=H)
                    
                    # These returned tiles start west and then work clockwise.
                    for _i, _hex in enumerate(surround_hexes):
                        if ownership_matrix[_hex[0], _hex[1]] == i:
                            continue
                        if _i == 0:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 0].set(1)
                        elif _i == 1:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 1].set(1)
                        elif _i == 2:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 5].set(1)
                        elif _i == 3:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 3].set(1)
                        elif _i == 4:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 4].set(1)
                        elif _i == 5:
                            ownership_borders = ownership_borders.at[i - 1, row, col, 2].set(1)
                        else:
                            raise ValueError("Hexagons only have 6 sizes, ya bozo!")

    return ownership_borders         
#print(get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(0, 3, 3, 4))

# ─── helper constants (unchanged) ─────────────────────────────────────────────
_DELTAS_EVEN = jnp.array([[ 0, -1], [-1, -1], [-1,  0],
                          [ 0,  1], [ 1,  0], [ 1, -1]])
_DELTAS_ODD  = jnp.array([[ 0, -1], [-1,  0], [-1,  1],
                          [ 0,  1], [ 1,  1], [ 1,  0]])
_DIR_REMAP   = jnp.array([0, 1, 5, 3, 4, 2], dtype=jnp.int32)


# ─── public wrapper (computes num_players concretely) ─────────────────────────
def generate_6d_border_vector_from_ownership_matrix_jit(ownership_matrix: jnp.ndarray
                                                    ) -> jnp.ndarray:
    num_players = int(ownership_matrix.max())   # concrete Python int
    return _generate_6d_border_vector_from_ownership_matrix_jit(
        ownership_matrix, num_players)


# ─── jittable core (num_players is static) ────────────────────────────────────
@partial(jax.jit, static_argnums=(1,))          # <- num_players is static
def _generate_6d_border_vector_from_ownership_matrix_jit(
        ownership_matrix: jnp.ndarray,
        num_players: int                        # now compile-time constant
) -> jnp.ndarray:

    H, W        = ownership_matrix.shape
    row_idx     = jnp.arange(H)[:, None]        # (H,1)
    col_idx     = jnp.arange(W)[None, :]        # (1,W)
    even_row    = (row_idx % 2 == 0)

    # --- neighbour owner IDs --------------------------------------------------
    nbr_ids = []
    for k in range(6):
        dr = jnp.where(even_row, _DELTAS_EVEN[k, 0], _DELTAS_ODD[k, 0])
        dc = jnp.where(even_row, _DELTAS_EVEN[k, 1], _DELTAS_ODD[k, 1])
        rr = jnp.clip(row_idx + dr, 0, H - 1)
        cc = (col_idx + dc) % W
        nbr_ids.append(ownership_matrix[rr, cc])
    nbr_ids = jnp.stack(nbr_ids, axis=-1)       # (H,W,6)

    # --- border mask ----------------------------------------------------------
    border_mask = (ownership_matrix[..., None] != nbr_ids)  # (H,W,6) bool

    # --- one-hot of owners (num_players is static so one_hot is OK) -----------
    owner_onehot = jax.nn.one_hot(ownership_matrix - 1,
                                  num_classes=num_players,
                                  dtype=jnp.uint8)           # (H,W,N)
    owner_onehot = jnp.moveaxis(owner_onehot, 2, 0)[..., None]  # (N,H,W,1)

    # --- final tensor ---------------------------------------------------------
    borders = owner_onehot * border_mask.astype(jnp.uint8)    # (N,H,W,6)
    borders = borders[..., _DIR_REMAP]                        # channel reorder
    return borders


def compute_dist_from_a_to_b(start_rc, goal_rc, elevation_map, nw_map, hex_neighbors):
    """
    Returns minimal step distance (int32) or +∞ if unreachable.

    """
    H, W = elevation_map.shape
    N    = H * W

    # 1. flatten helpers -------------------------------------------------------
    idx_map = jnp.arange(N, dtype=jnp.int32).reshape(H, W)

    # (N,6)  neighbour indices as flat ints
    nbr_idx = idx_map[
        hex_neighbors[..., 0].reshape(-1),
        hex_neighbors[..., 1].reshape(-1)
    ].reshape(N, 6)

    # 2. passability mask ------------------------------------------------------
    passable = ~( (elevation_map == 3) | (nw_map > 0) )   # (H,W) bool
    pass_flat = passable.ravel()

    # 3. start / goal indices --------------------------------------------------
    start_idx = start_rc[0] * W + start_rc[1]
    goal_idx  = goal_rc[0]  * W + goal_rc[1]

    # 4. BFS with boolean frontiers -------------------------------------------
    visited  = jnp.zeros(N, dtype=jnp.bool_).at[start_idx].set(True)
    frontier = visited                       # first wave contains start
    dist     = jnp.int32(0)

    def cond(carry):
        vis, front, d = carry
        still_searching = ~vis[goal_idx] & front.any()
        return still_searching

    def body(carry):
        vis, front, d = carry

        # tiles adjacent to current frontier
        adj = jnp.any(front[nbr_idx], axis=1)       # (N,) bool

        # next frontier: passable, unvisited, adjacent
        next_front = adj & pass_flat & (~vis)

        vis = vis | next_front
        return vis, next_front, d + 1

    vis, front, dist = jax.lax.while_loop(cond, body, (visited, frontier, dist))

    # 5. distance or ∞ ---------------------------------------------------------
    return jnp.where(vis[goal_idx], dist, jnp.inf)


def growth_threshold(n: jnp.ndarray):
    """
    This function computes the threshold of food to grow to the nth+1 population
    The (2/3) modifier is for "quick" speed
    """
    return jnp.floor((15 + 8 * (n - 1) + (n - 1)**1.5) * (2 / 3))


def social_policy_threshold(n_cities: jnp.ndarray, policies: jnp.ndarray):
    """
    """

    p = policies.sum().astype(jnp.float32)
    base = 5 * jnp.floor(1.25 * p * p - 0.25 * p + 3.5)  # p=0→15, 1→20, 2→40, 3→70
    scaled = base * (1.0 + 0.10 * jnp.maximum(n_cities - 1.0, 0.0))  # +10% per extra city
    return (5 * jnp.round(scaled / 5.0)).astype(jnp.int32)  # round to nearest 5

def pantheon_threshold(game_religions: jnp.ndarray):
    """
    10 + 5 * n_pantheons_chosen
    """
    n_pantheons_chosen = game_religions[:, :MAX_IDX_PANTHEON].sum()
    return 10 + 5 * n_pantheons_chosen



def compute_yields_and_resources_snapshot(all_resource_map, technologies, yield_map):
    """
    In order to save on memory requirements (i.e., not having to store the entire yield 
    and resource map) at every timestep in the replay buffer, we can compute it cheaply here

    player_id should not be necessary here, as all info is either coming from (1) ObservationSpace
    or (2) ReplayBuffer. In the case of (1), all data within will relate to the correct player_id
    because of the way we scan over player_id in the steps. For (2), we're only ever gathering
    and training with player_id=0
    """
    print("---------------------------------------")
    print(f"{all_resource_map.shape} / {technologies.shape} / {yield_map.shape}")
    print("---------------------------------------")
    techs_required = ALL_RESOURCES_TECH[all_resource_map][..., 0]
    
    # flipping bit to bool-like int?
    techs_have = 1 - technologies[techs_required]

    # Do not need to +1 to RESOURCE_TO_IDX, as it is already padded
    horses = all_resource_map == RESOURCE_TO_IDX["horses"]
    uranium = all_resource_map == RESOURCE_TO_IDX["uranium"]
    oil = all_resource_map == RESOURCE_TO_IDX["oil"]
    aluminum = all_resource_map == RESOURCE_TO_IDX["aluminium"]
    coal = all_resource_map == RESOURCE_TO_IDX["coal"]
    iron = all_resource_map == RESOURCE_TO_IDX["iron"]
    
    # yields, on the other hand, does need to be - 1
    horses_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["horses"] - 1][0]
    uranium_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["uranium"] - 1][0]
    oil_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["oil"] - 1][0]
    aluminum_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["aluminium"] - 1][0]
    coal_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["coal"] - 1][0]
    iron_yield = RESOURCE_YIELDS[RESOURCE_TO_IDX["iron"] - 1][0]

    new_yield_map = yield_map

    for _map, _yield in [(horses, horses_yield), (uranium, uranium_yield), (oil, oil_yield), (aluminum, aluminum_yield), (coal, coal_yield), (iron, iron_yield)]:
        new_yield_map = new_yield_map - techs_have[..., None] * _map[..., None] * _yield 
    
    # Lastly, update the resources that are visible to the players
    new_visible_resources = all_resource_map * (1 - techs_have)
    
    return new_yield_map, new_visible_resources


def oddr_to_cube(r, q):
    """Convert odd-r offset (row=r, col=q) → cube (x,y,z)."""
    x = q - ((r - (r & 1)) // 2)
    z = r
    y = -x - z
    return x, y, z

def compute_dist_from_a_to_b_as_crow_flies(rowcol_a, rowcol_b):
    """"""
    a_row, a_col = rowcol_a
    b_row, b_col = rowcol_b
    ax, ay, az = oddr_to_cube(a_row, a_col)
    bx, by, bz = oddr_to_cube(b_row, b_col)
    return jnp.max(jnp.abs(jnp.array([ax-bx, ay-by, az-bz])))

def compute_all_distances_vectorized(start_rowcol, all_city_rowcols):
    """Vectorized version - no vmap needed"""
    # Convert start position to cube coordinates
    start_row, start_col = start_rowcol
    start_x = start_col - ((start_row - (start_row & 1)) // 2)
    start_z = start_row
    start_y = -start_x - start_z
    
    # Convert all city positions to cube coordinates (vectorized)
    city_rows, city_cols = all_city_rowcols[:, 0], all_city_rowcols[:, 1]
    city_x = city_cols - ((city_rows - (city_rows & 1)) // 2)
    city_z = city_rows
    city_y = -city_x - city_z
    
    # Compute distances (fully vectorized)
    dx = jnp.abs(start_x - city_x)
    dy = jnp.abs(start_y - city_y) 
    dz = jnp.abs(start_z - city_z)
    
    return jnp.maximum(jnp.maximum(dx, dy), dz)

def border_growth_threshold(player_id, city_int, ownership_map):
    """
    https://steamcommunity.com/app/289070/discussions/0/312265473879006404/
    6.87 * n^2.06
    where nth tile being grown to
    Each city begins with 6 non-city-center tiles. Therefore, we subtract 5 to get the nth _next_ tile to 
    grow to.
    """
    nth_tile = (ownership_map[player_id[0], city_int] == 2).sum() - 5
    threshold = 6.87 * nth_tile**2.06
    return threshold


GRID_ROWS = 42
GRID_COLS = 66

def hex_flat_index(rowcol):
    """Convert (row, col) to flat index in a pointy-topped hex grid with odd-row horizontal offset."""
    return rowcol[:, 0] * GRID_COLS + rowcol[:, 1]

def compute_unit_maintenance(game, player_id):
    """(n * b * (1 + g * m) / 100) * (1 + g / d)"""
    has_olig = game.policies[player_id[0], SocialPolicies["oligarchy"]._value_] == 1
    n_cities = (game.player_cities.city_ids[player_id[0]] > 0).sum()
    to_sub = has_olig * n_cities

    u = (game.units.unit_type[player_id[0]] > 0).sum() - to_sub
    f = 1
    n = jnp.maximum(0, u - f)

    t = game.current_step[0]

    b = 50
    m = 8
    d = 7
    e = 330
    g = t / e

    return (n * b * (1 + g*m) / 100) * (1 + g/d)


# ---------- branchless zero-fill shift ----------
def _shift_bool(x: jnp.ndarray, dr: int, dc: int) -> jnp.ndarray:
    """Shift by (dr, dc) with zero-fill, no Python branching on arrays."""
    H, W = x.shape
    rp, rn = max(dr, 0), max(-dr, 0)  # rows to pad-top / crop-top
    cp, cn = max(dc, 0), max(-dc, 0)  # cols to pad-left / crop-left
    # crop the source to the overlapping rectangle
    src = x[rn:H-rp, cn:W-cp]
    # place it into zeros at the shifted location
    y = jnp.zeros_like(x, dtype=bool)
    y = jax.lax.dynamic_update_slice(y, src, (rp, cp))
    return y

def _hex_expand(frontier: jnp.ndarray, even_row_mask: jnp.ndarray) -> jnp.ndarray:
    fe = frontier &  even_row_mask
    fo = frontier & ~even_row_mask

    # common (west/east)
    neigh = (_shift_bool(frontier, 0, -1) | _shift_bool(frontier, 0,  1))

    # even rows: NW(-1,-1), NE(-1,0), SE(1,0), SW(1,-1)
    neigh |= (_shift_bool(fe, -1, -1) | _shift_bool(fe, -1, 0) |
              _shift_bool(fe,  1,  0) | _shift_bool(fe,  1, -1))
    # odd rows:  NW(-1,0), NE(-1,1), SE(1,1), SW(1,0)
    neigh |= (_shift_bool(fo, -1,  0) | _shift_bool(fo, -1, 1) |
              _shift_bool(fo,  1,  1) | _shift_bool(fo,  1, 0))
    return neigh

@jax.jit
def roads_connected(road_map: jnp.ndarray,
                    src_rc: jnp.ndarray,  # [2] int32
                    dst_rc: jnp.ndarray   # [2] int32
                   ) -> jnp.bool_:
    road = road_map > 0
    H, W = road.shape
    row_idxs = jnp.arange(H)[:, None]
    even_row_mask = (row_idxs & 1) == 0

    sr, sc = src_rc[0], src_rc[1]
    tr, tc = dst_rc[0], dst_rc[1]

    endpoints_ok = road[sr, sc] & road[tr, tc]

    def _run(_: jnp.int32) -> jnp.bool_:
        visited_a = jnp.zeros_like(road, dtype=bool).at[sr, sc].set(True)
        visited_b = jnp.zeros_like(road, dtype=bool).at[tr, tc].set(True)
        frontier_a = visited_a
        frontier_b = visited_b
        hit0 = (sr == tr) & (sc == tc)
        max_steps = jnp.int32(H * W)

        def cond_fn(state):
            va, vb, fa, fb, steps, hit = state
            more = jnp.logical_or(jnp.any(fa), jnp.any(fb))
            return (~hit) & more & (steps < max_steps)

        def body_fn(state):
            va, vb, fa, fb, steps, hit = state
            nbr_a = _hex_expand(fa, even_row_mask)
            nbr_b = _hex_expand(fb, even_row_mask)
            new_fa = nbr_a & road & (~va)
            new_fb = nbr_b & road & (~vb)
            va2, vb2 = va | new_fa, vb | new_fb
            meet = jnp.any(va2 & vb2)
            return (va2, vb2, new_fa, new_fb, steps + 1, meet)

        return jax.lax.while_loop(
            cond_fn, body_fn,
            (visited_a, visited_b, frontier_a, frontier_b, jnp.int32(0), hit0)
        )[-1]

    # Only dynamic branching is via lax.cond/while_loop (jit-safe).
    return jax.lax.cond(endpoints_ok, _run, lambda _: jnp.bool_(False), 0)
