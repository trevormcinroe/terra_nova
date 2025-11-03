
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Do this at the very beginning of your script

import argparse
import os
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
from tqdm import tqdm

from game.primitives import GameState
from game.units import Units
from utils.misc import set_from_subset


# ---------- per-task worker with hard timeout/kill ----------
def _runner(q, cfg, key, border):
    """Child process target: run map generation and push result/exception back via Queue."""
    try:
        from map.generate import generate_map  # local import inside child
        result = generate_map(cfg=cfg, key=key, border=border)
        q.put(("ok", result))  # may stream a lot of bytes; parent must read before join
        # print("DONE!")  # (optional) child-side stdout can be noisy
    except BaseException as e:
        import traceback
        q.put(("err", (type(e).__name__, str(e), traceback.format_exc())))


from queue import Empty
import concurrent.futures

def run_generate_map_with_timeout(cfg, key, border=3, timeout_s=600):
    """
    Run generate_map in an isolated child process and hard-kill on timeout.
    IMPORTANT: Read from Queue BEFORE join() to avoid deadlock on large payloads.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()  # parent will drain it before joining
    p = ctx.Process(target=_runner, args=(q, cfg, key, border))
    p.start()

    try:
        # 1) Drain the Queue within the timeout window (this lets the child flush the pipe).
        try:
            print("waiting....")
            tag, payload = q.get(timeout=timeout_s)
        except Empty:
            # Child didn’t send anything in time -> treat as timeout and kill it.
            print("Waited too damn long")
            if p.is_alive():
                p.terminate()
                p.join()
            raise concurrent.futures.TimeoutError

        # 2) Now the pipe is drained; the child should be able to exit promptly.
        p.join(timeout=10)
        if p.is_alive():
            # Belt-and-suspenders: if it still hasn’t exited, kill it.
            p.terminate()
            p.join()

        # 3) Handle the child’s message.
        if tag == "ok":
            print("Payload tag is oK")
            return payload
        else:
            cls, msg, tb = payload
            raise RuntimeError(f"{cls}: {msg}\n{tb}")

    finally:
        try:
            q.close()
        except Exception:
            pass


# Alternative version with horizontal wrapping support
@partial(jax.jit, static_argnames=['rows', 'cols'])
def update_cost_map_for_mountains_jax_v2(_gamestate, rows=42, cols=66):
    """
    Alternative JAX implementation with horizontal wrapping (cylindrical map).
    Handles east-west wraparound while treating north-south boundaries as invalid.
    """
    # Extract arrays
    movement_cost_map = _gamestate.movement_cost_map
    neighboring_hexes_map = _gamestate.neighboring_hexes_map
    elevation_map = _gamestate.elevation_map
    nw_map = _gamestate.nw_map
    
    # Get neighbor coordinates
    neighbor_rows = neighboring_hexes_map[..., 0]  # Shape: (42, 66, 6)
    neighbor_cols = neighboring_hexes_map[..., 1]  # Shape: (42, 66, 6)
    
    # Handle vertical boundaries (no wrapping)
    # Mark neighbors that are out of bounds vertically
    valid_row_mask = (neighbor_rows >= 0) & (neighbor_rows < rows)
    
    # Handle horizontal wrapping
    # Wrap column indices using modulo - this handles both negative and overflow
    wrapped_cols = neighbor_cols % cols  # This wraps -1 to 65, 66 to 0, etc.
    
    # For invalid rows, use safe index 0 but we'll mask them out later
    safe_rows = jnp.where(valid_row_mask, neighbor_rows, 0)
    
    # Get neighbor values using wrapped columns
    neighbor_elevations = elevation_map[safe_rows, wrapped_cols]
    neighbor_nw = nw_map[safe_rows, wrapped_cols]
    
    # Check impassability only for valid neighbors (valid row)
    # Invalid row neighbors stay at their original cost
    is_impassable = valid_row_mask & ((neighbor_elevations == 3) | (neighbor_nw > 0))
    
    # Update movement costs
    return jnp.where(is_impassable, 999, movement_cost_map).astype(movement_cost_map.dtype)



# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--start_seed", type=int, required=True, help="The seed to begin map generation.")
parser.add_argument("--num_maps", type=int, required=True, help="The number of maps to generate.")
args = parser.parse_args()


if __name__ == "__main__":
    key = jax.random.PRNGKey(args.start_seed)

    cfg = OmegaConf.load("./map/configs.yaml")
    print(f"Using config: \n{cfg}")

    codebase_dir = Path(__file__).resolve().parent
    maps_dir = codebase_dir / "saved_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    num_generated = 0

    with tqdm(total=args.num_maps) as pbar:
        while num_generated < args.num_maps:
            try:
                # Run one map generation in an isolated child with a hard timeout
                result = run_generate_map_with_timeout(cfg, key, border=3, timeout_s=300)

                (
                    landmask, elevation_map, terrain, river_map, lakes, features, fertility,
                    settler_rowcols, subregion_stats, nw_placements, cs_rowcols, cs_ownership_map,
                    all_resource_map, all_resource_quantity_map, resource_type_map, freshwater_map, yield_map
                ) = result

                # Build idx_to_hex_rowcol lookup
                l = []
                for row in range(landmask.shape[0]):
                    for col in range(landmask.shape[1]):
                        l.append([row, col])
                l = jnp.array(l)

                n_units = 250
                units = Units.create(6, n_units, settler_rowcols=settler_rowcols, warrior_rowcols=settler_rowcols)

                gamestate = GameState(
                    landmask_map=landmask,
                    elevation_map=elevation_map,
                    terrain_map=terrain,
                    edge_river_map=river_map,
                    lake_map=lakes,
                    feature_map=features,
                    nw_map=nw_placements,
                    player_ownership_map=None,
                    cs_ownership_map=cs_ownership_map,
                    cs_cities=jnp.array(cs_rowcols),
                    player_cities=None,
                    all_resource_map=all_resource_map,
                    all_resource_quantity_map=all_resource_quantity_map,
                    all_resource_type_map=resource_type_map,
                    freshwater_map=freshwater_map,
                    yield_map=yield_map,
                    units=units,
                    idx_arange=jnp.arange(start=0, stop=landmask.shape[0] * landmask.shape[1]),
                    idx_to_hex_rowcol=l,
                    current_step=jnp.array([0])
                )
                
                print("Computing movement cost...")
                gamestate = gamestate.compute_movement_cost_array()
                print("Factoring in mountain ranges...")
                new_movement_cost_map = update_cost_map_for_mountains_jax_v2(gamestate)
                print("Saving map...")
                gamestate = gamestate.replace(movement_cost_map=new_movement_cost_map)

                gamestate.save(maps_dir / f"{args.start_seed + num_generated}_turn0.gamestate")

                num_generated += 1
                pbar.update(1)

            except Exception:
                # On timeout or any failure, advance RNG and retry next seed
                key, _ = jax.random.split(key)
                print(f"A balanced map could not be made with the seed {args.start_seed + num_generated}, trying again...")

            finally:
                # Always advance RNG between attempts to avoid reusing a problematic seed path
                key, _ = jax.random.split(key)
