"""
The generation process is roughly based on the LekMod map script: https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/
For now, we are skipping island generation.


Terrain (GRASS or PLAINS or DESERT or TUNDRA or SNOW)
AND
Elevation(NONE or HILLS or MOUNTAINS)
AND
Features(JUNGLE or MARSH or OASIS or FLOOD_PLAINS or FOREST or FALLOUT)

Process:
    1. Generate landmass (and conversely, the ocean)
    2. Generate flatlands, hills, mountains (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/LekmapPangaeaRectangularv4.lua#L430)
    3. Generate terrain types: (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/LekmapPangaeaRectangularv4.lua#L1232 --> https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/LekmapPangaeaRectangularv4.lua#L1232)
    3. Generate rivers (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBMapGeneratorRectangular.lua#L501)
    4. Generate lakes (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBMapGeneratorRectangular.lua#L576)
    5. Generate features ()
    6. Assign starting locations - players & CS ()
"""
from enum import IntEnum
from threading import local
import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Tuple
import numpy as np
from numpy.lib.shape_base import row_stack
from itertools import product
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from copy import deepcopy
from game.resources import ALL_RESOURCES, LAND_LUX, LUX_BIAS_TABLE, OCEAN_LUX, RESOURCE_TO_IDX, RESOURCE_YIELDS, STRATEGIC_BIAS_TABLE, translate_terrain_bias_to_tile_samples, translate_terrain_bias_to_tile_samples_strategic

from game.natural_wonders import ALL_NATURAL_WONDERS, LAKE_VICTORIA_IDX, NW_SPAWN_CRITERIA, NW_YIELD_TABLE_IDX
from utils.maths import get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol, unique_rows


#@jit
def bilinear_upsample(coarse: jnp.ndarray, target_shape: Tuple[int, int]) -> jnp.ndarray:
    """Upsample a coarse grid to fine resolution using bilinear interpolation."""
    H, W = target_shape
    h, w = coarse.shape

    y = jnp.linspace(0, h - 1, H)
    x = jnp.linspace(0, w - 1, W)
    grid_y, grid_x = jnp.meshgrid(y, x, indexing="ij")

    y0 = jnp.floor(grid_y).astype(int)
    x0 = jnp.floor(grid_x).astype(int)
    y1 = jnp.clip(y0 + 1, 0, h - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)

    wy = grid_y - y0
    wx = grid_x - x0

    top_left = coarse[y0, x0]
    top_right = coarse[y0, x1]
    bottom_left = coarse[y1, x0]
    bottom_right = coarse[y1, x1]

    top = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    return top * (1 - wy) + bottom * wy

def enforce_ocean_border(landmask: jnp.ndarray, border: int = 2) -> jnp.ndarray:
    """
    Force ocean border around the landmass by zeroing out the outer 'border' tiles.
    
    Args:
        landmask: (H, W) array of {0.0, 1.0}
        border: number of tiles to force to ocean on each edge
    
    Returns:
        landmask with coastal ocean ring
    """
    H, W = landmask.shape
    mask = jnp.ones((H, W), dtype=jnp.float32)
    mask = mask.at[:border, :].set(0.0)    # top
    mask = mask.at[-border:, :].set(0.0)   # bottom
    mask = mask.at[:, :border].set(0.0)    # left
    mask = mask.at[:, -border:].set(0.0)   # right
    return landmask * mask


def flood_fill_ocean(landmask: jnp.ndarray, max_iters: int = 32) -> jnp.ndarray:
    """
    Flood fill ocean from the border to remove inland ocean tiles.
    Any ocean tile not connected to the map edge will be filled as land.

    Args:
        landmask: (H, W) array where 1 = land, 0 = ocean
        max_iters: number of iterations to propagate flood fill

    Returns:
        cleaned_landmask: (H, W) array with inland oceans filled as land
    """
    H, W = landmask.shape
    is_ocean = (landmask == 0.0)

    # Initial mask: border ocean tiles
    border_mask = jnp.zeros_like(landmask, dtype=landmask.dtype)
    border_mask = border_mask.at[0, :].set(1.0)
    border_mask = border_mask.at[-1, :].set(1.0)
    border_mask = border_mask.at[:, 0].set(1.0)
    border_mask = border_mask.at[:, -1].set(1.0)

    flood = is_ocean & border_mask

    def step(flood):
        padded = jnp.pad(flood, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
        neighbors = (
            padded[ :-2, 1:-1] |  # up
            padded[2:  , 1:-1] |  # down
            padded[1:-1,  :-2] |  # left
            padded[1:-1, 2:  ]    # right
        )
        return (neighbors & is_ocean) | flood

    def loop_body(i, flood):
        return step(flood)

    flood = jax.lax.fori_loop(0, max_iters, loop_body, flood)

    # Final landmask: ocean = connected component only; everything else is land
    cleaned_landmask = jnp.where(flood, 0.0, 1.0)
    return cleaned_landmask

#@jit
def generate_landmask(key: jnp.ndarray, shape: Tuple[int, int], border: int) -> jnp.ndarray:
    """
    Generate a binary landmask with one central, organic-looking landmass.

    Returns:
        landmask: jnp.ndarray of shape (W, H), values in {0.0, 1.0}
    """
    H, W = shape
    k1, k2 = jax.random.split(key)

    # 1. Coarse low-freq continent base
    coarse_scale = 2
    coarse_shape = (H // coarse_scale, W // coarse_scale)
    base_noise = jax.random.uniform(k1, coarse_shape)
    base_noise = bilinear_upsample(base_noise, (H, W))

    # 2. High-freq detail noise to break up edges
    detail_noise = jax.random.uniform(k2, (H, W)) * 0.4

    # 3. Central elliptical falloff
    y, x = jnp.indices((H, W))
    cx, cy = W / 2, H / 2
    rx, ry = W * 0.45, H * 0.5
    ellipse = 1.0 - ((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2
    ellipse = jnp.clip(ellipse, 0.0, 1.0)
    
    # 4. Combine all factors
    # Higher combined threshold == more craggy perimeter
    combined = base_noise + 0.6 * ellipse + detail_noise
    landmask = (combined > 0.75).astype(jnp.uint8)
    
    # 5. Morphological closure to ensure no inland ocean
    #landmask = fill_enclosed_ocean(landmask)
    landmask = flood_fill_ocean(landmask)

    # 5. Ensure contiguous ocean around pangea's perimeter
    landmask = enforce_ocean_border(landmask, border=border)

    return landmask.T.astype(jnp.bool)

def generate_elevation_map(cfg, key: jnp.ndarray, landmask: jnp.ndarray) -> jnp.ndarray:
    """
    Generate elevation map with mountains, hills, and flatland based on LekMod-style ridges.

    Elevation values:
        0 = ocean
        1 = flatland
        2 = hill
        3 = mountain
    """
    H, W = landmask.shape
    key_mtn, key_hill = jax.random.split(key)

    # Fractal-like mountain and hill layers
    scale_hill = cfg.elevation_noise_scale
    scale_mtn = scale_hill * 2
    mtn_noise = bilinear_upsample(jax.random.uniform(key_mtn, (H // scale_mtn, W // scale_mtn)), (H, W))
    hill_noise = bilinear_upsample(jax.random.uniform(key_hill, (H // scale_hill, W // scale_hill)), (H, W))

    # Apply thresholds
    mountain_mask = (mtn_noise > cfg.mountain_threshold) & (landmask == 1.0)
    foothill_mask = (mtn_noise > cfg.foothill_threshold) & (landmask == 1.0) & (~mountain_mask)
    hill_mask = (hill_noise > cfg.hill_threshold) & (landmask == 1.0) & (~mountain_mask) & (~foothill_mask)

    # Compose elevation
    elevation_map = jnp.zeros((H, W), dtype=jnp.int32)
    elevation_map = elevation_map.at[hill_mask].set(2)
    elevation_map = elevation_map.at[foothill_mask].set(2)
    elevation_map = elevation_map.at[mountain_mask].set(3)
    elevation_map = jnp.where((landmask == 1.0) & (elevation_map == 0), 1, elevation_map)

    return elevation_map


def generate_terrain_type_map(cfg, key: jnp.ndarray, landmask: jnp.ndarray) -> jnp.ndarray:
    """
    Generate base terrain types using latitude bands + fractal modulation like LekMod.

    Terrain type encoding:
        0 = ocean (masked)
        1 = grassland
        2 = plains
        3 = desert
        4 = tundra
        5 = snow
    """
    H, W = landmask.shape
    key_grass, key_plains, key_desert, key_tundra, key_snow  = jax.random.split(key, 5)

    # --- Symmetric Latitude ---
    lat = jnp.abs(jnp.linspace(-1, 1, H)).reshape(H, 1).repeat(W, axis=1)

    # --- Fractal Noise Upscaling ---
    scale = cfg.terrain_noise_scale
    def up(k): 
        return bilinear_upsample(jax.random.uniform(k, (H // scale, W // scale)), (H, W))

    grass_noise = up(key_grass)
    plains_noise = up(key_plains)
    desert_noise = up(key_desert)
    tundra_noise = up(key_tundra)
    snow_noise = up(key_snow)

    is_land = landmask == 1.0
    terrain = jnp.zeros((H, W), dtype=jnp.int32)

    # --- Thresholds ---
    snow_lat = cfg.snow_lat
    tundra_lat = cfg.tundra_lat
    desert_lat_max = cfg.desert_lat_max

    snow_thresh = cfg.snow_thresh
    tundra_thresh = cfg.tundra_thresh
    grass_thresh = cfg.grass_thresh
    desert_thresh = cfg.desert_thresh
    plains_thresh = cfg.plains_thresh

    # --- Assignments ---
    snow_mask = (lat >= snow_lat) & (snow_noise >= snow_thresh) & is_land
    terrain = jnp.where(snow_mask, 5, terrain)

    tundra_mask = (lat >= tundra_lat) & (lat < snow_lat) & (tundra_noise >= tundra_thresh) & is_land 
    terrain = jnp.where(tundra_mask, 4, terrain)

    desert_mask = (
        (lat <= desert_lat_max) &
        (desert_noise >= desert_thresh) & is_land & (terrain == 0)
    )
    terrain = jnp.where(desert_mask, 3, terrain)

    plains_mask = (plains_noise >= plains_thresh) & is_land & (terrain == 0)
    terrain = jnp.where(plains_mask, 2, terrain)

    grass_mask = (grass_noise >= grass_thresh) & is_land & (terrain == 0)
    terrain = jnp.where(grass_mask, 1, terrain)

    # Fallback
    terrain = jnp.where((terrain == 0) & is_land, 1, terrain)

    return terrain


def compute_coastline_mask(landmask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute coastline mask (1 = coast, 0 = non-coast) for a binary landmask.
    A tile is coastline if it is land and has at least one ocean neighbor,
    where neighbors are counted in all 8 directions (including diagonals).
    """
    is_land = landmask.astype(bool)
    is_ocean = ~is_land

    # Pad the ocean mask
    pad = jnp.pad(is_ocean, ((1, 1), (1, 1)), constant_values=0)

    # Count ocean neighbors in all 8 directions
    ocean_neighbor_count = (
        pad[:-2, 1:-1].astype(int) +  # north
        pad[2:, 1:-1].astype(int)  +  # south
        pad[1:-1, :-2].astype(int) +  # west
        pad[1:-1, 2:].astype(int)  +  # east
        pad[:-2, :-2].astype(int) +  # northwest
        pad[:-2, 2:].astype(int)  +  # northeast
        pad[2:, :-2].astype(int)  +  # southwest
        pad[2:, 2:].astype(int)       # southeast
    )

    # Coastline is land tiles with at least one ocean neighbor
    coastline_mask = is_land & (ocean_neighbor_count >= 1)
    return coastline_mask.astype(jnp.int32)


def generate_river_sources(cfg_rivers, cfg_game, key: jnp.ndarray, elevation: jnp.ndarray, landmask: jnp.ndarray, border: int) -> jnp.ndarray:
    """
    Generate river *source points* (start tiles, no paths yet). We try to achieve balance by dividing the map
    into n_player regions and generating ~the same number of rivers per region.

    """

    H, W = elevation.shape
    is_land = landmask
    is_ocean = ~landmask

    # Elevation: 0=ocean, 1=flatland, 2=hill, 3=mountain
    elevated = elevation >= cfg_rivers.river_source_min_elevation

    # Check for adjacent ocean
    #pad = jnp.pad(is_ocean, ((1, 1), (1, 1)), constant_values=0)
    #neighbors = (
    #    pad[:-2, 1:-1] | pad[2:, 1:-1] |
    #    pad[1:-1, :-2] | pad[1:-1, 2:]
    #)
    #near_ocean = neighbors == 1

    is_coastline = compute_coastline_mask(landmask)
    
    candidates = (is_land & elevated) | (is_coastline & is_land)

    #import matplotlib.pyplot as plt
    ##plt.imshow(landmask)
    ##plt.show()
    #plt.imshow(is_coastline)
    #plt.show()
    ##plt.imshow(candidates)
    ##plt.show()

    #qqq
    
    num_cols = (candidates.shape[1] - border * 2) // cfg_game.num_player_cols
    num_rows = (candidates.shape[0] - border * 2) // cfg_game.num_player_rows

    
    # Establishing regions
    region_counter = 0
    col_start = border
    row_start = border

    global_rivers = []
    for region in range(cfg_game.num_players):
        local_rivers = []
        while len(local_rivers) < cfg_rivers.river_multiplier:
            row = jax.random.randint(
                key=key, shape=(1,),  minval=row_start, maxval=row_start + num_rows
            )
            col = jax.random.randint(
                key=key, shape=(1,),  minval=col_start, maxval=col_start + num_cols
            ) 
            potential_river = jnp.concatenate([row, col])
            key, _ = jax.random.split(key, 2)
            
            if len(local_rivers) > 0:
                for current_rivers in global_rivers:
                    dist = jnp.sqrt(((current_rivers - potential_river)**2).sum())
                    if dist <= cfg_rivers.l2_dist_threshold:
                        potential_river = jnp.array([0, 0])

            if candidates[potential_river[0], potential_river[1]]:
                local_rivers.append(potential_river[None])

        global_rivers.extend(local_rivers)
        region_counter += 1
        col_start += num_cols
        if region_counter >= cfg_game.num_player_cols:
            col_start = border
            row_start += num_rows
            region_counter = 0

    rivers = jnp.concatenate(global_rivers, axis=0)
    rivers = jnp.zeros_like(candidates).at[jnp.index_exp[rivers[:, 0],  rivers[:, 1]]].set(1)
    return rivers.astype(jnp.int32)  # 1 = source, 0 = non-source


# We need a way to translate from the previous river onto the current river
def translate_edge_over_hexes(previous_direction, new_temp_edge, prev_xy):
    """
    This function translates "as the river flows" to connected hex tiles, and returns the [x,y] position 
    of all tiles that the river touches.

    We will always follow the rightmost-hex are the "leader" (x,y)  and the leftmost-hex as the "adjacent" (x,y)

    Returns:
        translated_edge_leader, xy_leader, translated_edge_adj, xy_adj

    """
    # River flowing north
    if previous_direction == 0:
        # north initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 0
            xy_leader = prev_xy
            edge_adj = 3
            xy_adj = (prev_xy[0] - 1, prev_xy[1])
        
        # Turning left to NW
        elif new_temp_edge == 5:
            edge_leader = 2
            edge_adj = 5

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0] - 1, prev_xy[1] - 1)

            else:
                xy_leader = (prev_xy[0], prev_xy[1] - 1)
            
            xy_adj = (prev_xy[0] - 1, prev_xy[1])
        
        # Turning right to NE
        else:
            edge_leader = 1
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_leader = prev_xy
                xy_adj = (prev_xy[0] - 1, prev_xy[1] - 1)

            else:
                xy_leader = prev_xy
                xy_adj = (prev_xy[0], prev_xy[1] - 1)

    # River flowing NE
    elif previous_direction == 1:
        # NE initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 1
            xy_leader = prev_xy
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_adj = (prev_xy[0] - 1, prev_xy[1] - 1)
            else:
                xy_adj = (prev_xy[0], prev_xy[1] - 1)
        
        # Turning left to N
        elif new_temp_edge == 0:
            edge_leader = 0
            edge_adj = 3

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1] - 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1] - 1)
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1] - 1)
                xy_adj = (prev_xy[0], prev_xy[1] - 1)
        
        # Turning right to SE
        else:
            edge_leader = 2
            edge_adj = 5

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1] - 1)
                xy_adj = (prev_xy[0], prev_xy[1])   
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1] - 1)
                xy_adj = (prev_xy[0], prev_xy[1])

    # River flowing SE
    elif previous_direction == 2:
        # SE initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 2
            edge_adj = 5
            
            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1] + 1)
            else:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1] + 1)
        
        # Turning left to NE
        elif new_temp_edge == 1:
            edge_leader = 1
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1] + 1)
                xy_adj = (prev_xy[0], prev_xy[1])
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1] + 1)
                xy_adj = (prev_xy[0], prev_xy[1])
        
        # Turning right to S
        else:
            edge_leader = 0
            edge_adj = 3

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1] + 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1] + 1)
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1] + 1)
                xy_adj = (prev_xy[0], prev_xy[1] + 1)

    # River flowing S
    elif previous_direction == 3:
        # S initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 0
            edge_adj = 3

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0] + 1, prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1])
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1])
        
        # Turn left to SE
        elif new_temp_edge == 2:
            edge_leader = 2
            edge_adj = 5

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1] + 1)
            else:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1] + 1)

        # Turn right to SW
        else:
            edge_leader = 1
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0] - 1, prev_xy[1] + 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1] + 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])

    # River flowing SW
    elif previous_direction == 4:
        # SW initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 1
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0], prev_xy[1])

        # Turning left to S
        if new_temp_edge == 3:
            edge_leader = 0
            edge_adj = 3

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
        
        # Turning right to NW
        else:
            edge_leader = 2
            edge_adj = 5

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0] - 1, prev_xy[1] - 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1] - 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])

    # River flowing NW
    elif previous_direction == 5:
        # NW initial conditions
        if new_temp_edge == previous_direction:
            edge_leader = 2
            edge_adj = 5

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1] - 1)
                xy_adj = (prev_xy[0], prev_xy[1])
            else:
                xy_leader = (prev_xy[0] + 1, prev_xy[1] - 1)
                xy_adj = (prev_xy[0], prev_xy[1])

        # Turning left to SW
        elif new_temp_edge  == 4:
            edge_leader = 1
            edge_adj = 4

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0] - 1, prev_xy[1] + 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1] + 1)
                xy_adj = (prev_xy[0] - 1, prev_xy[1])

        # Turning right to N
        else:
            edge_leader = 0
            edge_adj = 3

            if prev_xy[1] % 2 == 0:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1])
            else:
                xy_leader = (prev_xy[0], prev_xy[1])
                xy_adj = (prev_xy[0] - 1, prev_xy[1])

    else:
        raise ValueError("Should never be here.")
                
    return edge_leader, xy_leader, edge_adj, xy_adj


class FlowDir(IntEnum):
    NO_DIR = -1
    N  = 0
    NE = 1
    SE = 2
    S  = 3
    SW = 4
    NW = 5

class FlowDirStart(IntEnum):
    NO_DIR = -1
    N  = 0
    NE = 1
    SE = 2
    
FLOW_MAP = {
    # key = flow direction: (x, y, side) in terms of viz map
    # 1. Move in hex-tile space (center to center) (x, y)
    # 2. Convert to side of moved tile center
    # [even row, odd row]
    # Here we always default to the RHS tile in terms of shared corners
    FlowDir.N:  [(-1, 0, 0), (0, -1, 0)],   
    FlowDir.NE: [(0, -1, 1), (1, -1, 1)],
    FlowDir.SE: [(0, 1, 2), (1, 1, 2)],  
    FlowDir.S:  [(0, 1, 3), (1, 1, 3)], 
    FlowDir.SW: [(0, 1, 4), (1, 1,  4)],
    FlowDir.NW: [(-1, -1, 5), (0, -1, 5)],   # the only way we can go NW is from N. (left, up, bottom-left) 
}

FLOW_MAP_START = {
    # This should bias us s.t. we never have to worry about the swap to leader from adj
    # key = flow direction: (x, y, side) in terms of viz map
    # 1. Move in hex-tile space (center to center) (x, y)
    # 2. Convert to side of moved tile center
    # [even row, odd row]
    # Here we always default to the RHS tile in terms of shared corners
    FlowDir.N:  [(-1, 0, 0), (0, -1, 0)],   
    FlowDir.NE: [(0, -1, 1), (1, -1, 1)],
    FlowDir.SE: [(0, 1, 2), (1, 1, 2)],  
}

# These turn splits are in terms of the river's flow, NOT the tile's sides
# The mental model for turns is how the actual river would flow. We'll rely on
# another function to actually map this to the game hexes
TURN_LEFT = {
    FlowDir.N: FlowDir.NW,
    FlowDir.NE: FlowDir.N,
    FlowDir.SE: FlowDir.NE,  # keeps me on same row
    FlowDir.S: FlowDir.SE,
    FlowDir.SW: FlowDir.S,
    FlowDir.NW: FlowDir.SW,  # keeps me on same row
}

TURN_RIGHT = {
    FlowDir.N: FlowDir.NE,
    FlowDir.NE: FlowDir.SE,  # keeps me on same row
    FlowDir.SE: FlowDir.S,
    FlowDir.S: FlowDir.SW,
    FlowDir.SW: FlowDir.NW,  # keeps me on same row
    FlowDir.NW: FlowDir.N,
}

def flow_preference_order(prev_dir):
    # If there was no river, then we can go in any direction
    if prev_dir == FlowDir.NO_DIR:
        return list(FlowDirStart)
    
    # Now we are restricted on directions we can go!
    # E.g., if we were going north, then we can either go NE or NW
    return [
        TURN_LEFT[prev_dir],
        TURN_RIGHT[prev_dir]
    ]

def do_river_improved(
    start_y: int,
    start_x: int,
    elevation: jnp.ndarray,
    landmask: jnp.ndarray,
    ocean_mask: jnp.ndarray,
    edge_rivers: np.ndarray,
    river_id: int,
    key: jnp.ndarray,
    max_length: int = 40,
) -> np.ndarray:
    """
    Improved Civ V-style river walker:
    - Allows flat flow
    - Prefers continuous direction
    - Encodes rivers as edge-indexed per-tile rivers (W, NW, NE)
    """
    H, W = elevation.shape
    elevation_np = np.array(elevation)
    landmask_np = np.array(landmask)
    ocean_np = np.array(ocean_mask)

    def is_valid(y, x):
        return 0 <= y < H and 0 <= x < W

    cy, cx = start_y, start_x
    visited = set()
    flow_dir = FlowDir.NO_DIR
    
    # At each step, a  givem fiver is touching two tiles. So let's make a list two-at-a-time
    # we can handle ocean tiles at a later point in time
    # We are going to treat the (x, y) in viz terms and square-grid terms, as this will be ultiamtely
    # applied to the gamestate tiles, which are represented as a matrix[y, x].
    # All matrices in the gamestate are in gridform[y, x] with vizform (x, y)
    best_edge = -1
    prev_x, prev_y = (-99, -99)
    
    same_turns_in_a_row = 0
    previous_turn = None
    prev_d = None
    best_flow = None
    best_value = float("inf")
    best_coords = None
    
    # Need to first initialize the river in a valid location
    # This ensures  we don't always turn left (early break) or turn right (exhaust options)
    _flow_preference_order = flow_preference_order(flow_dir)

    # Using jax.random.permutation seem to lead to very biased sampling. All rivers form effectively the same shape...
    _flow_preference_order_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(_flow_preference_order))
    _flow_preference_order_idx = [_flow_preference_order_idx, 1 - _flow_preference_order_idx]
    #_flow_preference_order_idx = jax.random.permutation(key=key, x=jnp.arange(start=0, stop=len(_flow_preference_order)))
    _flow_preference_order = np.array(_flow_preference_order)[np.array(_flow_preference_order_idx).astype(jnp.int32)].tolist()
    key, _ = jax.random.split(key, 2)
    
    for d in np.array(_flow_preference_order).tolist():
        if d == FlowDir.NO_DIR:
            continue

        _, _, edge = FLOW_MAP_START[d][cy % 2]

        # This call should handle when we accidentally select an edge that makes the "current tile" an adj and not leader tile
        edge, cxy, edge_adj, cxy_adj = translate_edge_over_hexes(previous_direction=edge, new_temp_edge=edge, prev_xy=(cx, cy))
        n_cx, n_cy = cxy
        cx_adj, cy_adj = cxy_adj

        if not is_valid(n_cy, n_cx):# or (n_cy, n_cx) in visited:
            continue
        if ocean_np[n_cy, n_cx] or ocean_np[cy_adj, cx_adj]:# #or not landmask_np[n_cy, n_cx]:
            continue

        neigh_elev = elevation_np[n_cy, n_cx]

        # This will effectively place the river in the first valid location
        if neigh_elev <= best_value:
            best_value = neigh_elev
            best_flow = d
            best_coords = (n_cy, n_cx)
            best_edge = edge
            best_edge_adj = edge_adj
            best_coords_adj = (cy_adj, cx_adj)

            break
    
    if best_coords is not None:
        edge_rivers[best_coords[0], best_coords[1], best_edge] = 1
        edge_rivers[best_coords_adj[0], best_coords_adj[1], best_edge_adj] = 1
        cy, cx = best_coords
        previous_turn = cy % 2
        flow_dir = best_flow
        prev_d = best_flow
    else:
        # Return early -- do nothing if there is no valid river start location
        return edge_rivers
    
    # Swirls occurs like e.g., RLLRLLRLLRLL, which forms a closed loop
    swirl_detection = ""
    swirl_pattern_primitive = ["L", "R"]
    l_swirl_pattern = "LRRLRR"
    l_swirl_pattern2 = "LLRLLR"
    l_swirl_pattern3 = "LLRRLLRR"
    l_swirl_pattern4 = "RLLRRLL"
    l_swirl_pattern5 = "RRRLRRR"

    r_swirl_pattern = "RLLRLL"
    r_swirl_pattern2 = "RRLRRL"
    r_swirl_pattern3 = "RRLLRRLL"
    r_swirl_pattern4 = "LRRLLRR"
    r_swirl_pattern5 = "LLLRLLL"


    for step in range(max_length):
        visited.add((cy, cx))
        current_elev = elevation_np[cy, cx]
        best_flow = None
        best_value = float("inf")
        best_coords = None
        l_or_r_idx = 0
        
        # This ensures  we don't always turn left (early break) or turn right (exhaust options)
        _flow_preference_order = flow_preference_order(flow_dir)
        
        # We only want to select a new direction n% of the time
        # Setting this number > 1 should mean that the rivers snake LRLRL... for as long as they can
        if jax.random.uniform(key=key, shape=()) < 1.8:
            # Using jax.random.permutation seem to lead to very biased sampling. All rivers form effectively the same shape...
            _flow_preference_order_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(_flow_preference_order))
            _flow_preference_order_idx = [_flow_preference_order_idx, 1 - _flow_preference_order_idx]
            _flow_preference_order = np.array(_flow_preference_order)[np.array(_flow_preference_order_idx).astype(jnp.int32)].tolist()

        else:
            # Else just reverse the order, so the river goes RLRLRL... most of the time
            _flow_preference_order = [_flow_preference_order[1], _flow_preference_order[0]]

        key, _ = jax.random.split(key, 2)

        for d in np.array(_flow_preference_order).tolist():
            _, _, edge = FLOW_MAP[d][cy % 2]
            
            edge, cxy, edge_adj, cxy_adj = translate_edge_over_hexes(previous_direction=prev_d, new_temp_edge=edge, prev_xy=(cx, cy))
            n_cx, n_cy = cxy
            cx_adj, cy_adj = cxy_adj

            if not is_valid(n_cy, n_cx):# or (n_cy, n_cx) in visited:
                l_or_r_idx += 1
                continue
            if ocean_np[n_cy, n_cx] or ocean_np[cy_adj, cx_adj]:# #or not landmask_np[n_cy, n_cx]:
                l_or_r_idx += 1
                continue

            if previous_turn == cy % 2:
                if same_turns_in_a_row == 2:
                    same_turns_in_a_row = 0
                    l_or_r_idx += 1
                    continue
                same_turns_in_a_row += 1

            # We also want to make sure that our rivers from a global perspective do not surround a hex
            if edge_rivers[n_cy, n_cx].sum() > 2 or edge_rivers[cy_adj, cx_adj].sum() > 2:
                l_or_r_idx += 1
                continue

            neigh_elev = elevation_np[n_cy, n_cx]

            if neigh_elev <= best_value:
                best_value = neigh_elev
                best_flow = d
                best_coords = (n_cy, n_cx)
                best_edge = edge
                best_edge_adj = edge_adj
                best_coords_adj = (cy_adj, cx_adj)
                previous_turn = cy % 2

                swirl_detection = swirl_detection + swirl_pattern_primitive[l_or_r_idx]
                #swirl_detection.append(swirl_pattern_primitive[l_or_r_idx])

                if (l_swirl_pattern in swirl_detection) or (r_swirl_pattern in swirl_detection):
                    return edge_rivers
                if (l_swirl_pattern2 in swirl_detection) or (r_swirl_pattern2 in swirl_detection):
                    return edge_rivers
                if (l_swirl_pattern3 in swirl_detection) or (r_swirl_pattern3 in swirl_detection):
                    return edge_rivers
                if (l_swirl_pattern4 in swirl_detection) or (r_swirl_pattern4 in swirl_detection):
                    return edge_rivers
                if (l_swirl_pattern5 in swirl_detection) or (r_swirl_pattern5 in swirl_detection):
                    return edge_rivers
                break
            else:
                l_or_r_idx += 1

        # If we make it all the way through the list without declaring best_coords, then the river can go no further
        if best_coords is None:
            return edge_rivers
        
        # Now we will let rivers combine into eachother, but to avoid the weird criss-cross patterns,
        # let's stop the current river from growing
        if edge_rivers[best_coords[0], best_coords[1], best_edge] == 1:
            edge_rivers[best_coords[0], best_coords[1], best_edge] = 1
            edge_rivers[best_coords_adj[0], best_coords_adj[1], best_edge_adj] = 1
            return edge_rivers

        edge_rivers[best_coords[0], best_coords[1], best_edge] = 1
        edge_rivers[best_coords_adj[0], best_coords_adj[1], best_edge_adj] = 1

        cy, cx = best_coords
        flow_dir = best_flow
        prev_d = best_flow

    return edge_rivers


def generate_lakes(cfg, key: jnp.ndarray, landmask: jnp.ndarray, coastal_mask: jnp.ndarray, river_mask: jnp.ndarray, terrain_type: jnp.ndarray) -> jnp.ndarray:
    """
    Generate lakes on the map.

    Args:
        cfg: Configuration object with lake parameters:
             - lake_seed_prob: probability of placing a lake seed per eligible tile (float)
             - lake_growth_prob: probability for lake growth into neighbors (float)
             - lake_growth_iters: how many iterations to try growing lakes (int)
        key: JAX random key.
        landmask: Binary mask (1 = land, 0 = ocean).
        coastal_mask: Binary mask (1 = coastal, 0 = non-coastal).
        river_mask: Binary mask (1 = river present, 0 = no river).

    Returns:
        lakes: Binary mask (1 = lake, 0 = no lake).
    """
    H, W = landmask.shape
    eligible = (landmask == 1) & (coastal_mask == 0) & (river_mask.sum(-1) == 0)
    rng1, rng2 = jax.random.split(key)

    # Seed initial lakes
    tile_chance = jax.random.uniform(rng1, shape=(H, W))
    lake_seed_mask = eligible & (tile_chance < cfg.lake_seed_prob)
    lakes = lake_seed_mask.astype(jnp.int32)

    # Hex neighbor offsets (pointy-topped layout)
    neighbor_offsets_even = jnp.array([
        [0, -1], [-1, 0], [-1, 1],
        [0, 1],  [1, 1],  [1, 0]
    ])
    neighbor_offsets_odd = jnp.array([
        [0, -1], [-1, -1], [-1, 0],
        [0, 1],  [1, 0],   [1, -1]
    ])

    def grow_lake(carry, _):
        rng, lakes_current = carry
        growth_noise = jax.random.uniform(rng, shape=(H, W, 6))

        def process_tile(y, x):
            offsets = jax.lax.cond(
                (y % 2) == 0,
                lambda _: neighbor_offsets_even,
                lambda _: neighbor_offsets_odd,
                operand=None
            )
            #neighbors = jnp.stack([
            #    (y + dy, x + dx)
            #    for dy, dx in offsets
            #])
            #neighbors = jnp.stack([
            #    jnp.array([y + dy, x + dx])
            #    for dy, dx in offsets
            #])

            #in_bounds = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < H) & \
            #            (neighbors[:, 1] >= 0) & (neighbors[:, 1] < W)
            #neighbors = neighbors[in_bounds]
            #neighbor_values = lakes_current[neighbors[:, 0], neighbors[:, 1]]

            neighbors = jnp.stack([
                jnp.array([y + dy, x + dx])
                for dy, dx in offsets
            ])
            in_bounds = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < H) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < W)

            # Mask out invalid neighbors
            masked_neighbors = jnp.where(
                in_bounds[:, None], 
                neighbors, 
                -1  # invalid entries
            )
            ny, nx = masked_neighbors[:, 0], masked_neighbors[:, 1]
            valid_mask = (ny >= 0) & (nx >= 0) & (ny < H) & (nx < W)

            neighbor_values = jnp.where(valid_mask, lakes_current[ny, nx], 0)
            grow = (neighbor_values > 0).astype(jnp.float32)
            grow = jnp.any(grow * (growth_noise[y, x, :len(grow)] < cfg.lake_growth_prob))
            return grow

        new_lakes = jax.vmap(
            lambda y: jax.vmap(lambda x: process_tile(y, x))(jnp.arange(W))
        )(jnp.arange(H))

        new_lakes = new_lakes & eligible
        lakes_next = jnp.where(new_lakes, 1, lakes_current)
        rng = jax.random.split(rng, num=2)[0]
        return (rng, lakes_next), None

    # Iteratively grow lakes
    (final_rng, lakes), _ = jax.lax.scan(
        grow_lake, (rng2, lakes), None, length=cfg.lake_growth_iters
    )

    lakes = lakes * (terrain_type != 3)

    return lakes.astype(jnp.uint8)


def generate_feature_noise(key, shape, scale=4.0):
    """
    Generate smooth fractal noise for feature placement.

    Args:
        key: JAX random key.
        shape: (height, width) tuple.
        scale: Controls clump size (higher = larger clumps).

    Returns:
        noise_map: (height, width) float array in [0.0, 1.0].
    """

    raw_noise = jax.random.uniform(key, shape)
    smooth_noise = gaussian_filter(raw_noise, sigma=scale)
    # Normalize to 0..1
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    return smooth_noise


def generate_features(cfg, key, terrain, elevation, rivers, lakes, latitude_map, landmask):
    """
    Generate features (forest, jungle, marsh, oasis, floodplains, ice) over the terrain map.

    Atoll
    Fallout
    Flood Plains
    Forest: usually form as groups/clumps
    Ice
    Jungle
    Lakes
    Marsh
    Oasis
    Rivers

    Args:
        cfg: Config object with feature probabilities and latitude cutoffs.
        key: JAX random key.
        terrain: (H, W) integer array of terrain types.
        elevation: (H, W) elevation map.
        rivers: (H, W) river mask (1 = river, 0 = no river).
        lakes: (H, W) lake mask (1 = lake, 0 = no lake).
        latitude_map: (H, W) float array of latitude per tile (0.0 = equator, 1.0 = pole).

    Returns:
        features: (H, W) integer map of features:
                  0 = none, 1 = forest, 2 = jungle, 3 = marsh, 4 = oasis, 5 = floodplains, 6 = ice.
    """

    H, W = terrain.shape
    rngs = jax.random.split(key, 6)

    # Feature IDs
    FEATURE_NONE, FEATURE_FOREST, FEATURE_JUNGLE, FEATURE_MARSH = 0, 1, 2, 3
    FEATURE_OASIS, FEATURE_FLOODPLAINS, FEATURE_ICE = 4, 5, 6
    
    features = jnp.zeros((H, W), dtype=jnp.uint8)
    
    hills = elevation == 2
    mountains = elevation == 3

    # Flood Plains: all desert tiles adjacent to a river
    flood_plains = (terrain == cfg.TERRAIN_DESERT) & (rivers.sum(-1) > 0) & ~lakes & ~mountains & ~hills
    features = jnp.where(flood_plains, FEATURE_FLOODPLAINS, features)


    # Oasis
    oasis_possible = (terrain == cfg.TERRAIN_DESERT) & ~flood_plains & ~lakes & ~mountains & ~hills
    oasis = (jax.random.uniform(key=key, shape=oasis_possible.shape) < cfg.feature_prob_oasis) & oasis_possible
    key, _ = jax.random.split(key, 2)
    features = jnp.where(oasis, FEATURE_OASIS, features)

    # Ice no ice for now

    # Marsh: on grasslands (https://civilization.fandom.com/wiki/Marsh_(Civ5))
    marsh_possible = (terrain == cfg.TERRAIN_GRASSLAND)
    marsh = (jax.random.uniform(key=key, shape=features.shape) < cfg.feature_prob_marsh) & marsh_possible & ~lakes & ~hills & ~mountains
    key, _ = jax.random.split(key, 2)
    features = jnp.where(marsh, FEATURE_MARSH, features)


    # Jungle
    jungle_noise = bilinear_upsample(jax.random.uniform(key, (H // cfg.jungle_noise_scale, W // cfg.jungle_noise_scale)), features.shape)
    jungle_noise = jungle_noise < cfg.feature_prob_jungle

    possible_jungle = jungle_noise & (latitude_map < cfg.jungle_max_latitude) & ~(terrain == cfg.TERRAIN_DESERT) & ~lakes & ~mountains * landmask
    possible_jungle = possible_jungle & ((terrain == cfg.TERRAIN_GRASSLAND) | (terrain == cfg.TERRAIN_PLAINS)) #& ~lakes * landmask
    key, _ = jax.random.split(key, 2)
    features = jnp.where(possible_jungle.astype(jnp.bool), FEATURE_JUNGLE, features)

    # Apparently jungles cannot be on grassland, so these tiles are converted to plains
    terrain = jnp.where((terrain == 1) & (features == FEATURE_JUNGLE), 2, terrain)

    # Forest
    forest_noise = bilinear_upsample(jax.random.uniform(key, (H // cfg.forest_noise_scale, W // cfg.forest_noise_scale)), features.shape)
    forest_noise = forest_noise < cfg.feature_prob_forest
    forest_noise = forest_noise & ((terrain == cfg.TERRAIN_GRASSLAND) | (terrain == cfg.TERRAIN_PLAINS) | (terrain == cfg.TERRAIN_TUNDRA))
    possible_forest = forest_noise & ~lakes & landmask & ~(features == FEATURE_JUNGLE) & ~mountains
    features = jnp.where(possible_forest.astype(jnp.bool), FEATURE_FOREST, features)

    return features, terrain


def generate_latitude_map(height, width):
    """
    Radiating latitude map: 0.0 at the equator (middle row), 1.0 at both poles.
    Distance radiates outward from the equator symmetrically.

    Args:
        height: number of rows (Y dimension).
        width: number of columns (X dimension).

    Returns:
        latitude_map: (height, width) array with values from 0.0 (equator) to 1.0 (poles).
    """
    equator_row = (height - 1) / 2.0
    latitudes = jnp.abs(jnp.arange(height) - equator_row) / equator_row
    latitude_map = jnp.repeat(latitudes[:, None], width, axis=1)
    return latitude_map


def compute_fertility(
        terrain: jnp.ndarray,
        elevation: jnp.ndarray,
        feature: jnp.ndarray,
        river: jnp.ndarray,
        lake: jnp.ndarray,
        ocean: jnp.ndarray,
        coastal: jnp.ndarray,
        check_for_coastal_land: bool = True,):
    """
    The highest fertility comes from plains or grassland with hills, near rivers, floodplains, or oasis.

    Desert without a river = terrible.

    Tundra slightly better than desert, but worse than plains/grassland.

    Mountains and snow are almost always bad, except for adjacency bonuses.

    Compute the fertility score of each tile according to LekMod/Civ V logic.

    Args:
        terrain: (H, W) array. 0=plains, 1=grassland, 2=tundra, 3=desert, 4=snow
        elevation: (H, W) array. 0=flatland, 1=hill, 2=mountain
        feature: (H, W) array. 0=none, 1=forest, 2=jungle, 3=marsh, 4=oasis, 5=floodplains, 6=ice
        river: (H, W) array. 1 if adjacent to river, else 0.
        lake: (H, W) array. 1 if fresh water (lake), else 0.
        ocean: (H, W) array. 1 if ocean, else 0.
        coastal: (H, W) array. 1 if coastal land, else 0.
        check_for_coastal_land: whether to apply coastal bonus.

    Returns:
        fertility: (H, W) array of fertility scores.
    """
    # Base fertility by terrain
    terrain_scores = jnp.array([2, 3, 4, -1, 1, -2])  # ocean, grassland, plains, desert, tundra, snow
    terrain_score = terrain_scores[terrain]
    

    # Plot type modifiers
    is_mountain = (elevation == 3)
    is_hill = (elevation == 2)
    fertility = jnp.where(is_mountain, -1, terrain_score)
    fertility = jnp.where(is_hill, fertility + 2, fertility)

    # Feature modifiers
    is_forest = (feature == 1)
    is_jungle = (feature == 2)
    is_marsh = (feature == 3)
    is_oasis = (feature == 4)
    is_floodplains = (feature == 5)
    is_ice = (feature == 6)

    # Apply feature bonuses/penalties
    fertility = jnp.where(is_oasis, 4, fertility)
    fertility = jnp.where(is_floodplains, 4, fertility)
    fertility = jnp.where(is_ice, fertility - 1, fertility)
    fertility = jnp.where(is_jungle, fertility - 1, fertility)
    fertility = jnp.where(is_marsh, fertility - 2, fertility)
    # Forest bonus is 0 (no-op)

    # Water bonuses
    has_fresh_water = (river.sum(-1) > 0) | (lake.astype(jnp.bool) == 1)
    fertility = jnp.where(has_fresh_water, fertility + 1, fertility)

    # Coastal land bonus. The source game is + 2, but we get way too many coastal starts...
    #fertility = jnp.where(coastal, fertility, fertility)
    return fertility



def assign_starting_locations(terrain, elevation, features, river, lake, ocean, coastal, key):
    """
    Begins with StartPLotSystem (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBMapGeneratorRectangular.lua#L837)
    """
    # META-STEP (A) GenerateRegion (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L1875)
    # (1) measure fertility (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L1173)
    fertility = compute_fertility(terrain, elevation, features, river, lake, ocean, coastal)

    # (2) use fertility to divide into regions (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L1430)
    # The original code here calls some recursive algorithmn to sub-divide the landmass into regions of roughly equal fertility.
    # Instead of doing that, we're going to rely on our ability to have previously generated a landmass + features + terrain that are
    # roughly equally good across the landmass. If so, then we can just divide the landmass into 6 chunks (for small map).
    H, W = fertility.shape
    regions = []
    region_width = W // 3
    region_height = H // 2
    
    # Here x=col, y=row
    for row_idx in range(2):
        for col_idx in range(3):
            x_start = col_idx * region_width
            x_end = (col_idx + 1) * region_width
            y_start = row_idx * region_height
            y_end = (row_idx + 1) * region_height
            regions.append((y_start, y_end, x_start, x_end))
    
    # (3) Choose location (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L4308)
    def random_topk_choice(fertility_map, k, key):
        # Flatten the fertility map
        flat_fertility = fertility_map.flatten()
        
        # Get the top-k indices
        topk_values, topk_indices = jax.lax.top_k(flat_fertility, k)

        # Sample one index randomly from top-k
        sample_idx = jax.random.randint(key, (), 0, k)
        chosen_flat_idx = topk_indices[sample_idx]

        # Convert flat index back to (row, col)
        H, W = fertility_map.shape
        row = chosen_flat_idx // W
        col = chosen_flat_idx % W
        return (row.item(), col.item())

    def argmax_2d(arr: jnp.ndarray):
        flat_index = jnp.argmax(arr)
        rows, cols = arr.shape
        return ((flat_index // cols).item(), (flat_index % cols).item())
    
    # We need to make sure that the settler never spawns (1) in the ocean, (2) in a lake, or (3) on a mountain
    cannot_settle_mask = ocean | lake | (elevation == 3)
    settler_rowcols = []
    subregion_stats = {"mean_fertility": []}

    for i, region in enumerate(regions):
        subregion = fertility[region[0]: region[1], region[2]: region[3]]
        subregion_stats["mean_fertility"].append(subregion.mean())

        # In the small map with a 2-row x 3-col pangea, each chunk is ~21x22
        # We want to bias starting to the middle of each of these regions, so let's do that
        subregion_height_center = subregion.shape[0] // 2
        subregion_height_quartered = subregion.shape[0] // 4
        subregion_width_center = subregion.shape[1] // 2
        subregion_width_quartered = subregion.shape[1] // 4

        subregion_height_left = subregion_height_center - subregion_height_quartered
        subregion_height_right = subregion_height_center + subregion_height_quartered
        subregion_width_left = subregion_width_center - subregion_width_quartered
        subregion_width_rigt = subregion_width_center + subregion_width_quartered

        subregion = subregion[subregion_height_left: subregion_height_right, subregion_width_left: subregion_width_rigt]

        subregion_cannot_settle_mask = cannot_settle_mask[region[0]: region[1], region[2]: region[3]]
        subregion_cannot_settle_mask = subregion_cannot_settle_mask[subregion_height_left: subregion_height_right, subregion_width_left: subregion_width_rigt]
        subregion = jnp.where(subregion_cannot_settle_mask, -999, subregion)
        #point = argmax_2d(subregion)
        point = random_topk_choice(subregion, k=15, key=key)
        key, _ = jax.random.split(key, 2)

        # Now we need to project this back into the global row,col space. 
        local_r, local_c = point
        global_r = region[0] + subregion_height_center - subregion_height_quartered + local_r
        global_c = region[2] + subregion_width_center - subregion_width_quartered + local_c
        settler_rowcols.append((global_r, global_c))
    
    return fertility, settler_rowcols, subregion_stats


def random_true_index(rng_key, mask):
    # Step 1: Get the indices where mask is True
    true_indices = jnp.argwhere(mask)  # shape (num_true, 2)

    # Step 2: Choose a random index among those
    num_true = true_indices.shape[0]
    chosen_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=num_true)

    # Step 3: Return the selected (row, col) pair
    return true_indices[chosen_idx]


def place_natural_wonders(num_natural_wonders, settler_start_rowcols, landmask, elevation_map, terrain, rivers, lakes, features, nw_min_distance_from_start, key):
    """
    Based on (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L7645)
    
    These spawns can change: elevation, features 
    """
    # Natural wonders have different elligibellity 
    # Looks like the source code loops over every tile in the map first and checks if the given tile
    # can house **ANY** natural wonder. This feels overly-specific? WHat about ocean-based?
    
    # META: looks like these two checks ensure the natural wonder doesn't spawn in any players'
    # capital?
    # (1) filters our settler starts

    # (2) look in all adjacent hexes (the ring around the given hex)

    # (3) A bunch of requirement checks on tiles depending on the NW type (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L6844). Really need to find that xml file

    # (4) Now we should have a filtered list of which NW can be spawned on our given map
    # They then sort 
    nw_placements = jnp.zeros_like(landmask, dtype=jnp.int32)
    num_placed = 0
    nw_chosen = []
    inner_settler_start_rowcols = deepcopy(settler_start_rowcols)
    while num_placed < num_natural_wonders:
        found = False
            
        _non_duplicate_nw = False

        while not _non_duplicate_nw:
            nw_to_place_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(ALL_NATURAL_WONDERS)).item()
            key, _ = jax.random.split(key, 2)

            if nw_to_place_idx in nw_chosen:
                continue
            else:
                _non_duplicate_nw = True

        nw_to_place = ALL_NATURAL_WONDERS[nw_to_place_idx]
        nw_chosen.append(nw_to_place_idx)

        nw_placement_criteria = NW_SPAWN_CRITERIA[nw_to_place]
        
        # We start off with a map of all 1s (can place at 1), then narrow down based on bools in 
        # the spawn criteria
        can_place = jnp.ones_like(nw_placements, dtype=jnp.bool)
        spawns_ocean = nw_placement_criteria["ocean_land"][0]
        if spawns_ocean:
            can_place = can_place & ~landmask.astype(jnp.bool)
        else:
            can_place = can_place & landmask.astype(jnp.bool)

        # If the NW can spawn in the ocean, let's just place it and then end here
        # This assumes that all NWs that can spawn in the ocean ONLY spawn in the ocean,
        # which may not be true in general.
        if spawns_ocean:
            while not found:
                rowcol = random_true_index(key, can_place)
                key, _ = jax.random.split(key, 2)

                # I don't think there are any other issues?
                found = True
        
        else:
            # Now narrowing down based on all other factors
            for i, terrain_can_bool in enumerate(nw_placement_criteria["terrain"]):
                if terrain_can_bool == 1:
                    continue
                else:
                    can_place = can_place & ~(terrain == i)

            for i, feature_can_bool in enumerate(nw_placement_criteria["features"]):
                if feature_can_bool:
                    continue
                else:
                    can_place = can_place & ~(features == i)

            for i, elevation_can_bool in enumerate(nw_placement_criteria["elevation"]):
                if elevation_can_bool:
                    continue
                else:
                    can_place = can_place & ~(elevation_map == i)
            
            can_place = can_place & ~lakes
            while not found:
                rowcol = random_true_index(key, can_place)
                key, _ = jax.random.split(key, 2)

                # We want to enforce the distance away from the players' starts
                nw_to_start_dist = jnp.sqrt(((jnp.array(inner_settler_start_rowcols) - rowcol) ** 2).sum(-1)).min()
                if nw_to_start_dist > nw_min_distance_from_start:
                    found = True
        
        nw_placements = nw_placements.at[rowcol[0], rowcol[1]].set(nw_to_place_idx + 1)

        # Now we need set elevation -> flat 
        elevation_map = elevation_map.at[rowcol[0], rowcol[1]].set(1)
        # feature -> no feature
        features = features.at[rowcol[0], rowcol[1]].set(0)

        num_placed += 1

        # Let's also make sure the NWs don't spawn too close to one another...
        inner_settler_start_rowcols.append(rowcol)
        
    return nw_placements, elevation_map, features


def place_resources_and_cs(settler_start_rowcols, landmask, elevation_map, terrain, edge_rivers, lakes, features, nw_placements, cfg, key):
    """
    Call begins here (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBMapGeneratorRectangular.lua#L867)
    And is actually called here (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L14148)

    Description from the source code:
    -- Luxury resources are placed in relationship to Regions, adapting to the
	-- details of the given map instance, including number of civs and city 
	-- states present. At Jon's direction, Luxuries have been implemented to
	-- be diplomatic widgets for trading, in addition to sources of Happiness.
	--
	-- Strategic and Bonus resources are terrain-adjusted. They will customize
	-- to each map instance. Each terrain type has been measured and has certain 
	-- resource types assigned to it. You can customize resource placement to 
	-- any degree desired by controlling generation of plot groups to feed in
	-- to the process. The default plot groups are terrain-based, but any
	-- criteria you desire could be used to determine plot group membership.
	-- 
	-- If any default methods fail to meet a specific need, don't hesitate to 
	-- replace them with custom methods. I have labored to make this new 
	-- system as accessible and powerful as any ever before offered.
    """
    # (1)  (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L10081)
    # Looks like there are 1-9 region types, and luxuries are related to type
    # Looks like here (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L2387) regions typed are defined by the number of tile types in a given area of the map?
    """
	-- REGION TYPES
	-- 0. Undefined
	-- 1. Tundra
	-- 2. Jungle
	-- 3. Forest
	-- 4. Desert
	-- 5. Hills
	-- 6. Plains
	-- 7. Grassland
	-- 8. Hybrid
	-- 9. Marsh
    """

    # The luxes that can be sampled for the region are determined by some thresholds on the 
    # percentage of the area of each chunk being certain terrain or having certain features.
    # See (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L2470)

    # This is the same way we divided the map into chunks to sample the starting regions.
    # So, the settler indexes should all be within these chunks in the same order.
    H, W = landmask.shape
    regions = []
    region_width = W // 3
    region_height = H // 2
    
    # Here x=col, y=row
    for row_idx in range(2):
        for col_idx in range(3):
            x_start = col_idx * region_width
            x_end = (col_idx + 1) * region_width
            y_start = row_idx * region_height
            y_end = (row_idx + 1) * region_height
            regions.append((y_start, y_end, x_start, x_end))

    ocean_lux = deepcopy(OCEAN_LUX)
    land_lux = deepcopy(LAND_LUX)
    all_possible_lux = deepcopy(ocean_lux) + deepcopy(land_lux)
    chosen_lux_per_player = []
    chosen_lux_for_cs = []
    chosen_random_lux = []
    player_civ_is_coastal = []

    for i in range(6):
        region = regions[i]
        settler_rc = settler_start_rowcols[i]

         # We need to be a little careful. If a civ settler spawns on the coast, then we must ensure that 
        # they have a change to get a sea-based luxury set as their regional, and no one else
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=settler_rc[0], col=settler_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])

        coastal = (~landmask[hexes_surrounding[:, 0], hexes_surrounding[:, 1]]).sum() > 0

        if coastal:
            player_civ_is_coastal.append(True)
            # If the settler is coastal, then we should select coastal lux as the regions
            # Iff there are "enough" coastal tiles. The source code says >= 8
            # We can get all tiles in the city's 3-tile reach radius by calling 
            # get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol three times
            hexes_surrounding = jax.vmap(
                get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                in_axes=(0, 0, None, None),
            )(
                hexes_surrounding[:, 0],
                hexes_surrounding[:, 1],
                landmask.shape[0],
                landmask.shape[1]
            ).reshape(-1, 2)

            hexes_surrounding = unique_rows(hexes_surrounding)

            hexes_surrounding = jax.vmap(
                    get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                    in_axes=(0, 0, None, None),
            )(
                hexes_surrounding[:, 0],
                hexes_surrounding[:, 1],
                landmask.shape[0],
                landmask.shape[1]
            ).reshape(-1, 2)

            hexes_surrounding = unique_rows(hexes_surrounding)
            
            coastal_lux = (~landmask[hexes_surrounding[:, 0], hexes_surrounding[:, 1]]).sum() >= 8

            
        else:
            player_civ_is_coastal.append(False)
            coastal_lux = False

        if coastal_lux:
            lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=len(ocean_lux)).item()
            key, _ = jax.random.split(key, 2)
            chosen_lux_per_player.append(ocean_lux[lux_int])
            del ocean_lux[ocean_lux.index(ocean_lux[lux_int])]

        else:
            # Here we follow the LekMod logic 
            # (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L2470)
            region_terrain = terrain[region[0]: region[1], region[2]: region[3]]
            region_features = features[region[0]: region[1], region[2]: region[3]]
            region_elevation = elevation_map[region[0]: region[1], region[2]: region[3]]
            chunk_area = region_features.shape[0] * region_features.shape[1]

            tundra_bool = ((region_terrain == 4).sum() + (region_terrain == 5).sum()) / chunk_area > 0.1
            jungle_bool = (region_features == 2).sum() / chunk_area > 0.12
            jungle_mix_bool = (
                    ((region_features == 2).sum() / chunk_area > 0.10) & 
                    (((region_features == 1).sum() + (region_features == 2).sum()) / chunk_area > 0.24)
            )
            forest_bool = (region_features == 1).sum() / chunk_area > 0.21
            desert_bool = (region_terrain == 3).sum() / chunk_area > 0.15
            wetlands_bool = (
                ((region_features == 3).sum() / chunk_area > 0.11) or ((region_features == 3).sum() >= 8)
            )
            hills_bool = (region_elevation == 2).sum() / chunk_area > 0.37
            grass_bool = (
                ((region_terrain == 1).sum() / chunk_area > 0.2) & ((region_terrain == 1).sum() * 0.7 > (region_terrain == 2).sum())
            )
            plain_bool = (
                ((region_terrain == 2).sum() / chunk_area > 0.27) & ((region_terrain == 2).sum() * 0.8 > (region_terrain == 1).sum())
            )
            hybrid_bool = (
                (region_terrain == 1).sum() + (region_terrain == 2).sum() + (region_terrain == 3).sum() + (region_terrain == 4).sum() + (region_terrain == 5).sum() + (region_elevation == 2).sum() + (region_elevation == 3).sum()
            ) / chunk_area > 0.8

            if tundra_bool:
                regional_bias = "tundra"
            elif jungle_bool:
                regional_bias = "jungle"
            elif jungle_mix_bool:
                regional_bias = "jungle"
            elif forest_bool:
                regional_bias = "forest"
            elif desert_bool:
                regional_bias = "desert"
            elif hills_bool:
                regional_bias = "hills"
            elif grass_bool:
                regional_bias = "grass"
            elif plain_bool:
                regional_bias = "plains"
            elif wetlands_bool:
                regional_bias = "marsh"
            else:
                # If we're here, we should pick the regional bias that has the most options
                # remaining, so we minimize the likelihood to stealing someone else's lux?
                regional_bias = "grass"

            print(f"\tRegional bias: {regional_bias}")

            possible_regional_luxes = [
                x for x in LUX_BIAS_TABLE 
                if regional_bias in LUX_BIAS_TABLE[x]["regional_bias"] 
                and x not in chosen_lux_per_player
                and x not in chosen_lux_for_cs
            ]

            num_possible = len(possible_regional_luxes)
            print(f"\tNumber luxuries possible: {num_possible}")
            lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=num_possible).item()
            key, _ = jax.random.split(key, 2)
            chosen_lux = possible_regional_luxes[lux_int]
            chosen_lux_per_player.append(chosen_lux)
            print(f"\tChosen regional: {chosen_lux}")

            possible_regional_luxes = [
                x for x in LUX_BIAS_TABLE 
                if regional_bias in LUX_BIAS_TABLE[x]["regional_bias"]
                and x not in chosen_lux_per_player
                and x not in chosen_lux_for_cs
            ]

            num_possible = len(possible_regional_luxes)
            lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=num_possible).item()
            key, _ = jax.random.split(key, 2)
            chosen_lux = possible_regional_luxes[lux_int]
            chosen_lux_for_cs.append(chosen_lux)
            print(f"\tCity State luxury: {chosen_lux}")

    # Need to also sample secondary luxuries. It looks like secondary luxuries can be 
    # assigned to no more than 3 regions
    # Choosing other random luxes to have on map (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L10045). The number depends on the map size
    # Looks like for small is 12
    # According to this table (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L10045), small map gets 12 "random luxuries". These look like they *CAN* be the same lux assined to the CS
    # THen we have the "remaining" luxes, which are then randomly picked.
    for _ in range(cfg.num_random_lux):
        possible_randoms = [
            x for x in all_possible_lux
            if x not in chosen_lux_per_player
            and x not in chosen_random_lux
        ]
        num_possible = len(possible_randoms)
        lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=num_possible).item()
        key, _ = jax.random.split(key, 2)
        chosen_lux = possible_randoms[lux_int]
        chosen_random_lux.append(chosen_lux)

    print(f"\tExtra luxuries: {chosen_random_lux}")

    # Great, so the breakdown is like this:
    # Each region's player civ gets one regional, two CS (each with their own regional)
    # and then three random lux. 
    # So let's assign some of the random lux to each of the regions. At most, a random lux
    # can appear in three regions (including cs regional)
    # We can also take advantage of this loop to of give each player cic their secondary cap lux 
    # and also ensure that it is *not* one of the random luxes in their region
    random_lux_assignments = []
    secondary_lux_assignment = []
    random_lux_counts = {k: 0 for k in chosen_random_lux}
    for player_iterator in range(cfg.num_players):
        _inner_random_lux = []
        for _unused in range(4):
            possible_randoms = [
                k for k, v in random_lux_counts.items()
                if v < 3
                and k not in _inner_random_lux
            ]
            num_possible = len(possible_randoms)
            lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=num_possible).item()
            key, _ = jax.random.split(key, 2)
            chosen_lux = possible_randoms[lux_int]
            random_lux_counts[chosen_lux] += 1
            _inner_random_lux.append(chosen_lux)

        # Let's add the secondary lux as one of the luxuries **not** in the current region's randoms
        possible_secondaries = [x for x in chosen_random_lux if x not in _inner_random_lux]
        secondary_lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=len(possible_secondaries)).item()
        key, _ = jax.random.split(key, 2)
        chosen_secondary_lux = possible_secondaries[secondary_lux_int]
        secondary_lux_assignment.append(chosen_secondary_lux)
        random_lux_assignments.append(_inner_random_lux)

        print(f"\tExtras for player {player_iterator}: {_inner_random_lux}")
    print(f"\tExtras distribution: {random_lux_counts}")


    # DONE ASSIGNING LUXURY ROLES :)
    # Now placing cs (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L8395)
    # Because we are replicating lekmod's pangea rectangular, there are no "uninhabited" regions
    # So we're here (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L7812)
    # Our process here is potentially a big departure from the actual game. The actual game does some
    # re-balancing of regions based on landmass, region types, etc etc. We're going to keep it simple and
    # just assign two cs per region randomly.
    final_cs_lux = []
    cs_lux_distribution = {k: 0 for k in chosen_lux_for_cs}
    for _ in range(cfg.num_cs):
        possible_cs_lux = [
            k for k, v in cs_lux_distribution.items()
            if v < 3
        ]
        num_possible = len(possible_cs_lux)
        lux_int = jax.random.randint(key=key, shape=(), minval=0, maxval=num_possible).item()
        key, _ = jax.random.split(key, 2)
        chosen_lux = possible_cs_lux[lux_int]
        cs_lux_distribution[chosen_lux] += 1
        final_cs_lux.append(chosen_lux)

    # time to actually place the physical location of the cs city row/col
    region_counter = 0
    _region_cs_holder = []
    _global_cs_holder = []
    cs_ownership_map = jnp.zeros_like(features, dtype=jnp.int32)
    for i in range(cfg.num_cs):
        found = False
        region_boundaries = regions[region_counter]
        while not found:
            _row = jax.random.randint(key=key, shape=(), minval=region_boundaries[0], maxval=region_boundaries[1])
            key, _ = jax.random.split(key, 2)
            _col = jax.random.randint(key=key, shape=(), minval=region_boundaries[2], maxval=region_boundaries[3])
            key, _ = jax.random.split(key, 2)

            if landmask[_row, _col] == 0:
                continue

            if elevation_map[_row, _col] == 3:
                continue

            if lakes[_row, _col] == 1:
                continue

            if features[_row, _col] == 4:
                continue

            if nw_placements[_row, _col] > 0:
                continue

            if jnp.sqrt(((jnp.array(settler_start_rowcols[region_counter]) - jnp.array([_row, _col]))**2).sum()) < cfg.min_cs_distance_to_player:
                continue
            
            if len(_region_cs_holder) > 0:
                if jnp.sqrt(((jnp.array(_region_cs_holder[0]) - jnp.array([_row, _col]))**2).sum()) < cfg.min_cs_distance_to_cs:
                    continue

            found = True
        
        _region_cs_holder.append((_row, _col))
        _global_cs_holder.append((_row.item(), _col.item()))

        # Settling on a tile removes forest/jungle/etc
        features = features.at[_row, _col].set(0)
        cs_ownership_map = cs_ownership_map.at[_row, _col].set(i + 1)

        cs_surround_hex = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(_row, _col, max_rows=features.shape[0], max_cols=features.shape[1])
        for _hex in cs_surround_hex:
            cs_ownership_map = cs_ownership_map.at[_hex[0], _hex[1]].set(i + 1)

        if i % 2 != 0:
            region_counter += 1
            _region_cs_holder = []

    _global_cs_three_rings = []
    _global_player_three_rings = []
    for _cs_rowcol in _global_cs_holder:
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=_cs_rowcol[0], col=_cs_rowcol[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        for _hex in hexes_surrounding:
            _global_cs_three_rings.append([_hex[0].item(), _hex[1].item()])

    for settler_rc in settler_start_rowcols:
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=settler_rc[0], col=settler_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        for _hex in hexes_surrounding:
            _global_cs_three_rings.append([_hex[0].item(), _hex[1].item()])


    # Now it is time to place them luxuries!
    # Is this 4 lux in cap? (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L11036) 3 regional + 1 extra. Then there may be adjustments if the total fertility (or other)
    # stat is deemed to produce a "poor" start.

    # Also looks like multiple CS can have the same starting lux
    # (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L11122)
    # Let's start with regionals 
    # (1) map with all resource types as ints
    # (2) map with resource category (1=lux, 2=strat)
    all_resource_map = jnp.zeros_like(features, dtype=jnp.int32)
    all_resource_quantity_map = jnp.zeros_like(all_resource_map)
    resource_type_map = jnp.zeros_like(all_resource_map)

    for i, region in enumerate(regions):
        # There should be 3 regional luxuries within three workable tiles from the cap
        settler_rc = settler_start_rowcols[i]
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=settler_rc[0], col=settler_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)
        regional_luxury = chosen_lux_per_player[i]

        regional_luxury_idx = RESOURCE_TO_IDX[regional_luxury]
        bias_type = LUX_BIAS_TABLE[regional_luxury]["terrain_bias"][0]

        # Let's get our hex-tile preferences for sampling
        picked_rowcols = translate_terrain_bias_to_tile_samples(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=hexes_surrounding, settler_rc=settler_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=3)
        key, _ = jax.random.split(key)

        for rc in picked_rowcols:
            all_resource_map = all_resource_map.at[rc[0], rc[1]].set(regional_luxury_idx)
            all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
            resource_type_map = resource_type_map.at[rc[0], rc[1]].set(1)

        # Now we can place the remainder of the regionals
        # According to this table (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L11059) there should be 4 extra regionals in a small map with 6 players civs
        # These can go anywhere in the region
        all_rows_dummy = jnp.repeat(jnp.arange(0, landmask.shape[0]).reshape(-1, 1), landmask.shape[1], axis=-1)
        all_cols_dummy = jnp.repeat(jnp.arange(0, landmask.shape[1]).reshape(1, -1), landmask.shape[0], axis=0)

        this_regions_rows = all_rows_dummy[region[0]: region[1], region[2]: region[3]]
        this_regions_cols = all_cols_dummy[region[0]: region[1], region[2]: region[3]]
        
        # Here let's remove from the eligible rowcols anything within three rings of the settler spawn as well as the 
        # CS spawns :)
        this_regions_rowcols = jnp.concatenate([this_regions_rows.reshape(-1)[:, None], this_regions_cols.reshape(-1)[:, None]], axis=-1)
        _this_regions_rowcols = []
        for regions_rowcol in this_regions_rowcols:
            converted_rowcol = [regions_rowcol[0].item(), regions_rowcol[1].item()]
            if converted_rowcol in _global_cs_three_rings:
                continue
            if converted_rowcol in _global_player_three_rings:
                continue
            _this_regions_rowcols.append(converted_rowcol)
        this_regions_rowcols = jnp.array(_this_regions_rowcols)
        # Finally need to shuffle so the regionals are unlikely to end up near eachother...
        this_regions_rowcols = jax.random.permutation(key=key, x=this_regions_rowcols, axis=0)
        key, _ = jax.random.split(key, 2)

        # Let's loop through this list of rowcols and remove anything within 3 tiles of the cs
        print("\tScattering regionals...")
        picked_rowcols = translate_terrain_bias_to_tile_samples(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=this_regions_rowcols, settler_rc=settler_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=5)

        for rc in picked_rowcols:
            all_resource_map = all_resource_map.at[rc[0], rc[1]].set(regional_luxury_idx)
            all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[0]].set(1)
            resource_type_map = resource_type_map.at[rc[0], rc[1]].set(1)

        # Secondary time!
        secondary_lux_idx = RESOURCE_TO_IDX[secondary_lux_assignment[i]]
        bias_type = LUX_BIAS_TABLE[secondary_lux_assignment[i]]["terrain_bias"][0]
        
        print("\tScattering extras...")
        picked_rowcols = translate_terrain_bias_to_tile_samples(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=hexes_surrounding, settler_rc=settler_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=1)
        key, _ = jax.random.split(key)

        for rc in picked_rowcols:
            all_resource_map = all_resource_map.at[rc[0], rc[1]].set(secondary_lux_idx)
            all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[0]].set(1)
            resource_type_map = resource_type_map.at[rc[0], rc[1]].set(1)

        # Now we can do the same thing with the "random" luxuries using the same data
        # According to this function, there should be 80 luxuries in the world **not** counting the extra cap lux and 5 "random" luxury
        # (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L11079)
        this_regions_randoms = random_lux_assignments[i]
        for random_lux in this_regions_randoms:
            random_luxury_idx = RESOURCE_TO_IDX[random_lux]
            bias_type = LUX_BIAS_TABLE[random_lux]["terrain_bias"][0]

            num_random_lux_to_place = jax.random.randint(key=key, shape=(), minval=1, maxval=3).item()
            key, _ = jax.random.split(key, 2)
            picked_rowcols = translate_terrain_bias_to_tile_samples(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=this_regions_rowcols, settler_rc=settler_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=num_random_lux_to_place)

            for rc in picked_rowcols:
                all_resource_map = all_resource_map.at[rc[0], rc[1]].set(random_luxury_idx)
                all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
                resource_type_map = resource_type_map.at[rc[0], rc[1]].set(1)


    # Now we can place the cs luxes in the first ring(?)
    # There are now five luxes for us to loop through
    # CS lux placement routine (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L11254)
    num_cs_luxes = len(chosen_lux_for_cs)
    for i in range(cfg.num_cs):
        _cs_rowcol = _global_cs_holder[i]
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=_cs_rowcol[0], col=_cs_rowcol[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = unique_rows(hexes_surrounding)
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)


        cs_lux = chosen_lux_for_cs[i % num_cs_luxes]
        cs_lux_idx = RESOURCE_TO_IDX[cs_lux]
        bias_type = LUX_BIAS_TABLE[cs_lux]["terrain_bias"][0]
        picked_rowcols = translate_terrain_bias_to_tile_samples(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=hexes_surrounding, settler_rc=_cs_rowcol, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=1)

        for rc in picked_rowcols:
            all_resource_map = all_resource_map.at[rc[0], rc[1]].set(cs_lux_idx)
            all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
            resource_type_map = resource_type_map.at[rc[0], rc[1]].set(1)
    

    print(f"\tTotal luxuries placed: {(all_resource_map > 0).sum()}")
    
    # Amazing, now we can get to placing the strategic and bonus resources :)
    # (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L13831)
    # (1) place the strat-balance based resources
    # I'm going to do this slightly differently than the source code, just for my own sanity...
    # A "strat balance" start guarantees (i) iron, (ii) horses, (iii) coal, (iv) uranium, (v) oil, (vi) aluminium in the cap
    # assuming that the player civ does not move from their initial settler position, so let's do that 
    
    strategic_resources_with_quantity = {
        "iron": [3, 8],
        "horses": [3, 6],
        "coal": [4, 9],
        "uranium": [1, 3],
        "oil": [3, 8],
        "aluminium": [4, 9],
    }

    for i, region in enumerate(regions):
        settler_rc = settler_start_rowcols[i]
        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=settler_rc[0], col=settler_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)

        hexes_surrounding = jax.vmap(
                get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)

        for strat_res in strategic_resources_with_quantity:
            strat_res_id = RESOURCE_TO_IDX[strat_res]
            bias_type = STRATEGIC_BIAS_TABLE[strat_res]["terrain_bias"]
            picked_rowcols = translate_terrain_bias_to_tile_samples_strategic(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=hexes_surrounding, settler_rc=settler_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=1)

            for rc in picked_rowcols:
                quantity = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
                key, _ = jax.random.split(key, 2)
                all_resource_map = all_resource_map.at[rc[0], rc[1]].set(strat_res_id)
                all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(quantity)
                resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
    
    print("\tAdding City State strategic resources")
    for i, cs_rc in enumerate(_global_cs_holder):

        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=cs_rc[0], col=cs_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
        hexes_surrounding = jax.vmap(
            get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
            in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)

        hexes_surrounding = jax.vmap(
                get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                in_axes=(0, 0, None, None),
        )(
            hexes_surrounding[:, 0],
            hexes_surrounding[:, 1],
            landmask.shape[0],
            landmask.shape[1]
        ).reshape(-1, 2)

        hexes_surrounding = unique_rows(hexes_surrounding)

        which_strategic = jax.random.randint(key=key, shape=(), minval=0, maxval=4).item()
        key, _ = jax.random.split(key, 2)

        if which_strategic == 0:
            continue
        elif which_strategic == 1:
            strat_res = "coal"
        elif which_strategic == 2:
            strat_res = "oil"
        elif which_strategic == 3:
            strat_res = "aluminium"
        else:
            raise ValueError(f"Should not be here with which_strategic={which_strategic}")

        strat_res_id = RESOURCE_TO_IDX[strat_res]
        bias_type = STRATEGIC_BIAS_TABLE[strat_res]["terrain_bias"]
        picked_rowcols = translate_terrain_bias_to_tile_samples_strategic(bias_type=bias_type, landmask=landmask, features=features, terrain=terrain, elevation_map=elevation_map, potential_tiles=hexes_surrounding, settler_rc=cs_rc, nw_map=nw_placements, lakes=lakes, current_resource_map=all_resource_map, key=key, num_samples=1)

        for rc in picked_rowcols:
            quantity = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
            key, _ = jax.random.split(key, 2)
            all_resource_map = all_resource_map.at[rc[0], rc[1]].set(strat_res_id)
            all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(quantity)
            resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)

    # Now let's add some fish...
    fish_idx = RESOURCE_TO_IDX["fish"]

    # Let's first add a specifically to the region of any coastal civs
    for i in range(cfg.num_players):
        if player_civ_is_coastal[i]:
            settler_rc = settler_start_rowcols[i]
            hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=settler_rc[0], col=settler_rc[1], max_rows=landmask.shape[0], max_cols=landmask.shape[1])
            hexes_surrounding = jax.vmap(
                get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                in_axes=(0, 0, None, None),
            )(
                hexes_surrounding[:, 0],
                hexes_surrounding[:, 1],
                landmask.shape[0],
                landmask.shape[1]
            ).reshape(-1, 2)

            hexes_surrounding = unique_rows(hexes_surrounding)

            hexes_surrounding = jax.vmap(
                    get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol,
                    in_axes=(0, 0, None, None),
            )(
                hexes_surrounding[:, 0],
                hexes_surrounding[:, 1],
                landmask.shape[0],
                landmask.shape[1]
            ).reshape(-1, 2)

            hexes_surrounding = unique_rows(hexes_surrounding)
            
            total_ocean_tiles = 0
            possible_fish_tiles = []
            for tile in hexes_surrounding:
                if (landmask[tile[0], tile[1]] == 0) and (all_resource_map[tile[0], tile[1]] == 0):
                    total_ocean_tiles += 1
                    possible_fish_tiles.append(np.array(tile).tolist())
            
            if total_ocean_tiles >= 4:
                num_fish = 2
            else:
                num_fish = 1
            
            for _ in range(num_fish):
                idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(possible_fish_tiles)).item()
                key, _ = jax.random.split(key, 2)
                new_fish_tile = possible_fish_tiles[idx]
                all_resource_map = all_resource_map.at[new_fish_tile[0], new_fish_tile[1]].set(fish_idx)
                all_resource_quantity_map = all_resource_quantity_map.at[new_fish_tile[0], new_fish_tile[1]].set(1)
                resource_type_map = resource_type_map.at[new_fish_tile[0], new_fish_tile[1]].set(2)

    uniform_map = jax.random.uniform(key=key, shape=landmask.shape, minval=0, maxval=1)
    key, _ = jax.random.split(key, 2)
    fish_map = (uniform_map < cfg.ocean_fish_threshold) & ~landmask & (all_resource_map == 0)
    all_resource_map = all_resource_map + fish_map * fish_idx
    all_resource_quantity_map = all_resource_quantity_map + fish_map
    resource_type_map = resource_type_map + fish_map * 2

    # Now some oil in the ocean
    uniform_map = jax.random.uniform(key=key, shape=landmask.shape, minval=0, maxval=1)
    oil_amount_map = jax.random.randint(key=key, shape=landmask.shape, minval=strategic_resources_with_quantity["oil"][0], maxval=strategic_resources_with_quantity["oil"][1])
    key, _ = jax.random.split(key, 2)
    oil_map = (uniform_map < cfg.ocean_oil_threshold) & ~landmask & (all_resource_map == 0)
    all_resource_map = all_resource_map + oil_map * RESOURCE_TO_IDX["oil"]
    oil_amount_map = oil_amount_map * oil_map 
    all_resource_quantity_map = all_resource_quantity_map + oil_amount_map
    resource_type_map = resource_type_map + oil_map * 2


    # Before we move on to other resoures, we need to make sure the map has a certain number of each
    # quantity-based strategic resources.
    # Process: uniformly choose tile, filter for relevant placement criteria, place, increment
    # iron: 4 * num_civs
    print("\tEnsuring strategic resource balance...")

    _global_cs_holder = [np.array(x).tolist() for x in _global_cs_holder]

    num_iron_placed = (all_resource_map == RESOURCE_TO_IDX["iron"]).sum()
    strat_res = "iron"

    while num_iron_placed < 4 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if lakes[row, col] == 1:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_iron_placed += 1

    # Horse 4 * num_civs
    num_horse_placed = (all_resource_map == RESOURCE_TO_IDX["horses"]).sum()
    strat_res = "horses"

    while num_horse_placed < 4 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if elevation_map[row, col] == 2:
            continue
        if lakes[row, col] == 1:
            continue
        if features[row, col] == 2:
            continue
        if features[row, col] == 3:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_horse_placed += 1

    # coal 4 * num_civs
    num_coal_placed = (all_resource_map == RESOURCE_TO_IDX["coal"]).sum()
    strat_res = "coal"

    while num_coal_placed < 4 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if lakes[row, col] == 1:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_coal_placed += 1

    # oil 4 * num_civs
    num_oil_placed = (all_resource_map == RESOURCE_TO_IDX["oil"]).sum()
    strat_res = "oil"

    while num_oil_placed < 4 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if lakes[row, col] == 1:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_oil_placed += 1

    # aluminium = 4 * num_civs
    num_aluminum_placed = (all_resource_map == RESOURCE_TO_IDX["aluminium"]).sum()
    strat_res = "aluminium"

    while num_aluminum_placed < 4 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if lakes[row, col] == 1:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_aluminum_placed += 1

    # uranium = 4 * num_civs
    num_uranium_placed = (all_resource_map == RESOURCE_TO_IDX["uranium"]).sum()
    strat_res = "uranium"

    while num_uranium_placed < 2 * cfg.num_players:
        row = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[0]).item()
        key, _ = jax.random.split(key, 2)
        col = jax.random.randint(key=key, shape=(), minval=0, maxval=landmask.shape[1]).item()
        key, _ = jax.random.split(key, 2)

        if landmask[row, col] == 0:
            continue
        if all_resource_map[row, col] > 0:
            continue
        if elevation_map[row, col] == 3:
            continue
        if lakes[row, col] == 1:
            continue
        if nw_placements[row, col] > 0:
            continue
        if [row, col] in _global_cs_holder:
            continue

        all_resource_map = all_resource_map.at[row, col].set(RESOURCE_TO_IDX[strat_res])
        resource_amt = jax.random.randint(key=key, shape=(), minval=strategic_resources_with_quantity[strat_res][0], maxval=strategic_resources_with_quantity[strat_res][1]).item()
        key, _ = jax.random.split(key, 2)
        all_resource_quantity_map = all_resource_quantity_map.at[row, col].set(resource_amt)
        resource_type_map = resource_type_map.at[row, col].set(2)
        num_uranium_placed += 1

    # Now on to all of the other resources...
    # The way the source code does it (roughly) for each type of resource => num_tiles_possible / frequency = num_tiles_to_place
    # The way we get the number of possible tiles is to loop through each remaining tile in the entire map and catalogue it
    # To avoid confusion for me, I will be naming the lists in the same way they name it in the source code...
    extra_deer_list = []  # looks like only any tundra forest
    desert_wheat_list = []
    tundra_flat_no_feature = []
    banana_list = []
    plains_flat_no_feature = []
    grass_flat_no_feature = []
    dry_grass_flat_no_feature = []
    hills_open_list = []
    desert_flat_no_feature = []
    forest_flat_that_are_not_tundra = []
    hills_covered_list = []
    flat_covered = []
    tundra_flat_forest = []

    freshwater_map = jnp.zeros_like(landmask)
    
    

    # --- one-time precomputations ---
    lake_victoria_id = LAKE_VICTORIA_IDX + 1
    H, W = landmask.shape

    neighbors = [[None]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            neighbors[r][c] = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(
                row=r, col=c, max_rows=H, max_cols=W
            )

    river_any = (edge_rivers.sum(axis=-1) > 0)  # shape (H, W)

    _global_cs_holder_set = set(map(tuple, _global_cs_holder))

    def has_freshwater(r, c):
        # any neighbor is non-ocean water (lake, Lake Victoria, or feature==4)
        for nr, nc in neighbors[r][c]:
            if lakes[nr, nc]:             # lake tile
                return True
            if nw_placements[nr, nc] == lake_victoria_id:  # Lake Victoria
                return True
            if features[nr, nc] == 4:
                return True
        if river_any[r, c]:               # edge rivers on this tile
            return True
        return False


    for row in tqdm(range(H), desc="Cataloging map..."):
        for col in range(W):
            # local refs (avoid repeated indexing)
            lm = landmask[row, col]
            res = all_resource_map[row, col]
            elev = elevation_map[row, col]
            lake = lakes[row, col]
            nw   = nw_placements[row, col]
            feat = features[row, col]
            terr = terrain[row, col]

            # freshwater check once per tile; reused below
            fw = has_freshwater(row, col)
            if fw:
                freshwater_map = freshwater_map.at[row, col].set(1)

            if lm == 0:  # ocean/invalid
                continue
            if res > 0:
                continue
            if elev == 3:
                continue
            if lake == 1:
                continue
            if nw > 0:
                continue
            if feat == 4:
                continue
            if (row, col) in _global_cs_holder_set:
                continue

            # extra deer: feature==1 on tundra (terrain==4)
            if (feat == 1) and (terr == 4):
                extra_deer_list.append([row, col])

            # desert wheat always
            if feat == 5:
                desert_wheat_list.append([row, col])

            # desert wheat also if desert flat (terr==3 & elev!=2) with freshwater nearby
            if (terr == 3) and (elev != 2):
                if fw: 
                    desert_wheat_list.append([row, col])

            # tundra flat no feature
            if (terr == 4) and (feat == 0) and (elev != 2):
                tundra_flat_no_feature.append([row, col])

            # banana
            if feat == 2:
                banana_list.append([row, col])

            # plains flat no feature
            if (feat == 0) and (elev == 1) and (terr == 2):
                plains_flat_no_feature.append([row, col])

            # grass flat no feature (+ dry variant if NO freshwater nearby)
            if (feat == 0) and (elev == 1) and (terr == 1):
                grass_flat_no_feature.append([row, col])
                if not fw:
                    dry_grass_flat_no_feature.append([row, col])

            # hills open (no feature)
            if (elev == 2) and (feat == 0):
                hills_open_list.append([row, col])

            # desert flat no feature
            if (terr == 3) and (elev == 1) and (feat == 0):
                desert_flat_no_feature.append([row, col])

            # forest flat not tundra
            if (feat == 1) and (elev == 1) and (terr != 4):
                forest_flat_that_are_not_tundra.append([row, col])

            # hills covered (forest or jungle)
            if (elev == 2) and ((feat == 1) or (feat == 2)):
                hills_covered_list.append([row, col])
            # flat covered (forest or jungle on flat)
            if (elev == 1) and ((feat == 1) or (feat == 2)):
                flat_covered.append([row, col])

            # tundra flat forest
            if (elev == 1) and (terr == 4) and (feat == 1):
                tundra_flat_forest.append([row, col])

    #for row in tqdm(range(landmask.shape[0])):
    #    for col in range(landmask.shape[1]):
    #        # Let's take advantage of this loop... If any of the surrounding tiles are non-ocean water, then we have
    #        # freshwater on this tile. This is ofc consequential for many things
    #        hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=row, col=col, max_rows=landmask.shape[0], max_cols=landmask.shape[1])
    #        _found_freshwater = False
    #        for _hex in hexes_surrounding:
    #            if lakes[_hex[0], _hex[1]]:
    #                _found_freshwater = True
    #            if nw_placements[_hex[0], _hex[1]] == (LAKE_VICTORIA_IDX + 1):  # need to check for Lake Victoria
    #                _found_freshwater = True
    #            if features[_hex[0], _hex[1]] == 4:
    #                _found_freshwater = True

    #        if edge_rivers[row, col].sum() > 0:
    #            _found_freshwater = True
    #        
    #        if _found_freshwater:
    #            freshwater_map = freshwater_map.at[row, col].set(1)

    #        if landmask[row, col] == 0:
    #            continue
    #        if all_resource_map[row, col] > 0:
    #            continue
    #        if elevation_map[row, col] == 3:
    #            continue
    #        if lakes[row, col] == 1:
    #            continue
    #        if nw_placements[row, col] > 0:
    #            continue
    #        if features[row, col] == 4:
    #            continue
    #        if [row, col] in _global_cs_holder:
    #            continue

    #        if (features[row, col] == 1) & (terrain[row, col] == 4):
    #            extra_deer_list.append([row, col])
    #        if (features[row, col] == 5):
    #            desert_wheat_list.append([row, col])
    #        if (terrain[row, col] == 3) & (elevation_map[row, col] != 2):
    #            # Check all surroundings for lake, ergo, freshwater
    #            hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=row, col=col, max_rows=landmask.shape[0], max_cols=landmask.shape[1])
    #            _found_freshwater = False
    #            for _hex in hexes_surrounding:
    #                if lakes[_hex[0], _hex[1]]:
    #                    _found_freshwater = True
    #                if nw_placements[_hex[0], _hex[1]] == (LAKE_VICTORIA_IDX + 1):  # need to check for Lake Victoria
    #                    _found_freshwater = True
    #                if features[_hex[0], _hex[1]] == 4:
    #                    _found_freshwater = True

    #            if edge_rivers[row, col].sum() > 0:
    #                _found_freshwater = True
    #            
    #            if _found_freshwater:
    #                desert_wheat_list.append([row, col])

    #        if terrain[row, col] == 4:
    #            if features[row, col] == 0:
    #                if elevation_map[row, col] != 2:
    #                    tundra_flat_no_feature.append([row, col])

    #        if features[row, col] == 2:
    #            banana_list.append([row, col])

    #        if (features[row, col] == 0) & (elevation_map[row, col] == 1) & (terrain[row, col] == 2):
    #            plains_flat_no_feature.append([row, col])

    #        if (features[row, col] == 0) & (elevation_map[row, col] == 1) & (terrain[row, col] == 1):
    #            grass_flat_no_feature.append([row, col])

    #            hexes_surrounding = get_surrounding_hexes_in_gamestate_space_from_gamestate_rowcol(row=row, col=col, max_rows=landmask.shape[0], max_cols=landmask.shape[1])
    #            _found_freshwater = False
    #            for _hex in hexes_surrounding:
    #                if lakes[_hex[0], _hex[1]]:
    #                    _found_freshwater = True
    #                if nw_placements[_hex[0], _hex[1]] == (LAKE_VICTORIA_IDX + 1):  # need to check for Lake Victoria
    #                    _found_freshwater = True
    #                if features[_hex[0], _hex[1]] == 4:
    #                    _found_freshwater = True
    #            
    #            if edge_rivers[row, col].sum() > 0:
    #                _found_freshwater = True

    #            if not _found_freshwater:
    #                dry_grass_flat_no_feature.append([row, col])

    #        if (elevation_map[row, col] == 2) & (features[row, col] == 0):
    #            hills_open_list.append([row, col])

    #        if (terrain[row, col] == 3) & (elevation_map[row, col] == 1) & (features[row, col] == 0):
    #            desert_flat_no_feature.append([row, col])

    #        if (features[row, col] == 1) & (elevation_map[row, col] == 1) & (terrain[row, col] != 4):
    #            forest_flat_that_are_not_tundra.append([row, col])

    #        if (elevation_map[row, col] == 2) & ((features[row, col] == 1) | (features[row, col] == 2)):
    #            hills_covered_list.append([row, col])

    #        if (elevation_map[row, col] == 1) & ((features[row, col] == 1) | (features[row, col] == 2)):
    #            flat_covered.append([row, col])

    #        if (elevation_map[row, col] == 1) & (terrain[row, col] == 4) & (features[row, col] == 1):
    #            tundra_flat_forest.append([row, col])

    print("Done looping over the map and cataloging the tiles:")
    print(f"\textra_deer_list: {len(extra_deer_list)}")
    print(f"\tdesert_wheat_list: {len(desert_wheat_list)}")
    print(f"\ttundra_flat_no_feature: {len(tundra_flat_no_feature)}")
    print(f"\tbanana_list: {len(banana_list)}")
    print(f"\tplains_flat_no_feature: {len(plains_flat_no_feature)}")
    print(f"\tgrass_flat_no_feature: {len(grass_flat_no_feature)}")
    print(f"\tdry_grass_flat_no_feature: {len(dry_grass_flat_no_feature)}")
    print(f"\thills_open_list: {len(hills_open_list)}")
    print(f"\ttundra_flat_no_feature: {len(tundra_flat_no_feature)}")
    print(f"\ttundra_flat_no_feature: {len(tundra_flat_no_feature)}")
    print(f"\tforest_flat_that_are_not_tundra: {len(forest_flat_that_are_not_tundra)}")
    print(f"\thills_covered_list: {len(hills_covered_list)}")
    print(f"\tflat_covered: {len(flat_covered)}")
    print(f"\ttundra_flat_forest: {len(tundra_flat_forest)}")
    print(f"\tplains_flat_no_feature: {len(plains_flat_no_feature)}")

    # Placement begins here (https://github.com/EnormousApplePie/Lekmod/blob/main/Lekmap/HBAssignStartingPlots.lua#L14075)
    frequency = 6 * cfg.bonus_multiplier
    num_to_place = int(len(extra_deer_list) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(extra_deer_list)).item()
        key, _ = jax.random.split(key, 2)
        rc = extra_deer_list[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["deer"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 16 * cfg.bonus_multiplier
    num_to_place = int(len(desert_wheat_list) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(desert_wheat_list)).item()
        key, _ = jax.random.split(key, 2)
        rc = desert_wheat_list[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["wheat"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 8 * cfg.bonus_multiplier
    num_to_place = int(len(tundra_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(tundra_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = tundra_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["deer"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 28 * cfg.bonus_multiplier
    num_to_place = int(len(banana_list) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(banana_list)).item()
        key, _ = jax.random.split(key, 2)
        rc = banana_list[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["banana"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 33 * cfg.bonus_multiplier
    num_to_place = int(len(plains_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(plains_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = plains_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["wheat"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 18 * cfg.bonus_multiplier
    num_to_place = int(len(plains_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(plains_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = plains_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["bison"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 33 * cfg.bonus_multiplier
    num_to_place = int(len(plains_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(plains_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = plains_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["cow"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 33 * cfg.bonus_multiplier
    num_to_place = int(len(grass_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(grass_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = grass_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["cow"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 25 * cfg.bonus_multiplier
    num_to_place = int(len(dry_grass_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(dry_grass_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = dry_grass_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["stone"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 25 * cfg.bonus_multiplier
    num_to_place = int(len(dry_grass_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(dry_grass_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = dry_grass_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["bison"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 25 * cfg.bonus_multiplier
    num_to_place = int(len(hills_open_list) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(hills_open_list)).item()
        key, _ = jax.random.split(key, 2)
        rc = hills_open_list[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["sheep"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 10 * cfg.bonus_multiplier
    num_to_place = int(len(tundra_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(tundra_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = tundra_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["stone"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 20 * cfg.bonus_multiplier
    num_to_place = int(len(desert_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(desert_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = desert_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["stone"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 30 * cfg.bonus_multiplier
    num_to_place = int(len(forest_flat_that_are_not_tundra) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(forest_flat_that_are_not_tundra)).item()
        key, _ = jax.random.split(key, 2)
        rc = forest_flat_that_are_not_tundra[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["deer"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 35 * cfg.bonus_multiplier
    num_to_place = int(len(hills_covered_list) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(hills_covered_list)).item()
        key, _ = jax.random.split(key, 2)
        rc = hills_covered_list[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["hardwood"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 35 * cfg.bonus_multiplier
    num_to_place = int(len(flat_covered) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(flat_covered)).item()
        key, _ = jax.random.split(key, 2)
        rc = flat_covered[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["hardwood"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 8 * cfg.bonus_multiplier
    num_to_place = int(len(tundra_flat_forest) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(tundra_flat_forest)).item()
        key, _ = jax.random.split(key, 2)
        rc = tundra_flat_forest[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["hardwood"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    frequency = 35 * cfg.bonus_multiplier
    num_to_place = int(len(plains_flat_no_feature) / frequency)
    num_did_place = 0
    while num_did_place < num_to_place:
        _idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(plains_flat_no_feature)).item()
        key, _ = jax.random.split(key, 2)
        rc = plains_flat_no_feature[_idx]
        if all_resource_map[rc[0], rc[1]] > 0:
            continue
        all_resource_map = all_resource_map.at[rc[0], rc[1]].set(RESOURCE_TO_IDX["maize"])
        all_resource_quantity_map = all_resource_quantity_map.at[rc[0], rc[1]].set(1)
        resource_type_map = resource_type_map.at[rc[0], rc[1]].set(2)
        num_did_place += 1

    print(f"\n\t{(all_resource_map > 0).mean()}% of the map covered in resources")


    return _global_cs_holder, cs_ownership_map, features, all_resource_map, all_resource_quantity_map, resource_type_map, freshwater_map


def compute_yields(landmask, elevation_map, terrain, edge_rivers, lakes, features, nw_placements, cfg, key, all_resource_map):
    """
    0=food
    1=prod
    2=gold
    3=faith
    4=culture
    5=science
    6=happiness
    """
    print("Computing initial yields...")
    yield_map = jnp.zeros(shape=(*landmask.shape, 7))

    # terrain
    ocean_mask = (terrain == 0)[:, :, None]
    grass_mask = (terrain == 1)[:, :, None]
    plains_mask = (terrain == 2)[:, :, None]
    desert_mask = (terrain == 3)[:, :, None]
    tundra_mask = (terrain == 4)[:, :, None]
    snow_mask = (terrain == 5)[:, :, None]

    yield_map = yield_map + ocean_mask * jnp.array([1, 0, 0, 0, 0, 0, 0])[None, None]
    yield_map = yield_map + grass_mask * jnp.array([2, 0, 0, 0, 0, 0, 0])[None, None]
    yield_map = yield_map + plains_mask * jnp.array([1, 1, 0, 0, 0, 0, 0])[None, None]
    yield_map = yield_map + desert_mask * jnp.array([0, 0, 0, 0, 0, 0, 0])[None, None]
    yield_map = yield_map + tundra_mask * jnp.array([1, 0, 0, 0, 0, 0, 0])[None, None]
    yield_map = yield_map + snow_mask * jnp.array([0, 0, 0, 0, 0, 0, 0])[None, None]

    forest_mask = (features == 1)[:, :, None]
    jungle_mask = (features == 2)[:, :, None]
    marsh_mask = (features == 3)[:, :, None]
    oasis_mask = (features == 4)[:, :, None]
    floodplains_mask = (features == 5)[:, :, None]

    flatland_mask = (elevation_map == 1)[:, :, None]
    hills_mask = (elevation_map == 2)[:, :, None]
    mountain_mask = (elevation_map == 3)[:, :, None]

    lakes_mask = (lakes == 1)[:, :, None]

    # Forest interaction
    yield_map = yield_map + (flatland_mask & plains_mask & forest_mask) * jnp.array([0, 0, 0, 0, 0, 0, 0])
    yield_map = yield_map + (hills_mask & plains_mask & forest_mask) * jnp.array([0, 0, 0, 0, 0, 0, 0])
    yield_map = yield_map + (flatland_mask & grass_mask & forest_mask) * jnp.array([-1, 1, 0, 0, 0, 0, 0])
    yield_map = yield_map + (hills_mask & grass_mask & forest_mask) * jnp.array([-1, 1, 0, 0, 0, 0, 0])

    # Jungle interaction (grassland jungle does not exist!)
    yield_map = yield_map + (flatland_mask & plains_mask & jungle_mask) * jnp.array([1, -1, 0, 0, 0, 0, 0])
    yield_map = yield_map + (hills_mask & plains_mask & jungle_mask) * jnp.array([1, -1, 0, 0, 0, 0, 0])

    # marsh interaction (always one food?)
    yield_map = jnp.where(marsh_mask, jnp.array([1, 0, 0, 0, 0, 0, 0]), yield_map)

    # oasis interaction (always 3 food 1 gold)
    yield_map = jnp.where(oasis_mask, jnp.array([3, 0, 1, 0, 0, 0, 0]), yield_map)

    # floodplains interaction (always 2 food)
    yield_map = jnp.where(floodplains_mask, jnp.array([2, 0, 0, 0, 0, 0, 0]), yield_map)

    # Hills interaction
    # plains hills 0food 2prod
    yield_map = jnp.where(plains_mask & hills_mask & ~jungle_mask, jnp.array([0, 2, 0, 0, 0, 0, 0]), yield_map)
    yield_map = jnp.where(grass_mask & hills_mask, jnp.array([0, 2, 0, 0, 0, 0, 0]), yield_map)
    yield_map = jnp.where(desert_mask & hills_mask, jnp.array([0, 2, 0, 0, 0, 0, 0]), yield_map)
    yield_map = jnp.where(hills_mask & forest_mask, jnp.array([1, 1, 0, 0, 0, 0, 0]), yield_map)

    # Lakes
    yield_map = jnp.where(lakes_mask, jnp.array([3, 0, 0, 0, 0, 0, 0]), yield_map)

    # NW
    for row in range(nw_placements.shape[0]):
        for col in range(nw_placements.shape[1]):
            if nw_placements[row, col] == 0:
                continue
            yield_map = yield_map.at[row, col].set(NW_YIELD_TABLE_IDX[nw_placements[row, col] - 1])

    yield_map = jnp.clip(yield_map, a_min=0, a_max=99999)

    # Now time for resources, but only the ones that are visible at the beginning of the game!
    # All luxuries are revealed...
    # These resources have been written to be additive to the map
    # Let's make sure these are added... 
    for res in ALL_RESOURCES:
        #if res in ["iron", "uranium", "oil", "aluminium", "coal", "horses"]:
        #    continue

        # RESOURCE_TO_IDX does +1 to idx for us 
        res_idx = RESOURCE_TO_IDX[res]
        res_additive_yields = RESOURCE_YIELDS[res_idx - 1][0][None, None]
        res_mask = (all_resource_map == res_idx)[:, :, None]
        res_adds = res_mask * res_additive_yields
        yield_map = yield_map + res_adds * ~marsh_mask

    yield_map = jnp.where(mountain_mask, jnp.array([0, 0, 0, 0, 0, 0, 0]), yield_map)
    
    return yield_map


def generate_map(cfg, key, border=3):
    print("Generating landmass...")
    landmask = generate_landmask(key=key, shape=cfg.small.dimensions, border=border)
    print("Moving tectonic plates...")
    elevation_map = generate_elevation_map(cfg.small.elevation, key, landmask)
    print("Creating biomes...")
    terrain = generate_terrain_type_map(cfg.small.terrain, key, landmask)

    # water
    print("Spawning rivers...")
    river_sources = generate_river_sources(cfg.small.rivers, cfg.small, key, elevation_map, landmask, border=0)
    source_indices = np.argwhere(np.array(river_sources) == 1)


    edge_rivers = np.zeros((landmask.shape[0], landmask.shape[1], 6), dtype=jnp.uint8)

    # [row, col] =>  [y, x] 
    for river_id, (y, x) in enumerate(source_indices):
        edge_rivers = do_river_improved(y, x, elevation_map, landmask, ~landmask, edge_rivers, river_id, max_length=cfg.small.rivers.river_max_length, key=key)
        key, _ = jax.random.split(key, 2)
    
    print("Digging lakes...")
    lakes = generate_lakes(cfg=cfg.small.lakes, key=key, landmask=landmask, coastal_mask=compute_coastline_mask(landmask), river_mask=edge_rivers, terrain_type=terrain)
    print("Adding features...")
    features, terrain = generate_features(cfg.small.features, key=key, terrain=terrain, elevation=elevation_map, rivers=edge_rivers, lakes=lakes, latitude_map=generate_latitude_map(terrain.shape[0], terrain.shape[1]), landmask=landmask)
    print("Assigning start locations...")
    fertility, settler_rowcols, subregion_stats = assign_starting_locations(terrain, elevation_map, features, edge_rivers, lakes, ~landmask, compute_coastline_mask(landmask), key)
    print("Placing natural wonders...")
    nw_placements, elevation_map, features = place_natural_wonders(cfg.small.num_natural_wonders, settler_rowcols, landmask, elevation_map, terrain, edge_rivers.argmax(-1), lakes, features, cfg.small.nw_min_distance_from_start, key)
    print("Planting resources...")
    cs_rowcols, cs_ownership_map, features, all_resource_map, all_resource_quantity_map, resource_type_map, freshwater_map = place_resources_and_cs(settler_start_rowcols=settler_rowcols, landmask=landmask, elevation_map=elevation_map, terrain=terrain, edge_rivers=edge_rivers, lakes=lakes, features=features, nw_placements=nw_placements, cfg=cfg.small, key=key)
    print("Computing tile yields...")
    yield_map = compute_yields(landmask=landmask, elevation_map=elevation_map, terrain=terrain, edge_rivers=edge_rivers, lakes=lakes, features=features, nw_placements=nw_placements, cfg=cfg.small, key=key, all_resource_map=all_resource_map)
    print("~~~~~~~~~~~~~~~~~~~~~~ DONE ~~~~~~~~~~~~~~~~~~~~~")
    return landmask, elevation_map, terrain, edge_rivers, lakes, features, fertility, settler_rowcols, subregion_stats, nw_placements, cs_rowcols, cs_ownership_map, all_resource_map, all_resource_quantity_map, resource_type_map, freshwater_map, yield_map

