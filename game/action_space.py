"""
This files contains the standardized interfaces between the different types of actions a unit may perform and
the gamestate upon which those actions are performed.

All interfaces are of the form (GameState, (row, col), player_id) => GameState
All interfaces will have built-in filters for "can/cannot be done", s.t. the game engine code
does not need to do any manual re-weighting other than the tensordot between the returned leaves and
the [1,0] mask for the action category selected.

Actions for controlling units come in two forms:
    1. Categories
    2. Map placement

Categories include [movement, settle, farm, plantation, ...]
Map placements are e.g., (2772,) vector of logits that are to be masked to valid tiles depending on the corresponding action category.

Each unit type will have an action-category mask that tells the game engine and agent which action categories the unit being considered is capable of executing.

The process might look like:
    1. Logits for action-category output
    2. Action-category mask applied
    3. Sample category
    4. Category determines which valid-move mask to use
    5. Valid move mask applied
    6. Hex-tile sampled


These standardized interfaces need to return the minimal amount of data, as the lack of batched cond
forces the jax program to evaluate every branch within a `switch`, so whatever data is returned is effectively duplicated N times in memory, where N is the number of possible branches.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from dataclasses import fields, replace

from typing import TYPE_CHECKING
from game.buildings import GameBuildings
from game.constants import GOLD_IDX, INFLUENCE_PER_TURN_TRADEROUTE, LAND_TRADEROUTE_RANGE, RELIGIOUS_PRESSURE_TRADEROUTE, SCIENCE_PER_ERA_TRADEROUTE, HILLS_IDX, SNOW_IDX, JUNGLE_IDX
from game.improvements import Improvements, _farm, _pasture, _mine, _fishing_boat, _plantation, _camp, _quarry, _lumber_mill, _fort, _trading_post, _road, _chop_forest, _chop_jungle, _clear_marsh
from game.religion import MAX_IDX_FOUNDER, MAX_IDX_PANTHEON, ReligiousTenets
from game.social_policies import SocialPolicies

from game.techs import Technologies
from utils.maths import hex_flat_index, social_policy_threshold, compute_all_distances_vectorized, get_hex_rings_vectorized, rowcol_to_hex
from utils.misc import improvement_mask_for_batch

if TYPE_CHECKING:
    from game.primitives import GameState


@dataclass
class UnitsMinimal:
    """Minimal units dataclass containing only fields that can be updated by switch functions"""
    unit_type: jnp.ndarray
    unit_ap: jnp.ndarray  
    unit_rowcol: jnp.ndarray
    trade_to_player_int: jnp.ndarray
    trade_to_city_int: jnp.ndarray
    trade_from_city_int: jnp.ndarray
    trade_yields: jnp.ndarray

@dataclass 
class CitiesMinimal:
    """Minimal cities dataclass containing only fields that can be updated by switch functions"""
    ownership_map: jnp.ndarray
    city_rowcols: jnp.ndarray
    yields: jnp.ndarray
    city_center_yields: jnp.ndarray
    city_ids: jnp.ndarray
    population: jnp.ndarray
    potential_owned_rowcols: jnp.ndarray
    buildings_owned: jnp.ndarray
    is_coastal: jnp.ndarray

@dataclass
class GameStateMinimal:
    """Minimal game state containing only fields that can be updated by switch functions"""
    units: UnitsMinimal
    player_cities: CitiesMinimal  
    technologies: jnp.ndarray
    culture_threshold: jnp.ndarray
    improvement_additional_yield_map: jnp.ndarray
    feature_map: jnp.ndarray
    improvement_map: jnp.ndarray
    road_map: jnp.ndarray

# Pre-compute field names at module level (compile-time constants)
_UNITS_MINIMAL_FIELDS = tuple(f.name for f in fields(UnitsMinimal))
_CITIES_MINIMAL_FIELDS = tuple(f.name for f in fields(CitiesMinimal))

def set_from_subset(a, b, b_fieldnames):
    """
    Return a new A whose fields present in B are taken from B,
    everything else kept from A. Works in jitted code because:
      • field list is a compile-time constant (b_fieldnames)
      • we do no control-flow on JAX values
    """
    update_dict = {name: getattr(b, name) for name in b_fieldnames}
    return replace(a, **update_dict)

def apply_minimal_update_game_actions(full_game, minimal_update: GameStateMinimal):
    """Apply minimal update to full game state using generic set_from_subset."""
    
    # Update nested structures
    updated_units = set_from_subset(full_game.units, minimal_update.units, _UNITS_MINIMAL_FIELDS)
    updated_cities = set_from_subset(full_game.player_cities, minimal_update.player_cities, _CITIES_MINIMAL_FIELDS)
    
    # Update top-level game state
    updated_game = replace(
        full_game, 
        units=updated_units,
        player_cities=updated_cities,
        technologies=minimal_update.technologies,
        culture_threshold=minimal_update.culture_threshold,
        improvement_additional_yield_map=minimal_update.improvement_additional_yield_map,
        feature_map=minimal_update.feature_map,
        improvement_map=minimal_update.improvement_map,
        road_map=minimal_update.road_map,
    )
    
    return updated_game

# Function to create minimal objects with selective updates
def create_minimal_update(game, 
                         # Units fields
                         unit_type=None, unit_ap=None, unit_rowcol=None,
                          trade_to_player_int=None, trade_to_city_int=None, trade_from_city_int=None, trade_yields=None,
                         # Cities fields  
                         ownership_map=None, city_rowcols=None, yields=None,
                         city_center_yields=None, city_ids=None, population=None,
                         potential_owned_rowcols=None, buildings_owned=None, is_coastal=None,
                         # Top-level fields
                         technologies=None, culture_threshold=None, improvement_additional_yield_map=None, feature_map=None, improvement_map=None, road_map=None) -> GameStateMinimal:
    """
    Create a GameStateMinimal object. If an argument is None, use the value from the full game object.
    This ensures only explicitly updated fields are changed, others preserve original values.
    """
    return GameStateMinimal(
        units=UnitsMinimal(
            unit_type=unit_type if unit_type is not None else game.units.unit_type,
            unit_ap=unit_ap if unit_ap is not None else game.units.unit_ap,
            unit_rowcol=unit_rowcol if unit_rowcol is not None else game.units.unit_rowcol,

            trade_to_player_int=trade_to_player_int if trade_to_player_int is not None else game.units.trade_to_player_int,
            trade_to_city_int=trade_to_city_int if trade_to_city_int is not None else game.units.trade_to_city_int,
            trade_from_city_int=trade_from_city_int if trade_from_city_int is not None else game.units.trade_from_city_int,
            trade_yields=trade_yields if trade_yields is not None else game.units.trade_yields,
        ),
        player_cities=CitiesMinimal(
            ownership_map=ownership_map if ownership_map is not None else game.player_cities.ownership_map,
            city_rowcols=city_rowcols if city_rowcols is not None else game.player_cities.city_rowcols,
            yields=yields if yields is not None else game.player_cities.yields,
            city_center_yields=city_center_yields if city_center_yields is not None else game.player_cities.city_center_yields,
            city_ids=city_ids if city_ids is not None else game.player_cities.city_ids,
            population=population if population is not None else game.player_cities.population,
            potential_owned_rowcols=potential_owned_rowcols if potential_owned_rowcols is not None else game.player_cities.potential_owned_rowcols,
            buildings_owned=buildings_owned if buildings_owned is not None else game.player_cities.buildings_owned,
            is_coastal=is_coastal if is_coastal is not None else game.player_cities.is_coastal
        ),
        technologies=technologies if technologies is not None else game.technologies,
        culture_threshold=culture_threshold if culture_threshold is not None else game.culture_threshold,

        improvement_additional_yield_map=improvement_additional_yield_map if improvement_additional_yield_map is not None else game.improvement_additional_yield_map,
        feature_map=feature_map if feature_map is not None else game.feature_map,
        improvement_map=improvement_map if improvement_map is not None else game.improvement_map,
        road_map=road_map if road_map is not None else game.road_map,
    )

def do_settlev2(game: GameState, unit_rowcol: jnp.ndarray, player_id: jnp.ndarray, unit_int, actions_map):
    """
    This is to be called in a vmap-over-games context
    """
    # We first need to check if a city can even be settled in this particular tile. At this point,
    # the settler has already moved to the tile it wants to settle on.
    def sum_at_rowcol(x, row, col):
        """x.shape == (6, 5, 42, 66); row, col are dynamic scalars."""
        patch = jax.lax.dynamic_slice(
            x,  # tensor
            (0, 0, row, col),  # start indices
            (x.shape[0],  # slice size in each dim
             x.shape[1],
             1, 1)
       )
        return patch.sum()
    
    # The citystate ownership_map works differently than the player ownership map. Perhaps this needs to 
    # be computed differently than sum_at_rowcol()
    cs_blocked = game.cs_ownership_map[unit_rowcol[0], unit_rowcol[1]] > 0
    player_blocked = sum_at_rowcol(game.player_cities.ownership_map, unit_rowcol[0], unit_rowcol[1]) > 0

    ocean_blocked = game.landmask_map[unit_rowcol[0], unit_rowcol[1]] == 0
    lake_blocked = game.lake_map[unit_rowcol[0], unit_rowcol[1]] == 1
    
    hexes_surrounding_1st, second_ring, ring_3 = get_hex_rings_vectorized(
        unit_rowcol[0], unit_rowcol[1], 
        game.player_cities.ownership_map.shape[2], game.player_cities.ownership_map.shape[3],
    )
    check_hexes = jnp.concatenate([hexes_surrounding_1st, second_ring, ring_3], axis=0)

    player_blocked_dist = (
        game.player_cities.ownership_map[:, :, check_hexes[:, 0], check_hexes[:, 1]] > 2
    ).any()
    cs_blocked_dist = (game.cs_ownership_map[check_hexes[:, 0], check_hexes[:, 1]]).any()

    blocked = cs_blocked | player_blocked | ocean_blocked | lake_blocked | player_blocked_dist | cs_blocked_dist

    # Create update masks instead of doing .at operations
    unit_mask = jnp.zeros_like(game.units.unit_type, dtype=jnp.bool_)
    unit_mask = unit_mask.at[player_id, unit_int].set(True)
    
    # Apply all unit updates at once using where
    _type = jnp.where(unit_mask & ~blocked, 0, game.units.unit_type)
    _ap = jnp.where(unit_mask & ~blocked, 0, game.units.unit_ap)
    _rowcol = jnp.where(unit_mask[..., None] & ~blocked, 0, game.units.unit_rowcol)
    
    # Determine whether this proposed settle would be the capital city. This could occur if there
    # are currently no settles for this given player_id. We defer the "move capital to another city
    # due to warring" to another function
    is_cap_settle = game.player_cities.city_ids[player_id].sum() == 0
    id_to_use = jnp.where(is_cap_settle, 1, 2)
    #id_to_use = is_cap_settle * 1 + (1 - is_cap_settle) * 2

    # For the slot to use, we want to grab the first available open slot, which would just be argmin!
    slot_to_use = jnp.where(is_cap_settle, 0, game.player_cities.city_ids[player_id].argmin())
    
    # If it is the capital, we need to place the palace in it
    buildings_mask = jnp.zeros_like(game.player_cities.buildings_owned, dtype=jnp.bool_)
    buildings_mask = buildings_mask.at[player_id, slot_to_use, GameBuildings["palace"]._value_].set(is_cap_settle & ~blocked)
    _buildings = jnp.where(buildings_mask, 1, game.player_cities.buildings_owned)

    # Finally, do the settle, but also block
    _newly_placed_city, _is_coastal = game.player_cities.add_settle(unit_rowcol, player_id, slot_to_use, game)
    _ownership = _newly_placed_city.ownership_map
    _ownership = blocked * game.player_cities.ownership_map + (1 - blocked) * _ownership

    _potential_owned_rowcols = _newly_placed_city.potential_owned_rowcols
    _potential_owned_rowcols = blocked * game.player_cities.potential_owned_rowcols + (1 - blocked) * _potential_owned_rowcols
    _city_rowcol = game.player_cities.city_rowcols.at[player_id, slot_to_use].set(unit_rowcol)
    _city_rowcol = blocked * game.player_cities.city_rowcols + (1 - blocked) * _city_rowcol

    # Now adding yields based on the proposed_rowcol tile
    # Hill => (2f 2h), flat => (3f 1h)
    # Retains gold values
    base_yields = game.yield_map_players[player_id, unit_rowcol[0], unit_rowcol[1]][0]
    is_hill = game.elevation_map[unit_rowcol[0], unit_rowcol[1]] == HILLS_IDX
    is_snow = game.terrain_map[unit_rowcol[0], unit_rowcol[1]] == SNOW_IDX

    
    hill_yields = jnp.array(base_yields).at[0].set(2).at[1].set(2)
    flat_yields = jnp.array(base_yields).at[0].set(3).at[1].set(1)
    snow_yields = jnp.array(base_yields).at[0].set(0).at[1].set(0)

    _yields = jnp.where(is_snow, snow_yields, jnp.where(is_hill, hill_yields, flat_yields))
    
    yields_mask = jnp.zeros_like(game.player_cities.yields, dtype=jnp.bool_)
    yields_mask = yields_mask.at[player_id, slot_to_use].set(~blocked)
    _game_yields = jnp.where(yields_mask, _yields, game.player_cities.yields)
    _center_yields = jnp.where(yields_mask, _yields, game.player_cities.city_center_yields)

    # If blocked, revert back to previous city_ids
    city_ids_mask = jnp.zeros_like(game.player_cities.city_ids, dtype=jnp.bool_)
    city_ids_mask = city_ids_mask.at[player_id, slot_to_use].set(~blocked)
    _city_ids = jnp.where(city_ids_mask, id_to_use, game.player_cities.city_ids)

    # Population update
    has_merchant_navy = game.policies[player_id, SocialPolicies["merchant_navy"]] == 1
    pop_to_set = jnp.where(has_merchant_navy, 3, 1)
    pop_to_set = jnp.where(_is_coastal, pop_to_set, 1)

    # Create mask for population update
    pop_mask = jnp.zeros_like(game.player_cities.population, dtype=jnp.bool_)
    pop_mask = pop_mask.at[player_id, slot_to_use].set(~blocked)
    _pop = jnp.where(pop_mask, pop_to_set, game.player_cities.population)

    # increment (or not) social polich threshold
    # We add one to the n_cities b/c this would be the number of cities _if_ this city
    # is allowed to settle.
    new_social_policy_threshold = social_policy_threshold(
        n_cities=(game.player_cities.city_ids[player_id] > 0).sum() + 1,
        policies=game.policies[player_id]
    )
    new_social_policy_threshold = jnp.where(blocked, game.culture_threshold[player_id], new_social_policy_threshold)
    new_social_policy_threshold = game.culture_threshold.at[player_id].set(new_social_policy_threshold) 
    
    # When the first city is settled, agriculture is automatically researched. We can just 
    # set this as 1 for every city settled without issue, as game.technologies is a bool array
    tech_mask = jnp.zeros_like(game.technologies, dtype=jnp.bool_)
    tech_mask = tech_mask.at[player_id, Technologies["agriculture"]._value_].set(~blocked)
    _technologies = jnp.where(tech_mask, 1, game.technologies)

    _is_coastal = blocked + jnp.zeros_like(_is_coastal) + (1 - blocked) * _is_coastal
    _is_coastal = game.player_cities.is_coastal.at[player_id, slot_to_use].set(_is_coastal)

    # The final thing we need to do is add a road at the city center. This will allow for Civ-like city connections 
    # via roads.
    _road_map = jnp.where(
        blocked,
        game.road_map,
        game.road_map.at[unit_rowcol[0], unit_rowcol[1]].set(player_id[0] + 1)
    )
    
    # The string of Nones are for trade data,  which the settler unit type should never influence.
    minimal_game = create_minimal_update(
        game, _type, _ap, _rowcol, None, None, None, None,
        _ownership, _city_rowcol, _game_yields, _center_yields, _city_ids, _pop, _potential_owned_rowcols, _buildings, _is_coastal, _technologies, 
        new_social_policy_threshold, road_map=_road_map
    )
    return minimal_game, rowcol_to_hex(unit_rowcol)


### All of the below "improvement functions" need (1) a technology check and (2) a check 
# for if the selected improvement can be placed on the selected rowcol tile on the board.
def _do_tech_prereq_check(improvement_name, technologies, player_id):
    tech_req = Improvements[improvement_name].tech_prereq
    return technologies[player_id, tech_req]

def _do_tile_check(game, rowcol, player_id, improvement_name):
    out = improvement_mask_for_batch(
        game.improvement_bitfield,
        game.terrain_map,
        game.feature_map,
        game.elevation_map,
        game.freshwater_map,
        game.visible_resources_map_players[player_id[0]],
        game.lake_map,
        rowcol[None]
    )
    # Here, the second check ensures we cannot actually create an improvement 
    # on a tile that is jungle.
    improvement_id = Improvements[improvement_name]._value_
    is_jungle = game.feature_map[rowcol[0], rowcol[1]] == JUNGLE_IDX
    negate_from_jungle = is_jungle & (improvement_id != Improvements["chop_jungle"]._value_)
    return out[0, Improvements[improvement_name]._value_] & ~negate_from_jungle
   
def _do_ownership_check(game, rowcol, player_id):
    """
    There may be a scenario where a unit chooses an action category but all tiles on the map are -jnp.inf
    (i.e., invalid). Such a scenario may cause e.g., an improvement to be made outside of player_id's 
    owned territory.
    """
    return (game.player_cities.ownership_map[player_id[0], :, rowcol[0], rowcol[1]] >= 2).any()

def _do_engagement_check(game, rowcol, player_id, unit_int):
    return game.units.engaged_for_n_turns[player_id[0], unit_int] == 0

def do_farm(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("farm", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "farm")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _farm(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map

    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_pasture(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("pasture", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "pasture")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _pasture(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_mine(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("mine", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "mine")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _mine(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_boat(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("fishing_boat", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "fishing_boat")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _fishing_boat(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_plantation(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("plantation", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "plantation")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _plantation(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_camp(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("camp", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "camp")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _camp(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_quarry(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("quarry", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "quarry")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _quarry(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_lumber_mill(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("lumber_mill", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "lumber_mill")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _lumber_mill(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_fort(game, rowcol, player_id, unit_int, actions_map):
    tech_bool = _do_tech_prereq_check("fort", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "fort")
    can_do = tech_bool & tile_bool
    new_yields, new_features, new_improvements, new_roads = _fort(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_trading_post(game, rowcol, player_id, unit_int, actions_map):
    own_bool = _do_ownership_check(game, rowcol, player_id)
    tech_bool = _do_tech_prereq_check("trading_post", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "trading_post")
    can_do = tech_bool & tile_bool & own_bool
    new_yields, new_features, new_improvements, new_roads = _trading_post(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_road(game, rowcol, player_id, unit_int, actions_map):
    tech_bool = _do_tech_prereq_check("road", game.technologies, player_id)
    # Roads can be built on any non-ocean, non-mountain tile
    elevation_bool = (game.elevation_map[rowcol[0], rowcol[1]] > 0) & (game.elevation_map[rowcol[0], rowcol[1]] < 3)

    #tile_bool = _do_tile_check(game, rowcol, player_id,  "road")
    can_do = tech_bool & elevation_bool
    new_yields, new_features, new_improvements, new_roads = _road(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol, player_id
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_chop_forest(game, rowcol, player_id, unit_int, actions_map):
    tech_bool = _do_tech_prereq_check("chop_forest", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "chop_forest")
    can_do = tech_bool & tile_bool
    new_yields, new_features, new_improvements, new_roads = _chop_forest(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_chop_jungle(game, rowcol, player_id, unit_int, actions_map):
    tech_bool = _do_tech_prereq_check("chop_jungle", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "chop_jungle")
    can_do = tech_bool & tile_bool
    new_yields, new_features, new_improvements, new_roads = _chop_jungle(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)

def do_clear_marsh(game, rowcol, player_id, unit_int, actions_map):
    tech_bool = _do_tech_prereq_check("clear_marsh", game.technologies, player_id)
    tile_bool = _do_tile_check(game, rowcol, player_id,  "clear_marsh")
    can_do = tech_bool & tile_bool
    new_yields, new_features, new_improvements, new_roads = _clear_marsh(
        game.improvement_additional_yield_map, game.feature_map, game.improvement_map,
        game.visible_resources_map_players[player_id[0]], game.elevation_map, game.terrain_map,
        game.road_map, rowcol
    )

    new_yields = can_do * new_yields + (1 - can_do) * game.improvement_additional_yield_map
    new_features = can_do * new_features + (1 - can_do) * game.feature_map
    new_improvements = can_do * new_improvements + (1 - can_do) * game.improvement_map
    new_roads = can_do * new_roads + (1 - can_do) * game.road_map
    
    return create_minimal_update(
        game,
        improvement_additional_yield_map=new_yields,
        feature_map=new_features,
        improvement_map=new_improvements,
        road_map=new_roads
    ), rowcol_to_hex(rowcol)


def do_traderoute_transport(game, rowcol, player_id, unit_int, actions_map):
    """Moves a trade route from one of player_id's cities to another of player_id's cities"""
    # Shift (row, col) city locations to (2772,)
    # Ensure that we mask out (0,0) as a location, as this represents "no city"
    # After that we should limit the action-sampling ability to the player's cities
    flat_city_loc = hex_flat_index(game.player_cities.city_rowcols[player_id[0]])
    valid_mask = flat_city_loc != 0

    # Apply mask to both locations and logits
    valid_city_loc = jnp.where(valid_mask, flat_city_loc, -1)  # -1 for invalid
    valid_logits = jnp.where(valid_mask, actions_map[flat_city_loc], -jnp.inf)

    # Get argmax among valid cities
    best_idx = valid_logits.argmax()
    sampled_city = valid_city_loc[best_idx]

    new_unit_rowcol = game.idx_to_hex_rowcol[sampled_city]
    _executed_to_return = rowcol_to_hex(new_unit_rowcol)
    new_unit_rowcol = game.units.unit_rowcol.at[player_id[0], unit_int].set(new_unit_rowcol)

    # If a traderoute is being moved, then we can safely zero-out any previous yields it has generated
    new_yields = game.units.trade_yields.at[player_id[0], unit_int].set(0)
    new_trade_to_player = game.units.trade_to_player_int.at[player_id[0], unit_int].set(0)
    new_trade_to_city = game.units.trade_to_city_int.at[player_id[0], unit_int].set(0)
    new_trade_from_city = game.units.trade_from_city_int.at[player_id[0], unit_int].set(0)

    return create_minimal_update(game, unit_rowcol=new_unit_rowcol, trade_yields=new_yields, trade_to_player_int=new_trade_to_player, trade_to_city_int=new_trade_to_city, trade_from_city_int=new_trade_from_city), _executed_to_return

def do_traderoute_send(game, rowcol, player_id, unit_int, actions_map):
    """
    Let's try to avoid .reshape(), as there is always more than one way to reshape. We want to ensure we're 
    keeping the cities in player-order (e.g., [0, 0..., 1, 1,..., 2,2,...])
    """
    # (0) which cities can this traderoute reach?
    # E.g., (30, 2), (30,), (5,)
    #city_rowcols = jnp.concatenate([game.player_cities.city_rowcols[i] for i in range(6)], axis=0)
    #cs_rowcols = jnp.concatenate([game.cs_cities.city_rowcols[i] for i in range(12)], axis=0)
    city_rowcols = game.player_cities.city_rowcols.reshape(-1, 2)  # (6*max_cities, 2)
    cs_rowcols = game.cs_cities.city_rowcols.reshape(-1, 2)          # (12*max_cs_cities, 2)
    city_rowcols = jnp.concatenate([city_rowcols, cs_rowcols], axis=0)

    is_city = game.player_cities.city_ids.reshape(-1) > 0
    is_city = jnp.concatenate([is_city, jnp.ones(shape=(12,), dtype=is_city.dtype)])
    
    # In Civ V, you can only send routes between your own cities IFF you have a granary or workshop.
    # To make things simpler for us, we'll give a boost if these are made.
    in_this_city_mask = (game.player_cities.city_rowcols[player_id[0]] == rowcol).mean(-1) == 1
    can_send_food = (game.player_cities.can_trade_food[player_id[0]] * in_this_city_mask).sum() > 0
    can_send_prod = (game.player_cities.can_trade_prod[player_id[0]] * in_this_city_mask).sum() > 0
    
    distances = compute_all_distances_vectorized(rowcol, city_rowcols)

    # Cannot send trade route to city it is currently in
    mult_land = game.player_cities.trade_land_dist_mod[player_id[0], in_this_city_mask.argmax()]
    mult_sea = game.player_cities.trade_sea_dist_mod[player_id[0], in_this_city_mask.argmax()]
    mult = mult_land + mult_sea - 1  # need to sub 1 b/c both are base 1
    distances_mask = (distances > 0) & (distances < (LAND_TRADEROUTE_RANGE * mult))
     
    # (30 + 12,)
    flat_city_loc = hex_flat_index(city_rowcols)

    # We also cannot send trade routes to players or CS we haven't met
    n_cities = game.player_cities.city_ids.shape[-1]
    _have_met = game.have_met[player_id[0]].at[player_id[0]].set(True)  # can send to own cities
    _have_met_players = _have_met[:6].repeat(n_cities)
    _have_met_mask = jnp.concatenate([_have_met_players, _have_met[6:]]) > 0

    can_send_mask = (distances_mask & is_city & _have_met_mask) 
    can_send_tiles = can_send_mask * flat_city_loc

    route_possible = jnp.any(can_send_tiles > 0)
    logits = actions_map[can_send_tiles]          # 2772 → ~42 gather
    logits = jnp.where(can_send_tiles == 0, -jnp.inf, logits)

    sampled_city = can_send_tiles[logits.argmax()]
    sampled_city_rowcol = game.idx_to_hex_rowcol[sampled_city]

    # Find which city was selected by checking distances to the sampled location
    city_matches = ((city_rowcols - sampled_city_rowcol) == 0).mean(-1) == 1
    matching_city_idx = city_matches.argmax()
    
    n_cities = game.player_cities.city_ids.shape[-1]
    n_player_cities_total = 6 * n_cities
    
    # Determine if it's a city-state, other player's city, or own city
    is_cs = matching_city_idx >= n_player_cities_total
    is_player_city = matching_city_idx < n_player_cities_total
    
    # For player cities: determine which player and which city within that player
    player_idx_if_player = matching_city_idx // n_cities  # which player (0-5)
    city_idx_if_player = matching_city_idx % n_cities  # which city within that player
    is_own_city = is_player_city & (player_idx_if_player == player_id[0])
    
    # For city-states: get the city-state index
    cs_idx_if_cs = matching_city_idx - n_player_cities_total
    
    # Compute the final indices
    to_player_idx_bucket = jnp.where(
        is_cs,
        6 + cs_idx_if_cs,  # City-states start at index 6
        jnp.where(
            is_own_city,
            player_id[0],
            player_idx_if_player
        )
    )
    
    city_within_player = jnp.where(
        is_cs,
        0,  # City-states only have one city
        city_idx_if_player
    )

    # yield structure [standard 8, influence religios pressure]
    # Religious pressure iff the player has founded a religion. We make another simplifying
    # change to the mechanics. A player's trade routes can only send their religion.
    player_has_founded_religion = game.religious_tenets[player_id, MAX_IDX_PANTHEON: MAX_IDX_FOUNDER].sum() > 0
    religious_pressure_to_send = RELIGIOUS_PRESSURE_TRADEROUTE * player_has_founded_religion
    
    # If own city, then should send food and/or prod to city, receive nothing 
    own_from_city = jnp.zeros(shape=(10,))
    own_neither = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, religious_pressure_to_send])
    own_just_granary = jnp.array([4, 0, 0, 0, 0, 0, 0, 0, 0, religious_pressure_to_send])
    own_just_workshop = jnp.array([0, 4, 0, 0, 0, 0, 0, 0, 0, religious_pressure_to_send])
    own_both = jnp.array([4, 4, 0, 0, 0, 0, 0, 0, 0, religious_pressure_to_send])

    to_city_own = (
        (can_send_food & can_send_prod) * own_both +
        (can_send_food & ~can_send_prod) * own_just_granary +
        (~can_send_food & can_send_prod) * own_just_workshop +
        (~can_send_food & ~can_send_prod) * own_neither
    )
    to_city_own = jnp.concatenate([own_from_city[None], to_city_own[None]], axis=0)
    
    # If cs, then gold, influence, religious pressure
    city_int = in_this_city_mask.argmax() 
    gold_to_add = 1 + 0.05 * game.player_cities.yields[player_id[0], city_int, GOLD_IDX] + 0.5 * game.player_cities.resources_owned[player_id[0], city_int].sum()
    own_from_cs = jnp.array([0,  0, gold_to_add / 2, 0, 0, 0,  0, 0, 0, 0])
    cs_to_cs = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, INFLUENCE_PER_TURN_TRADEROUTE, religious_pressure_to_send])

    # Bonus from Social Policies (Merchant confederacy)
    # (6, 5, 8) -> (8,)
    culture_traderoute_bonus = game.culture_info.cs_trade_route_yields[player_id[0], in_this_city_mask.argmax()]
    to_city_cs = jnp.concatenate([own_from_cs[None], cs_to_cs[None]], axis=0) + culture_traderoute_bonus
    
    # If other player, then gold, science, religious pressure, tourism
    # We'll make some simplifications here...
    # 1 base gold + 5% of gold output from each city +0.5 for each improved resource in each city, +25% gold if city on river
    # 2 science for ~every age  finished
    # The tourism is actually a multiplier?
    caravansary_gold = game.player_cities.trade_gold_add_owner[player_id[0], in_this_city_mask.argmax()]
    market_dest = game.player_cities.trade_gold_add_dest[player_id[0], in_this_city_mask.argmax()]
    science_to_add = (game.technologies[player_id[0]].sum() / 10) * SCIENCE_PER_ERA_TRADEROUTE
    tourism_to_add = 0.25

    # Bonus from religion: Religious Troubadours
    has_troub = game.religious_tenets[player_id[0], ReligiousTenets["religious_troubadours"]]

    own_from_player = jnp.array([0, 0, gold_to_add + caravansary_gold, 3 * has_troub, 0, science_to_add, 0, 0, 0, 0])
    player_to_player = jnp.array([0, 0, gold_to_add / 2 + market_dest, 0, 0, science_to_add / 2, 0, tourism_to_add, 0, religious_pressure_to_send])

    to_city_player = jnp.concatenate([own_from_player[None], player_to_player[None]], axis=0)
    
    new_yields = (
        is_own_city * to_city_own +
        is_cs * to_city_cs +
        (~is_own_city & ~is_cs) * to_city_player
    )
    
    # Now we need to zero-out all computations IFF no trade routes are possible. This is mainly for safety
    new_yields = jnp.where(
        route_possible,
        new_yields,
        game.units.trade_yields[player_id[0], unit_int] * 0
    )

    new_to_player_int = jnp.where(
        route_possible,
        to_player_idx_bucket,
        0
    )

    new_to_city_int = jnp.where(
        route_possible,
        city_within_player,
        0
    )

    new_from_city_int = jnp.where(
        route_possible,
        city_int,
        0
    )
    
    new_to_player_int = game.units.trade_to_player_int.at[player_id[0], unit_int].set(new_to_player_int)
    new_to_city_int = game.units.trade_to_city_int.at[player_id[0], unit_int].set(new_to_city_int)
    new_from_city_int = game.units.trade_from_city_int.at[player_id[0], unit_int].set(new_from_city_int)
    new_yields = game.units.trade_yields.at[player_id[0], unit_int].set(new_yields)

    _executed_to_return = jnp.where(
        route_possible,
        rowcol_to_hex(sampled_city_rowcol),
        -1
    )

    return create_minimal_update(game, trade_to_player_int=new_to_player_int, trade_to_city_int=new_to_city_int, trade_from_city_int=new_from_city_int, trade_yields=new_yields), _executed_to_return

def do_combat(game, rowcol, player_id, unit_int, actions_map):
    """Combat is resolved in GameState.step_unitsv2 -> scan over _resolve_actions()
    We keep this identity function here to allow units to select combat action if they can.
    E.g., using len(UnitActionCategories) as a reference for the number of categories.
    As this is the identity function, the compiler should(?) remove it from the HLO?
    """
    return create_minimal_update(game), rowcol_to_hex(rowcol)

# 0th action category is unit movement. This is already handled by default before the action-category functions 
# are applied to the gamestate. Ergo, we can use an identity function. 
ALL_ACTION_FUNCTIONS = [
    lambda x, rowcol, player_id, unit_int, actions_map: (create_minimal_update(x), rowcol_to_hex(rowcol)),
    do_settlev2,
    do_farm,
    do_pasture,
    do_mine,
    do_boat,
    do_plantation,
    do_camp,
    do_quarry,
    do_lumber_mill,
    do_fort,
    do_trading_post,
    do_road,
    do_chop_forest,
    do_chop_jungle,
    do_clear_marsh,
    do_traderoute_transport,
    do_traderoute_send,
    do_combat
]

ALL_ACTION_ENGAGEMENT_TURNS = jnp.array([
    0,
    0, 
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    0,
    14,
    0
])
