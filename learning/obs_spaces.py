from __future__ import annotations
from functools import partial
from flax import struct
from jax._src.api import F
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, Union
import jax

from dataclasses import replace, fields, is_dataclass
from typing import TYPE_CHECKING
from game.buildings import GameBuildings
from game.constants import FOOD_IDX, GOLD_IDX, PROD_IDX

from game.primitives import GameState, CultureInfo, Cities, ReligionInfo
from game.techs import Technologies
from game.units import ALL_UNIT_COMBAT, Units

if TYPE_CHECKING:
    from learning.algorithms import Algorithm
    from learning.buffers import ReplayBuffer
    from learning.metrics import EpisodicMetrics
    from learning.goals import Goals
    from learning.obs_spaces import ObservationSpace

def _copy_overlap(dst_obj, src_obj, dst_type=None):
    """
    Recursively copy only the fields that exist on `dst_obj` from `src_obj`.
    - If both are dataclasses, walk the destination's fields and pull
      matching attributes from the source (recurse).
    - Otherwise return src_obj (leaf tensors/arrays).
    """
    # If we don't have a destination instance yet but we know the type,
    # synthesize one from src using only the type's declared fields.
    if dst_obj is None and dst_type is not None and is_dataclass(dst_type) and is_dataclass(src_obj):
        sub_kwargs = {}
        for f in fields(dst_type):
            if hasattr(src_obj, f.name):
                v_src = getattr(src_obj, f.name)
                # If this field is itself a dataclass, recurse with the field type
                if is_dataclass(f.type) and is_dataclass(v_src):
                    sub_kwargs[f.name] = _copy_overlap(None, v_src, f.type)
                else:
                    sub_kwargs[f.name] = v_src
        return dst_type(**sub_kwargs)

    # Normal path: both instances exist and are dataclasses -> project by dst's fields
    if is_dataclass(dst_obj) and is_dataclass(src_obj):
        updates = {}
        for f in fields(dst_obj):
            if hasattr(src_obj, f.name):
                v_dst = getattr(dst_obj, f.name)
                v_src = getattr(src_obj, f.name)
                if is_dataclass(v_dst) and is_dataclass(v_src):
                    updates[f.name] = _copy_overlap(v_dst, v_src)
                else:
                    updates[f.name] = v_src
        return replace(dst_obj, **updates)

    # Leaf (arrays, numbers, etc.): take the source value
    return src_obj

@struct.dataclass
class ObservationSpace(ABC):
    """"""
    @classmethod
    @abstractmethod
    def create(cls, game: GameState, *args, **kwargs) -> "ObservationSpace":
        """"""
        raise NotImplementedError()

    @abstractmethod
    def gather_observation(self, games: GameState, algorithn: Algorithm, goals: Goals, episode_metrics: EpisodicMetrics, replays: ReplayBuffer) -> "ObservationSpace":
        """"""
        raise NotImplementedError()

    @abstractmethod
    def form_observation(self) -> jnp.ndarray:
        """"""
        raise NotImplementedError()

    @abstractmethod
    def update(self) ->  "ObservationSpace":
        raise  NotImplementedError()

@struct.dataclass
class TerraNovaUnitsObservation:
    unit_type: jnp.ndarray
    unit_rowcol: jnp.ndarray
    unit_ap: jnp.ndarray
    engaged_for_n_turns: jnp.ndarray
    engaged_action_id: jnp.ndarray
    trade_to_player_int: jnp.ndarray
    trade_to_city_int: jnp.ndarray
    trade_from_city_int: jnp.ndarray
    trade_yields: jnp.ndarray
    combat_bonus_accel: jnp.ndarray
    health: jnp.ndarray


@struct.dataclass
class TerraNovaCultureInfoObservation:
    building_yields: jnp.ndarray
    yields_per_kill: jnp.ndarray
    honor_finisher_yields_per_kill: jnp.ndarray
    cs_resting_influence: jnp.ndarray
    cs_trade_route_yields: jnp.ndarray
    additional_yield_map: jnp.ndarray

@struct.dataclass
class TerraNovaCitystateInfoObservation:
    religious_population: jnp.ndarray
    relationships: jnp.ndarray
    influence_level: jnp.ndarray
    cs_type: jnp.ndarray
    quest_type: jnp.ndarray
    culture_tracker_mine: jnp.ndarray
    faith_tracker_mine: jnp.ndarray
    tech_tracker_mine: jnp.ndarray
    trade_tracker_mine: jnp.ndarray
    religion_tracker_mine: jnp.ndarray
    wonder_tracker_mine: jnp.ndarray
    resource_tracker_mine: jnp.ndarray
    culture_tracker_lead: jnp.ndarray
    faith_tracker_lead: jnp.ndarray
    tech_tracker_lead: jnp.ndarray
    trade_tracker_lead: jnp.ndarray
    religion_tracker_lead: jnp.ndarray
    wonder_tracker_lead: jnp.ndarray
    resource_tracker_lead: jnp.ndarray
    city_rowcols: jnp.ndarray


@struct.dataclass
class TerraNovaPlayerCitiesObservation:
    city_ids: jnp.ndarray
    city_rowcols: jnp.ndarray
    ownership_map: jnp.ndarray
    yields: jnp.ndarray
    city_center_yields: jnp.ndarray
    building_yields: jnp.ndarray
    population: jnp.ndarray
    worked_slots: jnp.ndarray
    specialist_slots: jnp.ndarray
    gw_slots: jnp.ndarray
    food_reserves: jnp.ndarray
    growth_carryover: jnp.ndarray
    prod_reserves: jnp.ndarray
    prod_carryover: jnp.ndarray
    is_constructing: jnp.ndarray
    bldg_maintenance: jnp.ndarray
    defense: jnp.ndarray
    hp: jnp.ndarray
    buildings_owned: jnp.ndarray
    resources_owned: jnp.ndarray
    additional_yield_map: jnp.ndarray
    is_coastal: jnp.ndarray
    religion_info: TerraNovaReligionInfoObservation 
    culture_reserves_for_border: jnp.ndarray
    great_person_points: jnp.ndarray


@struct.dataclass
class TerraNovaReligionInfoObservation:
    religious_population: jnp.ndarray
    religious_tenets_per_city: jnp.ndarray
    building_yields: jnp.ndarray
    additional_yield_map: jnp.ndarray
    cs_perturn_influence_cumulative: jnp.ndarray
    player_perturn_influence_cumulative: jnp.ndarray

@struct.dataclass
class TerraNovaObservation:
    ### BASE INFORMATION FROM ENV ###
    elevation_map: jnp.ndarray
    terrain_map: jnp.ndarray
    edge_river_map: jnp.ndarray
    lake_map: jnp.ndarray
    feature_map: jnp.ndarray
    nw_map: jnp.ndarray
    cs_ownership_map: jnp.ndarray
    units: TerraNovaUnitsObservation
    technologies: jnp.ndarray
    policies: jnp.ndarray
    player_cities: TerraNovaPlayerCitiesObservation 
    yield_map_players: jnp.ndarray
    visible_resources_map_players: jnp.ndarray
    science_reserves: jnp.ndarray
    culture_reserves: jnp.ndarray
    faith_reserves: jnp.ndarray
    is_researching: jnp.ndarray
    num_trade_routes: jnp.ndarray
    cs_perturn_influence: jnp.ndarray
    num_delegates: jnp.ndarray
    culture_threshold: jnp.ndarray
    religious_tenets: jnp.ndarray
    free_techs: jnp.ndarray
    free_policies: jnp.ndarray
    great_works: jnp.ndarray
    culture_info: TerraNovaCultureInfoObservation
    improvement_additional_yield_map: jnp.ndarray
    improvement_map: jnp.ndarray
    road_map: jnp.ndarray
    gpps: jnp.ndarray
    gp_threshold: jnp.ndarray
    golden_age_turns: jnp.ndarray
    tourism_total: jnp.ndarray
    culture_total: jnp.ndarray
    citystate_info: TerraNovaCitystateInfoObservation
    visibility_map: jnp.ndarray
    trade_offers: jnp.ndarray
    trade_ledger: jnp.ndarray
    trade_length_ledger: jnp.ndarray
    trade_gpt_adjustment: jnp.ndarray
    trade_resource_adjustment: jnp.ndarray
    have_met: jnp.ndarray
    at_war: jnp.ndarray
    has_sacked: jnp.ndarray
    treasury: jnp.ndarray
    happiness: jnp.ndarray

    ### Derived Information ###
    # This is info not directly contained in the environment but is exposed to 
    # each player. E.g., demographcs 1st and last
    most_population: jnp.ndarray
    least_population: jnp.ndarray
    most_crop_yield: jnp.ndarray
    least_crop_yield: jnp.ndarray
    most_manufactured_goods: jnp.ndarray
    least_manufactured_goods: jnp.ndarray
    most_gnp: jnp.ndarray
    least_gnp: jnp.ndarray
    most_land: jnp.ndarray
    least_land: jnp.ndarray
    most_soldiers: jnp.ndarray
    least_soldiers: jnp.ndarray
    most_approval: jnp.ndarray
    least_approval: jnp.ndarray
    most_literacy: jnp.ndarray
    least_literacy: jnp.ndarray
    
    player_id: jnp.ndarray
    current_turn: jnp.ndarray

@jax.jit
def update_field(fog_of_war, self_field, games_field):
    player_fog = fog_of_war
    can_see = (player_fog == 0)
    return jnp.where(can_see, games_field, self_field)

@jax.jit
def update_field_per_visibility(fog_of_war, player_cities, _self, _games):
    """
    If the given player can see the tiles that contains the city center, then the player 
    is able to ascertain information about the city

    fow: (n_games, 6, 42, 66)
    """
    # (n_games, 6, n_cities, 2) and (n_games, 6, n_cities)
    city_locs = player_cities.city_rowcols
    is_city = player_cities.city_ids > 0
    
    # Extract row and col indices
    rows = city_locs[..., 0]  # (n_games, 6, n_cities)
    cols = city_locs[..., 1]  # (n_games, 6, n_cities)
    
    # Create indices for the batch dimensions
    n_games, n_players, n_cities = rows.shape

    # We need to check: can player j see player k's city l in game i?
    # This requires checking fog_of_war[i, j, row_of_k's_city_l, col_of_k's_city_l]
    game_idx = jnp.arange(n_games)[:, None, None, None]  # (n_games, 1, 1, 1)
    observer_idx = jnp.arange(n_players)[None, :, None, None]  # (1, n_players, 1, 1)
    owner_idx = jnp.arange(n_players)[None, None, :, None]  # (1, 1, n_players, 1)
    
    # Broadcast city locations to (n_games, 1, n_players, n_cities)
    rows_broadcast = rows[:, None, :, :]  # (n_games, 1, n_players, n_cities)
    cols_broadcast = cols[:, None, :, :]  # (n_games, 1, n_players, n_cities)
    
    # Check visibility: can each observer see each city?
    # This gives us (n_games, n_players, n_players, n_cities)
    can_see = fog_of_war[game_idx, observer_idx, rows_broadcast, cols_broadcast]
    
    # Also broadcast is_city to match
    is_city_broadcast = is_city[:, None, :, :]  # (n_games, 1, n_players, n_cities)
    city_visible = (can_see == 0) & is_city_broadcast
    
    # Now update fields - broadcast games data to add observer dimension
    _is_coastal = jnp.where(
        city_visible,
        _games.player_cities.is_coastal[:, None, :, :],  # Broadcast to (n_games, 1, n_players, n_cities)
        _self.player_cities__is_coastal,  # Already (n_games, 6, 6, n_cities)
    )
    
    _defense = jnp.where(
        city_visible,
        _games.player_cities.defense[:, None, :, :],
        _self.player_cities__defense,
    )
    
    _hp = jnp.where(
        city_visible,
        _games.player_cities.hp[:, None, :, :],
        _self.player_cities__hp,
    )
    
    _population = jnp.where(
        city_visible,
        _games.player_cities.population[:, None, :, :],
        _self.player_cities__population,
    )

    _ids = jnp.where(
        city_visible,
        _games.player_cities.city_ids[:, None],
        _self.player_cities__city_ids
    )
    
    _rowcols = jnp.where(
        city_visible[..., None],
        _games.player_cities.city_rowcols[:, None],
        _self.player_cities__city_rowcols
    )

    return _is_coastal, _defense, _hp, _population, _ids, _rowcols
    

@struct.dataclass
class TerraNovaObservationSpaceTracker:
    """
    We need to keep track of what we saw previously. This is because we should not update information
    in regions of the map the agent cannot currently see. E.g.,:
    (1) Agent sees Truffles resource on some tile
    (2) Agent walks away and can no longer see the tile
    (3) Another agent improves the resource

    In the above scenario, the agent from (1-2) **should not** be able to see the improvement.
    The only thing the agent should see if the Truffles resource on the tile. 

    One way we can accomplish this is by tracking the previous turn's maps and updating only the 
    tiles that are marked with 2 in the fog of war map.
    """
    elevation_map: jnp.ndarray
    terrain_map: jnp.ndarray
    edge_river_map: jnp.ndarray
    lake_map: jnp.ndarray
    feature_map: jnp.ndarray
    nw_map: jnp.ndarray
    cs_ownership_map: jnp.ndarray
    yield_map_players: jnp.ndarray
    visible_resources_map_players: jnp.ndarray
    improvement_additional_yield_map: jnp.ndarray
    improvement_map: jnp.ndarray
    road_map: jnp.ndarray
    player_cities__ownership_map: jnp.ndarray
    player_cities__is_coastal: jnp.ndarray
    player_cities__defense: jnp.ndarray
    player_cities__hp: jnp.ndarray
    player_cities__population: jnp.ndarray
    player_cities__city_ids: jnp.ndarray
    player_cities__city_rowcols: jnp.ndarray

    @classmethod
    def create(
        cls,
        n_games,
        games,
        sharding_ref=None
    ):
        """
        Build a TerraNovaObservationSpaceTracker from a 'games' container whose fields
        mirror the dataclass fields. If arrays are sharded (via jax.make_array_from_single_device_arrays),
        pass a reference array as `sharding_ref` to place the derived 1D zeros on the same sharding.

        Here we keep everything as float32, as we can safely cast from any relevatn dtype into float32.
        The user should then be able to do whatever they want with it from there. 

        These will all be initialized to -1, which means the values are not currently known
        """
        # --- pick a sharding reference automatically if not provided ---
        if sharding_ref is None:
            # Prefer a big per-game tensor so 1D zeros can shard over axis 0.
            for name in ("feature_map", "yield_map_players", "cs_ownership_map"):
                if hasattr(games, name):
                    cand = getattr(games, name)
                    if isinstance(cand, jax.Array) and cand.shape[1] == n_games:
                        sharding_ref = cand
                        break
        
        n_devices = games.feature_map.shape[0]
        n_games = games.feature_map.shape[1]
        n_cities = games.player_cities.city_ids.shape[-1]
        
        # We cannot safely inherit dtypes from the GameState object, as some of these fields are unsigned integers. The safest
        # thing to do is case them into int32!
        _cls = cls(
            elevation_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.feature_map.shape[-2:]), dtype=jnp.int32) - 1,
            terrain_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.feature_map.shape[-2:]), dtype=jnp.int32) - 1,
            feature_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.feature_map.shape[-2:]), dtype=jnp.int32) - 1,
            lake_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.feature_map.shape[-2:]), dtype=jnp.int32) - 1,
            nw_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.feature_map.shape[-2:]), dtype=jnp.int32) - 1,
            edge_river_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.edge_river_map.shape[-3:]), dtype=jnp.int32) - 1,
            cs_ownership_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.cs_ownership_map.shape[-2:]), dtype=jnp.int32) - 1,
            
            yield_map_players=jnp.zeros(shape=games.yield_map_players.shape) - 1,
            
            visible_resources_map_players=jnp.zeros(shape=games.visible_resources_map_players.shape, dtype=jnp.int32) - 1,
            improvement_additional_yield_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.improvement_additional_yield_map.shape[-3:])) - 1,
            
            improvement_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.improvement_map.shape[-2:]), dtype=jnp.int32) - 1,
            road_map=jnp.zeros(shape=(n_devices, n_games, 6, *games.road_map.shape[-2:]), dtype=jnp.int32) - 1,
            player_cities__ownership_map=jnp.zeros(shape=(n_devices, n_games, 6, 6, *games.player_cities.ownership_map.shape[-3:]), dtype=jnp.int32) - 1,
            player_cities__is_coastal=jnp.zeros(shape=(n_devices, n_games, 6, 6, games.player_cities.is_coastal.shape[-1]), dtype=jnp.int32) - 1,
            player_cities__defense=jnp.zeros(shape=(n_devices, n_games, 6, 6, games.player_cities.defense.shape[-1])) - 1,
            player_cities__hp=jnp.zeros(shape=(n_devices, n_games, 6, 6, games.player_cities.hp.shape[-1])) - 1,
            player_cities__population=jnp.zeros(shape=(n_devices, n_games, 6, 6, games.player_cities.population.shape[-1])) - 1,
            player_cities__city_ids=jnp.zeros(shape=(n_devices, n_games, 6, 6, games.player_cities.population.shape[-1]), dtype=jnp.int32) - 1,
            player_cities__city_rowcols=jnp.zeros(shape=(n_devices, n_games, 6, 6, *games.player_cities.city_rowcols.shape[-2:]), dtype=jnp.int32) - 1,
        )

        return  jax.tree_map(lambda x: jax.device_put(x, sharding_ref.sharding), _cls)

    def reset(self):
        """Reset all fields to -1 while preserving their shapes and dtypes."""
        return self.replace(
            elevation_map=self.elevation_map * 0 - 1,
            terrain_map=self.terrain_map * 0 - 1,
            edge_river_map=self.edge_river_map * 0 - 1,
            lake_map=self.lake_map * 0 - 1,
            feature_map=self.feature_map * 0 - 1,
            nw_map=self.nw_map * 0 - 1,
            cs_ownership_map=self.cs_ownership_map * 0 - 1,
            yield_map_players=self.yield_map_players * 0 - 1,
            visible_resources_map_players=self.visible_resources_map_players * 0 - 1,
            improvement_additional_yield_map=self.improvement_additional_yield_map * 0 - 1,
            improvement_map=self.improvement_map * 0 - 1,
            road_map=self.road_map * 0 - 1,
            player_cities__ownership_map=self.player_cities__ownership_map * 0 - 1,
            player_cities__is_coastal=self.player_cities__is_coastal * 0 - 1,
            player_cities__defense=self.player_cities__defense * 0 - 1,
            player_cities__hp=self.player_cities__hp * 0 - 1,
            player_cities__population=self.player_cities__population * 0 - 1,
            player_cities__city_ids=self.player_cities__city_ids * 0 - 1,
            player_cities__city_rowcols=self.player_cities__city_rowcols * 0 - 1,
        )

    def _update_and_grab_obs(self, games, player_id):
        """
        fow: (n_games, 6, 42, 66)

        This function is called from within a shard_map'ed context. We therefore lose the device axis.
        """
        # Before this dataclass is created, we should have already run the fog of war compution on the gamestate
        fog_of_war = games.visibility_map
        
        _feature_map = update_field(
            fog_of_war, self.feature_map, games.feature_map[:, None]
        )
        _elevation_map = update_field(
            fog_of_war, self.elevation_map, games.elevation_map[:, None]
        )
        _terrain_map = update_field(
            fog_of_war, self.terrain_map, games.terrain_map[:, None]
        )
        _lake_map = update_field(
            fog_of_war, self.lake_map, games.lake_map[:, None]
        )
        _nw_map = update_field(
            fog_of_war, self.nw_map, games.nw_map[:, None]
        )
        _edge_river_map = update_field(
            fog_of_war[..., None], self.edge_river_map, games.edge_river_map[:, None]
        )
        _cs_ownership_map = update_field(
            fog_of_war, self.cs_ownership_map, games.cs_ownership_map[:, None]
        )
        _yield_map_players = update_field(
            fog_of_war[..., None], self.yield_map_players, games.yield_map_players
        )
        _visible_resources_map_players = update_field(
            fog_of_war, self.visible_resources_map_players, games.visible_resources_map_players
        )
        _improvement_additional_yield_map = update_field(
                fog_of_war[..., None], self.improvement_additional_yield_map, games.improvement_additional_yield_map[:, None]
        )
        _improvement_map = update_field(
                fog_of_war, self.improvement_map, games.improvement_map[:, None]
        )
        _road_map = update_field(
                fog_of_war, self.road_map, games.road_map[:, None]
        )
        _ownership_map = update_field(
                fog_of_war[:, :, None, None], self.player_cities__ownership_map, games.player_cities.ownership_map[:, None]
        )

        # The following fields are based on visibility of the given tiles on a per-city basis
        _is_coastal, _defense, _hp, _population, _ids, _rowcols = update_field_per_visibility(
            fog_of_war, games.player_cities, self, games
        )

        _self = self.replace(
            feature_map=_feature_map,
            elevation_map=_elevation_map,
            terrain_map=_terrain_map,
            lake_map=_lake_map,
            nw_map=_nw_map,
            edge_river_map=_edge_river_map,
            cs_ownership_map=_cs_ownership_map,
            yield_map_players=_yield_map_players,
            visible_resources_map_players=_visible_resources_map_players,
            improvement_additional_yield_map=_improvement_additional_yield_map,
            improvement_map=_improvement_map,
            road_map=_road_map,
            player_cities__ownership_map=_ownership_map,
            player_cities__is_coastal=_is_coastal,
            player_cities__defense=_defense,
            player_cities__hp=_hp,
            player_cities__population=_population,
            player_cities__city_ids=_ids, 
            player_cities__city_rowcols=_rowcols
        )

        # Now time to form the observation
        games_index = jnp.arange(player_id.shape[0]) 

        # --- nested dataclasses from games ---
        culture_info = TerraNovaCultureInfoObservation(
            building_yields=games.culture_info.building_yields[games_index, player_id],
            yields_per_kill=games.culture_info.yields_per_kill[games_index, player_id],
            honor_finisher_yields_per_kill=games.culture_info.honor_finisher_yields_per_kill[games_index, player_id],
            cs_resting_influence=games.culture_info.cs_resting_influence[games_index, player_id],
            cs_trade_route_yields=games.culture_info.cs_trade_route_yields[games_index, player_id],
            additional_yield_map=games.culture_info.additional_yield_map[games_index, player_id],
        )

        # Allyship -> returns player int when ally exists, otherwise -1
        is_ally = games.citystate_info.relationships == 2
        has_ally = is_ally.any(axis=2)
        ally_indices = jnp.argmax(is_ally, axis=2)
        allies = jnp.where(has_ally, ally_indices, -1)
        
        # Can see the rowcol location of the CS we have met
        _have_met = games.have_met[games_index, player_id][:, 6:]
        _cs_rowcols = games.cs_cities.city_rowcols[:, :, 0]
        _seen_rowcols = jnp.where(_have_met[..., None], _cs_rowcols, jnp.array([-1, -1]))

        _cs_type = jnp.where(_have_met, games.citystate_info.cs_type, -1)
        _quest_type = jnp.where(_have_met, games.citystate_info.quest_type, -1)

        citystate_info = TerraNovaCitystateInfoObservation(
            religious_population=games.citystate_info.religious_population,
            relationships=allies,
            influence_level=games.citystate_info.influence_level[games_index[:, None], jnp.arange(12), player_id[:, None]],
            cs_type=_cs_type,
            quest_type=_quest_type,
            culture_tracker_mine=games.citystate_info.culture_tracker[games_index, player_id],
            faith_tracker_mine=games.citystate_info.faith_tracker[games_index, player_id],
            tech_tracker_mine=games.citystate_info.tech_tracker[games_index, player_id],
            trade_tracker_mine=games.citystate_info.trade_tracker[games_index, player_id],
            religion_tracker_mine=games.citystate_info.religion_tracker[games_index, player_id],
            wonder_tracker_mine=games.citystate_info.wonder_tracker[games_index, player_id],
            resource_tracker_mine=games.citystate_info.resource_tracker[games_index, player_id],
            culture_tracker_lead=games.citystate_info.culture_tracker.max(-1),
            faith_tracker_lead=games.citystate_info.faith_tracker.max(-1),
            tech_tracker_lead=games.citystate_info.tech_tracker.max(-1),
            trade_tracker_lead=games.citystate_info.trade_tracker.max(-2),
            religion_tracker_lead=games.citystate_info.religion_tracker.max(-1),
            wonder_tracker_lead=games.citystate_info.wonder_tracker.max(-1),
            resource_tracker_lead=games.citystate_info.resource_tracker.max(-1),
            city_rowcols=_seen_rowcols,
        )
        
        # For unit information, we use the current visibility of the given player
        fog_of_war_me = fog_of_war[games_index, player_id]

        all_unit_rowcols = games.units.unit_rowcol
        unit_rows = all_unit_rowcols[..., 0]  # (num_games, 6, max_num_units)
        unit_cols = all_unit_rowcols[..., 1]  # (num_games, 6, max_num_units)
        valid_units = games.units.unit_type > 0
        num_games, num_players, max_units = unit_rows.shape
        #game_idx = jnp.arange(num_games)[:, None, None].repeat(num_players, axis=1).repeat(max_units, axis=2)
        game_idx = jnp.arange(num_games)[:, None, None]

        can_see_unit = jnp.where(
            valid_units,
            fog_of_war_me[game_idx, unit_rows, unit_cols] == 0,
            False
        ).at[games_index, player_id, :].set(True)  # (num_games, 6, max_num_units)

        units = TerraNovaUnitsObservation(
            unit_type=jnp.where(can_see_unit, games.units.unit_type, -1),
            unit_rowcol=jnp.where(can_see_unit[..., None], games.units.unit_rowcol, jnp.array([-1, -1])),
            unit_ap=jnp.where(can_see_unit, games.units.unit_ap, -1),
            engaged_for_n_turns=games.units.engaged_for_n_turns[games_index, player_id],
            engaged_action_id=games.units.engaged_action_id[games_index, player_id],
            trade_to_player_int=games.units.trade_to_player_int[games_index, player_id],
            trade_to_city_int=games.units.trade_to_city_int[games_index, player_id],
            trade_from_city_int=games.units.trade_from_city_int[games_index, player_id],
            trade_yields=games.units.trade_yields[games_index, player_id],
            combat_bonus_accel=jnp.where(can_see_unit, games.units.combat_bonus_accel, -1),
            health=jnp.where(can_see_unit, games.units.health, -1),
        )
        
        religion_info = TerraNovaReligionInfoObservation(
            religious_population=games.player_cities.religion_info.religious_population[games_index, player_id],
            religious_tenets_per_city=games.player_cities.religion_info.religious_tenets_per_city[games_index, player_id],
            building_yields=games.player_cities.religion_info.building_yields[games_index, player_id],
            #pressure=games.player_cities.religion_info.pressure[games_index, player_id],
            additional_yield_map=games.player_cities.religion_info.additional_yield_map[games_index, player_id],
            cs_perturn_influence_cumulative=games.player_cities.religion_info.cs_perturn_influence_cumulative[games_index, player_id],
            player_perturn_influence_cumulative=games.player_cities.religion_info.player_perturn_influence_cumulative[games_index, player_id],
        )
        
        player_cities = TerraNovaPlayerCitiesObservation(
            city_ids=_self.player_cities__city_ids[games_index, player_id],
            city_rowcols=_self.player_cities__city_rowcols[games_index, player_id],
            ownership_map=_self.player_cities__ownership_map[games_index, player_id],
            yields=games.player_cities.yields[games_index, player_id],
            city_center_yields=games.player_cities.city_center_yields[games_index, player_id],
            building_yields=games.player_cities.building_yields[games_index, player_id],
            population=_self.player_cities__population[games_index, player_id],
            worked_slots=games.player_cities.worked_slots[games_index, player_id],
            specialist_slots=games.player_cities.specialist_slots[games_index, player_id],
            gw_slots=games.player_cities.gw_slots[games_index, player_id],
            food_reserves=games.player_cities.food_reserves[games_index, player_id],
            growth_carryover=games.player_cities.growth_carryover[games_index, player_id],
            prod_reserves=games.player_cities.prod_reserves[games_index, player_id],
            prod_carryover=games.player_cities.prod_carryover[games_index, player_id],
            is_constructing=games.player_cities.is_constructing[games_index, player_id],
            bldg_maintenance=games.player_cities.bldg_maintenance[games_index, player_id],
            defense=_self.player_cities__defense[games_index, player_id],
            hp=_self.player_cities__hp[games_index, player_id],
            buildings_owned=games.player_cities.buildings_owned[games_index, player_id],
            resources_owned=games.player_cities.resources_owned[games_index, player_id],
            additional_yield_map=games.player_cities.additional_yield_map[games_index, player_id],
            is_coastal=_self.player_cities__is_coastal[games_index, player_id],
            religion_info=religion_info,
            culture_reserves_for_border=games.player_cities.culture_reserves_for_border[games_index, player_id],
            great_person_points=games.player_cities.great_person_points[games_index, player_id],
        )
        
        # (num_games, 6) and (num_games, 6, 6)
        _have_met = games.have_met[games_index, player_id][:, :6].at[games_index, player_id].set(True)
        double_mask = _have_met[..., None] & _have_met[:, None, :]

        _culture_total = jnp.where(_have_met, games.culture_total, -1)
        _at_war = jnp.where(double_mask, games.at_war, -1)
        _has_sacked = jnp.where(double_mask, games.has_sacked, -1)
        _tourism_total = jnp.where(double_mask, games.tourism_total, -1)
        _num_delegates = jnp.where(_have_met, games.num_delegates, -1)

        full_obs = TerraNovaObservation(
            # Base info (copied straight through; sharded arrays stay sharded)
            elevation_map=_self.elevation_map[games_index, player_id],
            terrain_map=_self.terrain_map[games_index, player_id],
            edge_river_map=_self.edge_river_map[games_index, player_id],
            lake_map=_self.lake_map[games_index, player_id],
            feature_map=_self.feature_map[games_index, player_id],
            nw_map=_self.nw_map[games_index, player_id],
            cs_ownership_map=_self.cs_ownership_map[games_index, player_id],
            units=units,
            technologies=games.technologies[games_index, player_id],
            policies=games.policies[games_index, player_id],
            player_cities=player_cities,
            yield_map_players=_self.yield_map_players[games_index, player_id],
            visible_resources_map_players=_self.visible_resources_map_players[games_index, player_id],
            science_reserves=games.science_reserves[games_index, player_id],
            culture_reserves=games.culture_reserves[games_index, player_id],
            faith_reserves=games.faith_reserves[games_index, player_id],
            is_researching=games.is_researching[games_index, player_id],
            num_trade_routes=games.num_trade_routes[games_index, player_id],
            cs_perturn_influence=games.cs_perturn_influence[games_index, player_id],
            num_delegates=_num_delegates,
            culture_threshold=games.culture_threshold[games_index, player_id],
            religious_tenets=games.religious_tenets,
            free_techs=games.free_techs[games_index, player_id],
            free_policies=games.free_policies[games_index, player_id],
            great_works=games.great_works[games_index, player_id],
            culture_info=culture_info,
            improvement_additional_yield_map=_self.improvement_additional_yield_map[games_index, player_id],
            improvement_map=_self.improvement_map[games_index, player_id],
            road_map=_self.road_map[games_index, player_id],
            gpps=games.gpps[games_index, player_id],
            gp_threshold=games.gp_threshold[games_index, player_id],
            golden_age_turns=games.golden_age_turns[games_index, player_id],
            tourism_total=_tourism_total,
            culture_total=_culture_total,
            citystate_info=citystate_info,
            visibility_map=games.visibility_map[games_index, player_id],
            trade_offers=games.trade_offers[games_index[:, None], jnp.arange(6), player_id[:, None]],  # [i,j] ith player offer to jth player
            trade_ledger=games.trade_ledger[games_index, player_id],
            trade_length_ledger=games.trade_length_ledger[games_index, player_id],
            trade_gpt_adjustment=games.trade_gpt_adjustment[games_index, player_id],
            trade_resource_adjustment=games.trade_resource_adjustment[games_index, player_id],
            have_met=games.have_met[games_index, player_id],
            at_war=_at_war,
            has_sacked=_has_sacked,
            treasury=games.treasury[games_index, player_id],
            happiness=games.happiness[games_index, player_id],

            # Derived info (per-game int32 zeros; optionally sharded like `sharding_ref`)
            most_population=games.player_cities.population.sum(-1).argmax(-1),
            least_population=games.player_cities.population.sum(-1).argmin(-1),
            most_crop_yield=games.player_cities.yields[:, :, :, FOOD_IDX].sum(-1).argmax(-1),
            least_crop_yield=games.player_cities.yields[:, :, :, FOOD_IDX].sum(-1).argmin(-1),
            most_manufactured_goods=games.player_cities.yields[:, :, :, PROD_IDX].sum(-1).argmax(-1),
            least_manufactured_goods=games.player_cities.yields[:, :, :, PROD_IDX].sum(-1).argmin(-1),
            most_gnp=games.player_cities.yields[:, :, :, GOLD_IDX].sum(-1).argmax(-1),
            least_gnp=games.player_cities.yields[:, :, :, GOLD_IDX].sum(-1).argmin(-1),
            most_land=games.player_cities.ownership_map.sum((-1, -2, -3)).argmax(-1),
            least_land=games.player_cities.ownership_map.sum((-1, -2, -3)).argmin(-1),
            most_soldiers=((ALL_UNIT_COMBAT[games.units.unit_type - 1]) * (games.units.unit_type > 0)).sum(-1).argmax(-1),
            least_soldiers=((ALL_UNIT_COMBAT[games.units.unit_type - 1]) * (games.units.unit_type > 0)).sum(-1).argmin(-1),
            most_approval=games.happiness.argmax(-1),
            least_approval=games.happiness.argmin(-1),
            most_literacy=games.technologies.sum(-1).argmax(-1),
            least_literacy=games.technologies.sum(-1).argmin(-1),
            player_id=player_id,
            current_turn=games.current_step,
        )
        return _self, full_obs 
