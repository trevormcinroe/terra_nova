from functools import partial
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map as shmap

from typing import List, Tuple, Callable
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from game.primitives import GameState
from game.religion import ReligiousTenets
from game.social_policies import SocialPolicies
from game.resources import ALL_RESOURCES
from game.units import Units 
from learning.goals import compute_rewards
from learning.obs_spaces import TerraNovaObservationSpaceTracker, ObservationSpace 
from game.termination_fns import reset_episode as termination_fn
from game.primitives import Cities, CityStateInfo, ResetGameState, CultureInfo
from game.techs import Technologies
from game.constants import MAX_NUM_CITIES, MAX_NUM_UNITS, MAX_TRADE_DEALS
from learning.metrics import TerraNovaEpisodeMetrics

def build_simulator(
        loaded_maps: List[GameState],
        distributed_strategy: str,
        key: jnp.ndarray,
    ):
    assert distributed_strategy in ["split", "duplicate"], f"Do not currently support provided distributed_strategy={distributed_strategy}"
    
    # First some hardware bookkeeping.
    LOCAL_DEVICE_COUNT = jax.local_device_count()
    all_devices = mesh_utils.create_device_mesh((LOCAL_DEVICE_COUNT,))
    GLOBAL_MESH = Mesh(all_devices, axis_names=("gpus",))
    sharding = jax.sharding.NamedSharding(GLOBAL_MESH, P("gpus",))
    
    print(f"Found {LOCAL_DEVICE_COUNT} XLA device(s).")

    if distributed_strategy == "split":
        _n_games = len(loaded_maps)
        if _n_games % LOCAL_DEVICE_COUNT != 0:
            raise ValueError(
                f"The number of maps ({_n_games}) cannot be split equally across the number of visible XLA devices ({LOCAL_DEVICE_COUNT})"
            )


    # Step 0: taking the raw map data and adding all of the relevant things!
    _loaded_maps = []
    
    for gamestate in tqdm(loaded_maps, desc="Initializing gamestates..."):
        # First, we need to take the numpy-fied object and convert it into a GameState of jax arrays
        state_jax = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            gamestate,
        )

        # 3. Reconstruct the GameState directly from the dict
        gamestate = GameState(**state_jax)
        gamestate = gamestate.replace(units=Units(**gamestate.units))

        # For now, let's create the citieson the fly
        cs_cities = Cities.create(num_players=12, max_num_cities=1, game=gamestate)
        cs_cities = cs_cities.replace(city_rowcols=gamestate.cs_cities[:, None])
        citystate_info = CityStateInfo.create(key)

        max_num_cities = MAX_NUM_CITIES
        player_cities = Cities.create(num_players=6, max_num_cities=max_num_cities, game=gamestate)
        gamestate = gamestate.replace(cs_cities=cs_cities, player_cities=player_cities)
        
        # Init techs for each player. Everyone starts with tech 0 (agriculture). This is to ensure that 
        # all base resources are visible. Should have no unintended side-effects (I hope)
        techs = jnp.zeros(shape=(6, len(Technologies)), dtype=jnp.uint8)
        is_researching = jnp.zeros(shape=(6,), dtype=jnp.int32) - 1
        
        techs = techs.at[jnp.arange(6), 0].set(1)
        policies = jnp.zeros(shape=(6, len(SocialPolicies)), dtype=jnp.uint8)
        science_reserves = jnp.zeros(shape=(6,), dtype=jnp.float32)
        culture_reserves = jnp.zeros(shape=(6,), dtype=jnp.float32)
        faith_reserves = jnp.zeros(shape=(6,), dtype=jnp.float32)
        num_trade_routes = jnp.zeros(shape=(6,), dtype=jnp.int32) + 2
        cs_resting_influence = jnp.zeros(shape=(6), dtype=jnp.int32)
        cs_perturn_influence = jnp.zeros(shape=(6, 12), dtype=jnp.int32)

        # All trade route arrays are (player, from city, to city)
        cs_trade_routes = jnp.zeros(shape=(6, max_num_cities, 12), dtype=jnp.uint8)
        player_trade_routes = jnp.zeros(shape=(6, max_num_cities, max_num_cities), dtype=jnp.uint8)
        trade_route_yields = jnp.zeros(shape=(6, max_num_cities, 8), dtype=jnp.float32)
        num_delegates = jnp.zeros(shape=(6,), dtype=jnp.uint8)
        spent_great_prophet = jnp.zeros(shape=(6,), dtype=jnp.uint8)
        culture_threshold = jnp.zeros(shape=(6,), dtype=jnp.float32)
        religious_tenets = jnp.zeros(shape=(6, len(ReligiousTenets)), dtype=jnp.uint8)
        free_techs = jnp.zeros(shape=(6,), dtype=jnp.uint8)
        
        gamestate = gamestate.replace(
            technologies=techs,
            policies=policies,
            science_reserves=science_reserves,
            culture_reserves=culture_reserves,
            faith_reserves=faith_reserves,
            is_researching=is_researching,
            research_started=is_researching,
            research_finished=is_researching,
            num_trade_routes=num_trade_routes,
            cs_resting_influence=cs_resting_influence,
            cs_perturn_influence=cs_perturn_influence,
            cs_trade_routes=cs_trade_routes,
            player_trade_routes=player_trade_routes,
            trade_route_yields=trade_route_yields,
            num_delegates=num_delegates,
            culture_threshold=culture_threshold,
            religious_tenets=religious_tenets,
            spent_great_prophet=spent_great_prophet,
            visible_resources_map_players=jnp.concatenate([
                gamestate.all_resource_map[None] for _ in range(6)
            ], axis=0),
            yield_map_players=jnp.concatenate([
                gamestate.yield_map[None] for _ in range(6)
            ], axis=0),
            free_techs=free_techs,
            free_tech_from_oxford=deepcopy(free_techs),
            free_tech_from_great_lib=deepcopy(free_techs),
            free_workers_from_pyramids=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            tile_improvement_speed_from_pyramids=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            tile_improvement_speed=jnp.ones(shape=(6,), dtype=jnp.float32),
            free_cargo_ship_from_colossus=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_trade_route_from_colossus=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_trade_route_from_nattreas=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_policies=deepcopy(free_techs),
            free_policy_from_oracle=deepcopy(free_techs),
            golden_age_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            golden_age_accel_from_chichen=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            combat_friendly_terr_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            combat_friendly_terr_accel_from_himeji=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            culture_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            culture_accel_from_sistine=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            delegates_from_forbidden=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            gold_purchase_mod=jnp.ones(shape=(6,), dtype=jnp.float32),
            gold_purchase_mod_from_ben=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_policy_from_statue=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_pop_from_statue=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            culture_threshold_mod=jnp.ones(shape=(6,), dtype=jnp.float32),
            culture_threshold_mod_from_cristo=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            unit_upgrade_cost_mod=jnp.ones(shape=(6,), dtype=jnp.float32),
            unit_upgrade_cost_mod_from_pentagon=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_policy_from_sydney=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            great_works=jnp.zeros(shape=(6, 4), dtype=jnp.uint8),
            attacking_cities_add=jnp.zeros(shape=(6,), dtype=jnp.int32),
            attacking_cities_add_from_zeus=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            gold_per_gp_expend=jnp.zeros(shape=(6,), dtype=jnp.int32),
            gold_per_gp_expend_from_maso=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_pop_from_cn=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            global_great_person_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            global_great_person_accel_from_lt=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            missionary_spreads_from_djenne=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_caravan_from_petra=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_trade_route_from_petra=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            religious_pressure_from_gt=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_prophet_from_hagia=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_artist_from_uffizi=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_writer_from_globe=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_musician_from_broadway=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            defense_accel_from_red_fort=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            global_defense_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            free_missionaries_from_boro=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_merchant_from_panama=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            culture_info=CultureInfo.create(6, max_num_cities, None),
            free_settler_from_collective_rule=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_worker_from_citizenship=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            tile_improvement_speed_from_citizenship=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            golden_age_from_representation=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_warriors_from_wc=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            reformation_belief_from_ref=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            delegates_from_consulates=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_writer_from_ethics=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_artist_from_art_genius=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            golden_age_from_flourishing=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            trade_routes_from_ent=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_scientist_from_sci_rev=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            tradition_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            liberty_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            honor_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            piety_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            patronage_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            aesthetics_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            commerce_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            exploration_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            rationalism_finished=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            growth_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            nat_wonder_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            science_per_kill=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            happiness_per_unique_lux=jnp.zeros(shape=(6,), dtype=jnp.int32) + 4,
            science_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            prophet_threshold_accel=jnp.ones(shape=(6,), dtype=jnp.float32),
            prophet_threshold_from_messiah=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            trade_route_from_troub=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_prophet_from_cog=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            improvement_additional_yield_map=jnp.zeros(shape=(42, 66, 7), dtype=jnp.float32),
            improvement_map=jnp.zeros(shape=(42, 66), dtype=jnp.int32),
            road_map=jnp.zeros(shape=(42, 66), dtype=jnp.uint8),
            gpps=jnp.zeros(shape=(6, 6), dtype=jnp.int32),
            gp_threshold=jnp.zeros(shape=(6,), dtype=jnp.int32) + 67,
            in_golden_age=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            golden_age_turns=jnp.zeros(shape=(6,), dtype=jnp.int32),
            tourism_total=jnp.zeros(shape=(6, 6), dtype=jnp.float32),
            culture_total=jnp.zeros(shape=(6,), dtype=jnp.float32),
            tourism_this_turn=jnp.zeros(shape=(6,), dtype=jnp.float32),
            citystate_info=citystate_info,
            visibility_map=jnp.zeros(shape=(6, 42, 66), dtype=jnp.uint8) + 2,
            trade_offers=jnp.zeros(shape=(6, 6, 2), dtype=jnp.uint8),
            trade_ledger=jnp.zeros(shape=(6, 6, MAX_TRADE_DEALS, 2), dtype=jnp.uint8),
            trade_length_ledger=jnp.zeros(shape=(6, MAX_TRADE_DEALS), dtype=jnp.uint8),
            trade_gpt_adjustment=jnp.zeros(shape=(6,), dtype=jnp.int8),
            trade_resource_adjustment=jnp.zeros(shape=(6, len(ALL_RESOURCES)), dtype=jnp.int8),
            have_met=jnp.zeros(shape=(6, 6 + 12), dtype=jnp.bool_),
            at_war=jnp.zeros(shape=(6, 6), dtype=jnp.bool_),
            has_sacked=jnp.zeros(shape=(6, 6), dtype=jnp.bool_),
            treasury=jnp.zeros(shape=(6,), dtype=jnp.float32),
            happiness=jnp.zeros(shape=(6,), dtype=jnp.float32) + 8,
            golden_age_from_taj=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            free_great_artist_from_louvre=jnp.zeros(shape=(6,), dtype=jnp.uint8),
            aesthetics_finisher_bonus=jnp.ones(shape=(6,), dtype=jnp.float32),
            commerce_finisher_bonus=jnp.ones(shape=(6,), dtype=jnp.float32),
            is_connected_to_cap=jnp.zeros(shape=(6, max_num_cities), dtype=jnp.uint8),
        )
        
        # After the game has been instantiated, we need to remove several resources from each players'
        # visibility, as these resources require certain techs to see. E.g., iron, horses, uranium
        for i in range(6):
            gamestate = gamestate.update_player_visible_resources_and_yields(jnp.array([i]))
        
        gamestate = gamestate.create_improvement_bitfield_mask()

        trimmed_units = jax.tree.map(lambda x: x[:, :MAX_NUM_UNITS] if len(x.shape) > 1 else x, gamestate.units)
        gamestate = gamestate.replace(units=trimmed_units)
        initial_state_cache = ResetGameState().replace(**{name: deepcopy(getattr(gamestate, name)) for name in ResetGameState.__dataclass_fields__})
        gamestate = gamestate.replace(initial_state_cache=initial_state_cache)
        _loaded_maps.append(gamestate)
    
    loaded_maps = _loaded_maps

    n_games = len(loaded_maps)
    print(f"Found {n_games} games.")
    
    if distributed_strategy == "split":
        # For the distributed games, 
        b_idx = 0
        n_games_per_device = n_games // LOCAL_DEVICE_COUNT
        e_idx = b_idx + n_games_per_device

        # IIUC, the only difference mechnaically between duplicating and splitting takes place with the gamestates
        # themselves. Every other component can be duplicated across the mesh in the same way as "duplicate" branch.
        # Here we duplicate the set of maps across each device. HOWEVER, we first need to split the keys
        # for each map, otherwise the randomness would be duplicated across the GPUs as well.
        updated_gamestates = []

        # To ensure that each gamestate, regardless of XLA device, has a unique random key, let's generate
        # all of them here!
        all_keys = jax.random.split(key, LOCAL_DEVICE_COUNT * len(loaded_maps))
        all_keys = all_keys.reshape(LOCAL_DEVICE_COUNT, len(loaded_maps), 2)


        for game in loaded_maps:
            key, _ = jax.random.split(key, 2)
            game = game.replace(key=key)
            updated_gamestates.append(deepcopy(game))

        # Now we need to take the lists of arrays and combine them into single arrays with an expanded
        # leading dimension.
        # One way to do this is to first expand the leading dim, then concat along that dim
        updated_gamestates = [jax.tree.map(lambda x: x[None], game) for game in updated_gamestates]
        episode_metrics = TerraNovaEpisodeMetrics.create(n_games_per_device, num_episodes_to_track=10)

        # Now we can loop through each of the xla devices and place them on there
        # The replays are going one-per-device
        distributed_games = []
        distributed_obs_spaces = []
        distributed_metrics = []
        players_turn_id = []
        distributed_key_helper = []
        
        print("Distributing arrays...")
        for i, mesh_device in enumerate(GLOBAL_MESH.devices):
            games = jax.tree.map(lambda *arr: jnp.concatenate(arr, axis=0), *updated_gamestates[b_idx: e_idx])
            
            b_idx += n_games_per_device
            e_idx += n_games_per_device
            
            game_bundle = jax.tree.map(lambda x: jax.device_put(deepcopy(x[None]), mesh_device), games)
            placed_metrics = jax.tree.map(lambda x: jax.device_put(deepcopy(x[None]), mesh_device), episode_metrics)
            
            distributed_games.append(game_bundle)
            distributed_metrics.append(placed_metrics)

            players_turn_id.append(jax.device_put(jnp.zeros(shape=(1, n_games_per_device), dtype=jnp.int32), mesh_device))
            distributed_key_helper.append(jax.device_put(all_keys[i, :n_games_per_device][None], mesh_device))

        distributed_games = jax.tree.map(lambda *x: list(x), *distributed_games)
        distributed_games = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                (LOCAL_DEVICE_COUNT, *x[0].shape[1:]),
                sharding, 
                x
            ),
            distributed_games,
            is_leaf=lambda x: isinstance(x, list)
        )
        distributed_keys = jax.make_array_from_single_device_arrays(
            (LOCAL_DEVICE_COUNT, n_games_per_device, 2),
            sharding,
            distributed_key_helper
        )
        distributed_games = distributed_games.replace(key=distributed_keys)

        # The observation space tracker
        distributed_obs_spaces = TerraNovaObservationSpaceTracker.create(n_games_per_device, distributed_games)

        distributed_metrics = jax.tree.map(lambda *x: list(x), *distributed_metrics)
        distributed_metrics = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                (LOCAL_DEVICE_COUNT, *x[0].shape[1:]),
                sharding, 
                x
            ),
            distributed_metrics,
            is_leaf=lambda x: isinstance(x, list)
        )

        distributed_players_turn_id = jax.make_array_from_single_device_arrays((LOCAL_DEVICE_COUNT, n_games_per_device), sharding, players_turn_id)

    elif distributed_strategy == "duplicate":
        # Here we duplicate the set of maps across each device. HOWEVER, we first need to split the keys
        # for each map, otherwise the randomness would be duplicated across the GPUs as well.
        updated_gamestates = []

        # To ensure that each gamestate, regardless of XLA device, has a unique random key, let's generate
        # all of them here!
        all_keys = jax.random.split(key, LOCAL_DEVICE_COUNT * len(loaded_maps))
        all_keys = all_keys.reshape(LOCAL_DEVICE_COUNT, len(loaded_maps), 2)


        for game in loaded_maps:
            key, _ = jax.random.split(key, 2)
            game = game.replace(key=key)
            updated_gamestates.append(deepcopy(game))

        # Now we need to take the lists of arrays and combine them into single arrays with an expanded
        # leading dimension.
        # One way to do this is to first expand the leading dim, then concat along that dim
        updated_gamestates = [jax.tree.map(lambda x: x[None], game) for game in updated_gamestates]
        games = jax.tree.map(lambda *arr: jnp.concatenate(arr, axis=0), *updated_gamestates)
        
        episode_metrics = TerraNovaEpisodeMetrics.create(n_games, num_episodes_to_track=10)

        # Now we can loop through each of the xla devices and place them on there
        # The replays are going one-per-device
        distributed_games = []
        distributed_obs_spaces = []
        distributed_metrics = []
        players_turn_id = []
        distributed_key_helper = []

        print("Distributing arrays...")
        for i, mesh_device in enumerate(GLOBAL_MESH.devices):
            new_key = jax.vmap(jax.random.split, in_axes=(0))(games.key)[:, :, 0]
            games = games.replace(key=new_key)

            game_bundle = jax.tree.map(lambda x: jax.device_put(deepcopy(x[None]), mesh_device), games)
            placed_metrics = jax.tree.map(lambda x: jax.device_put(deepcopy(x[None]), mesh_device), episode_metrics)
            
            distributed_games.append(game_bundle)
            distributed_metrics.append(placed_metrics)

            players_turn_id.append(jax.device_put(jnp.zeros(shape=(1, n_games), dtype=jnp.int32), mesh_device))
            distributed_key_helper.append(jax.device_put(all_keys[i][None], mesh_device))
        
        distributed_games = jax.tree.map(lambda *x: list(x), *distributed_games)
        distributed_games = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                (LOCAL_DEVICE_COUNT, *x[0].shape[1:]),
                sharding, 
                x
            ),
            distributed_games,
            is_leaf=lambda x: isinstance(x, list)
        )
        distributed_keys = jax.make_array_from_single_device_arrays(
            (LOCAL_DEVICE_COUNT, n_games, 2),
            sharding,
            distributed_key_helper
        )
        distributed_games = distributed_games.replace(key=distributed_keys)

        # The observation space tracker
        distributed_obs_spaces = TerraNovaObservationSpaceTracker.create(n_games, distributed_games)

        distributed_metrics = jax.tree.map(lambda *x: list(x), *distributed_metrics)
        distributed_metrics = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                (LOCAL_DEVICE_COUNT, *x[0].shape[1:]),
                sharding, 
                x
            ),
            distributed_metrics,
            is_leaf=lambda x: isinstance(x, list)
        )

        distributed_players_turn_id = jax.make_array_from_single_device_arrays((LOCAL_DEVICE_COUNT, n_games), sharding, players_turn_id)
        
    @partial(
        shmap, 
        mesh=GLOBAL_MESH, 
        in_specs=(P("gpus"), P("gpus"), P("gpus")),
        out_specs=(P("gpus"), P("gpus"), P("gpus")),
    )
    def _reset_at_start(_games, _obs_spaces, _distributed_players_turn):
        """
        Need to run the fog of war computation before the game starts
        """
        _games = jax.tree.map(lambda x: x[0], _games)
        _obs_spaces = jax.tree.map(lambda x: x[0], _obs_spaces)
        _distributed_players_turn = jax.tree.map(lambda x: x[0], _distributed_players_turn)

        _games = _games.compute_fog_of_war()
        _obs_spaces, _obs_for_reset = _obs_spaces._update_and_grab_obs(_games, _distributed_players_turn)

        _games = jax.tree.map(lambda x: x[None], _games)
        _obs_spaces = jax.tree.map(lambda x: x[None], _obs_spaces)
        _obs_for_reset = jax.tree.map(lambda x: x[None], _obs_for_reset) 
        return _games, _obs_spaces, _obs_for_reset
    
    print("Initializing observation space...")
    distributed_games, distributed_obs_spaces, distributed_obs_for_reset = _reset_at_start(
        distributed_games, distributed_obs_spaces, distributed_players_turn_id
    )
    
    @jax.jit
    def reset_episode(inp: Tuple[GameState, TerraNovaEpisodeMetrics, jnp.ndarray, TerraNovaObservationSpaceTracker]):
        games, episode_metrics, player_id, obs_space = inp
        
        # Need to track what happened this episode before we reset the entire gamestate, obv
        episode_metrics = episode_metrics.track_end_of_episode(games)
        
        stepped_keys, _ = jax.random.split(games.key)

        games = games.replace(
            **{name: getattr(games.initial_state_cache, name) for name in games.initial_state_cache.__dataclass_fields__}
        )
        games = games.replace(key=stepped_keys)
        
        episode_metrics = episode_metrics.step_episode()
        return games, episode_metrics, 0, jnp.array([True]), obs_space.reset()

    @partial(jax.vmap, in_axes=(0, None, 0, 0))
    def maybe_reset(games: GameState, obs_space: ObservationSpace, episode_metrics: TerraNovaEpisodeMetrics, player_id):
        games, episode_metrics, player_id, done_flag, obs_space = jax.lax.cond(
            termination_fn(games, obs_space, episode_metrics, player_id)[0],
            reset_episode,
            lambda x: (x[0], x[1], (x[2] + 1) % 6, jnp.array([False]), x[3]),
            operand=(games, episode_metrics, player_id, obs_space)
        )
        return games, episode_metrics, player_id, done_flag

    @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
    def step_games_for_agent(games, logits, obs_space, player_id, valid_move_map):
        """
        Things that happen between turns (e.g., techs finishing), should be done first, as many of 
        these processes effect what can be built, upgraded, etc
        (1) Research
        (2) Units
        (3) Cities
        (4) Social
        (5) Religion
        (6) Diplo

        """
        # For backwards compat (more like for my own sanity), the player_id value needs to go from just an integer to shape (1,)
        player_id = player_id[None]
        
        trade_deal_logits, logits_sp, logits_rel, logits_tech, unit_logits, city_logits = logits

        ### Stepping the game mechanics ###
        # For all "selected_*" actions that we pass back to the user, -1 indicates
        # that _no_ action was taken (e.g., no action of that type could be 
        # taken on the current turn for player_id)
        # Some of the action types (e.g., religion) have a slightly "unusual" response.
        # One of the relgion triggers is to found a religion - this involves selecting three
        # religious tenets on the same turn. So all types need to return vector (3,)
        games, executed_trade_actions = games.step_trade_between_players(trade_deal_logits, player_id) 
        games, executed_religion_actions = games.step_religion(logits_rel, player_id)
        games, executed_policy_action = games.step_policies(logits_sp, player_id)
        games, executed_tech_action = games.step_technology(logits_tech, player_id)
        #
        games = games.update_player_visible_resources_and_yields(player_id)


        ## Units may be built in .step_cities(), so we need to compute the valid moves map at this specific point
        valid_move_map = games.get_valid_moves_pergame_perspectivev2(player_id)

        games, executed_unit_actions = games.step_unitsv2(unit_logits, obs_space, player_id, valid_move_map)
        games, executed_city_actions = games.step_citiesv2((city_logits[0], city_logits[1]), obs_space, player_id)
        games = games.step_tourism(player_id)
        games = games.step_specialists_great_people_and_golden_age(player_id)
        games = games.step_empire(player_id)
        
        amt_to_add = player_id[0] == 5
        games = games.replace(current_step=games.current_step + amt_to_add)

        ### Exectured actions to be returned ###
        # trade: (6,), (), (), ()
        # policy: (2,) <- (non-free, free)
        # religion: (3,) <- (only three non -1 when founding)
        # technology: (2,) <- (non-free, free)
        # units: (30,), (30,) <- (categories), (map) 
        # cities: (10, 36), (10,) <- (pop placement),  (constructing)
        selected_actions = (
            executed_trade_actions, 
            executed_policy_action, 
            executed_religion_actions, 
            executed_tech_action,
            executed_unit_actions,
            executed_city_actions,
        )
        return games, selected_actions

    @partial(jax.vmap, in_axes=(0))
    def run_citystates(inps):
        """
        We are here in a vmap-over-games context

        (1) Grow border every N turns (can equivalently be accomplished with pct chance)
        (2) Compute influence change
        (3) Compute religion change
        (4) Change status with all players & give out bonuses
        """
        games, player_ids = inps
        games = games.step_citystates()
        return (games, player_ids)

    @jax.jit
    def run_simulation_once(games, logits, obs_space, episode_metrics, player_ids):
        games, selected_actions = step_games_for_agent(games, logits, obs_space, player_ids, None)
        games = games.compute_fog_of_war()
        
        # Now we can _potentially_ reset the episodes
        games, episode_metrics, player_ids, done_flags = maybe_reset(games, obs_space, episode_metrics, player_ids)

        (games, player_ids) = jax.lax.cond(
            jnp.any(player_ids == 5),
            run_citystates,
            lambda x: x,
            operand=(games, player_ids)
        )

        new_key = jax.vmap(jax.random.split, in_axes=(0))(games.key)[:, :, 0]
        games = games.replace(key=new_key)
        return games, obs_space, episode_metrics, player_ids, done_flags, selected_actions

    @partial(jax.jit)
    @partial(
        shmap,
        mesh=GLOBAL_MESH,
        in_specs=(P("gpus"), P("gpus"), P("gpus"), P("gpus"), P("gpus")),
        out_specs=(P("gpus"), P("gpus"), P("gpus"), P("gpus"), P("gpus"), P("gpus"), P("gpus"), P("gpus")),
        check_rep=False 
    )
    def env_step(games, logits, obs_space, episode_metrics, player_ids):
        games = jax.tree.map(lambda x: x[0], games)
        logits = jax.tree.map(lambda x: x[0], logits)
        obs_space = jax.tree.map(lambda x: x[0], obs_space)
        episode_metrics = jax.tree.map(lambda x: x[0], episode_metrics)
        player_ids = player_ids[0]

        games, obs_space, episode_metrics, new_player_ids, done_flags, selected_actions = run_simulation_once(
            games, logits, obs_space, episode_metrics, player_ids
        )

        # After the environment has been stepped for a given player, we then update the observation 
        # space tracker and return the observation for the next player. This should work well with the
        # episode-reset flow, as this is done at the end of `run_simulation_once()`. The reset function
        # inside there should call obs_space.reset() in the event of an episode ending. 
        obs_space, next_obs = obs_space._update_and_grab_obs(games, player_ids)

        # We need to ensure that we return the rewards to the player_id that just executed actions 
        # in the environments, _not_ the incremented player_id
        player_rewards, episode_metrics = compute_rewards(games, player_ids, episode_metrics)
        games = jax.tree.map(lambda x: x[None], games)
        obs_space = jax.tree.map(lambda x: x[None], obs_space)
        episode_metrics = jax.tree.map(lambda x: x[None], episode_metrics)
        player_ids = new_player_ids[None]  # ensure we increment
        next_obs = jax.tree.map(lambda x: x[None], next_obs)
        player_rewards = player_rewards[None]
        done_flags = done_flags[None]
        selected_actions = jax.tree.map(lambda x: x[None], selected_actions)

        return games, obs_space, episode_metrics, player_ids, next_obs, player_rewards, done_flags, selected_actions

    return env_step, distributed_games, distributed_obs_spaces, distributed_metrics, distributed_players_turn_id,  distributed_obs_for_reset, GLOBAL_MESH, sharding
