from dataclasses import asdict
from flax.struct import dataclass, field
from jax._src.lax.slicing import index_take
import jax.numpy as jnp
import jax
import numpy as np
from typing import Tuple, Union
import json
import os
import tempfile
import gzip
import io
import shutil

from game.primitives import GameState
from game.religion import ReligiousTenets
from utils.maths import generate_6d_border_vector_from_ownership_matrix, generate_6d_border_vector_from_ownership_matrix_jit


def atomic_write_ndjson_gzip(iter_objects, path, level=6):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_ndjson_", suffix=".gz")
    os.close(fd)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=level, newline="\n") as gz:
            for obj in iter_objects:  # yield dicts for each turn/section
                gz.write(json.dumps(obj, separators=(",", ":")))
                gz.write("\n")
            gz.flush()
        os.replace(tmp, path)
    except Exception:
        try: 
            os.remove(tmp)
        except OSError: 
            pass
        raise


def iter_gamestate_lines(gamestate: dict):
    """
    Stream a single, huge `gamestate` dict as NDJSON lines without changing how the
    recorder builds that dict. We auto-detect "step-major" keys (lists whose length
    equals num_steps) and emit:
      1) a small meta line (num_steps + key lists),
      2) one line per turn with ONLY the step-major slices for that turn,
      3) one line per constant key for everything else.

    Usage (drop-in for your current `save_to_json` tail):
        atomic_write_ndjson_gzip(iter_gamestate_lines(gamestate),
                                 filename + ".ndjson.gz")
    """
    # --- detect num_steps from any list-like top-level value ---
    def _first_seq_len(d):
        for v in d.values():
            if isinstance(v, list):
                return len(v)
        raise ValueError("iter_gamestate_lines: could not infer num_steps")

    num_steps = _first_seq_len(gamestate)

    # A key is considered step-major if its top-level value is a list of length == num_steps
    step_keys = [k for k, v in gamestate.items() if isinstance(v, list) and len(v) == num_steps]
    const_keys = [k for k in gamestate.keys() if k not in step_keys]

    # --- 1) meta line ---
    yield {
        "meta": {
            "num_steps": num_steps,
            "step_keys": step_keys,
            "const_keys": const_keys,
        }
    }

    # --- 2) per-turn lines (only include the step-major slices for that turn) ---
    # This keeps each line relatively small and lets the browser stream-parse.
    for t in range(num_steps):
        line = {"turn": t}
        for k in step_keys:
            # Each step-major value is expected to be indexable at [t]
            line[k] = gamestate[k][t]
        yield line

    # --- 3) constants (one line per key) ---
    for k in const_keys:
        yield {"const": k, "value": gamestate[k]}


@dataclass
class GameStateRecorder(GameState):
    recorder_step: jnp.ndarray = field(pytree_node=True, default=None)
    @classmethod
    def create(cls, reference_gamestate, num_steps):
        return cls(
            **asdict(jax.tree.map(lambda x: jnp.zeros(shape=(num_steps, *x.shape)), reference_gamestate)),
            recorder_step=jnp.zeros(shape=(1,), dtype=jnp.int32),
        )

    def record(self, gamestate_snapshot):
        #print(f"=================== CURRENT STEP: {gamestate_snapshot.current_step} vs {self.recorder_step} =======================")
        fields_to_update = {k: v for k, v in asdict(self).items()}
        del fields_to_update["recorder_step"]
        updated_fields = jax.tree.map(
            lambda x, y: x.at[self.recorder_step[0].astype(jnp.int32)].set(y),
            fields_to_update,
            asdict(gamestate_snapshot),
        )

        return self.replace(**updated_fields, recorder_step=self.recorder_step.at[0].add(1))

    def save_replay(self, filename):
        """
        Note on yields:
            * player_yields is on a per-player, per-city basis
            * *yield_map are the tiles on the map

        * improvement_map will give us the improvement icons
        * road_map gives us the location of roads        
        """
        print("Beginning data aggregation for save...")

        terrain_np = np.array(self.terrain_map)
        terrain_lol = [inner_l.tolist() for inner_l in [l for l in terrain_np]]

        rivers_np = np.array(self.edge_river_map)
        rivers_lolol = [inner_l.tolist() for inner_l in [l for l in [x for x in rivers_np]]]

        lakes_np = np.array(self.lake_map)
        lakes_lol = [l.tolist() for l in [x for x in lakes_np]]

        elevation_np = np.array(self.elevation_map)
        elevation_lol = [l.tolist() for l in [x for x in elevation_np]]

        features_np = np.array(self.feature_map)
        features_lol = [l.tolist() for l in [x for x in features_np]]

        nw_np = np.array(self.nw_map)
        nw_lol = [l.tolist() for l in [x for x in nw_np]]

        all_resource_map_np = np.array(self.all_resource_map)
        all_resource_map_lol = [l.tolist() for l in [x for x in all_resource_map_np]]

        yield_map = np.array(self.yield_map)
        yield_map_lolol = [inner_l.tolist() for inner_l in [l for l in [x for x in yield_map]]]
        
        # both are (6, 42, 66, 7)
        # additional_yield_map is things effected by cities, religion, etc
        # yield_map_players is based on what the player can see based on tech!

        additional_yield_map = np.array(self.player_cities["additional_yield_map"])
        additional_yield_map_religion = np.array(self.player_cities["religion_info"]["additional_yield_map"]).sum(2)
        additional_yield_map_improvements = np.array(self.improvement_additional_yield_map)

        additional_yield_map_culture = np.array(self.culture_info["additional_yield_map"])

        players_yield_map = np.array(self.yield_map_players)
        player_view_yield_map = additional_yield_map + players_yield_map + additional_yield_map_religion + additional_yield_map_improvements[:, None] + additional_yield_map_culture
        player_yield_map_lololol = [q.tolist() for q in [inner_l for inner_l in [l for l in [x for x in player_view_yield_map]]]]

        cs_ownership_np = np.array(self.cs_ownership_map)
        cs_ownership_lol = [l.tolist() for l in [x for x in cs_ownership_np]]
        
        cs_rowcols = np.array(self.cs_cities["city_rowcols"])[:, :, 0, :]
        cs_rowcols = [l.tolist() for l in [x for x in cs_rowcols]]

        # We only need this for viz, so not-jitable is not a blocker
        cs_ownership_borders_list = []
        for cs_ownership_map in self.cs_ownership_map:
            cs_ownership_borders = generate_6d_border_vector_from_ownership_matrix_jit(cs_ownership_map.astype(jnp.int32))
            cs_ownership_borders_list.append(cs_ownership_borders)
        
        cs_ownership_borders_np = np.array(cs_ownership_borders_list)
        cs_ownership_borders_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in cs_ownership_borders_np]]]
        
        # (steps, 6, max_num_cities, 2) => (steps, num_players, 2)
        player_rowcols = []
        # (steps, num_players, 42, 66, 6)
        player_ownership_borders_list = []
        player_ownership = []

        # Loop over steps
        for i in range(self.player_cities["city_rowcols"].shape[0]):
            step_player_rowcols = []
            is_there_city_for_player = []
            rowcol = self.player_cities["city_rowcols"][i]
            ownership = self.player_cities["ownership_map"][i].max(1)  # max over 1 will collapse all city ownership to (6, 42, 66)
            
            # The ownership_map is (42, 66) where the numbers are either 0 (no one ones) or a number
            # relating each player (1-6)
            dummy_map = jnp.zeros(shape=(ownership.shape[1], ownership.shape[2]))
            for ii in range(6):
                player_i_ownership = ownership[ii]
                this_player = (player_i_ownership == 2) | (player_i_ownership == 3)
                dummy_map += (this_player * (ii + 1))
            
            player_ownership.append(dummy_map)

            # Loop through each player
            for j in range(rowcol.shape[0]):
                _inner = []
                is_there_city_for_player.append(
                    self.player_cities["city_ids"][i, j].sum() > 0
                )
                # Loop through each city
                for k in range(rowcol.shape[1]):
                    if self.player_cities["city_ids"][i, j, k] == 0:
                        _inner.append([-1, -1])
                        continue
                    _rc = rowcol[j, k]

                    _inner.append(_rc)
                step_player_rowcols.append(np.array(_inner))
            
            # needs (42, 66)
            player_ownership_borders = generate_6d_border_vector_from_ownership_matrix_jit(dummy_map.astype(jnp.int32))
            shpe = (6 - player_ownership_borders.shape[0], 42, 66, 6)
            player_ownership_borders = jnp.concatenate(
                [player_ownership_borders, jnp.zeros(shpe) - 1],
                axis=0
            )
            player_rowcols.append(np.array(step_player_rowcols))
            player_ownership_borders_list.append(player_ownership_borders)
        
        player_ownership_np = np.array(player_ownership)
        player_ownership_lol = [l.tolist() for l in [x for x in player_ownership_np]]

        player_ownership_borders_np = np.array(player_ownership_borders_list)
        player_rowcols_np = np.array(player_rowcols)
        
        player_ownership_borders_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in player_ownership_borders_np]]]

        player_rowcols = [inner_l.tolist() for inner_l in [l for l in [x for x in player_rowcols_np]]]
        
        # worked_slots (steps, players, cities, 36)
        worked_slots_np = np.array(self.player_cities["worked_slots"])
        worked_slots_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in worked_slots_np]]]
        
        # yields: (steps, players, cities, 7)
        # need to add tourism!
        yields = self.player_cities["yields"]
        tourism = self.player_cities["building_yields"][..., -1]
        yields = np.concatenate([yields, tourism[..., None]], axis=-1)
        yields_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in yields]]]
        
        # pop (num_steps, players, cities)
        pop = self.player_cities["population"]
        pop_lol = [inner_l.tolist() for inner_l in [l for l in pop]]

        # buildings_owned (steps, players, cities, len(GameBuildings))
        bldgs = self.player_cities["buildings_owned"]
        bldgs_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in bldgs]]]


        movement_cost_np = np.array(self.movement_cost_map)
        movement_cost_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in movement_cost_np]]]

        #can_move_to_np = np.array(self.can_move_to)
        #can_move_to_lol = [inner_l.tolist() for inner_l in [l for l in [x for x in can_move_to_np]]]

        units_type = np.array(self.units["unit_type"])
        units_military = np.array(self.units["military"])
        units_rowcol = np.array(self.units["unit_rowcol"])
        units_trade_player_to = np.array(self.units["trade_to_player_int"])
        units_trade_city_to = np.array(self.units["trade_to_city_int"])
        units_trade_city_from = np.array(self.units["trade_from_city_int"])
        units_trade_yields = np.array(self.units["trade_yields"])
        units_engaged = np.array(self.units["engaged_for_n_turns"])

        units_type = [l.tolist() for l in [x for x in units_type]]
        units_military = [l.tolist() for l in [x for x in units_military]]
        units_rowcol = [l.tolist() for l in [x for x in units_rowcol]]
        units_trade_player_to = [l.tolist() for l in [x for x in units_trade_player_to]]
        units_trade_city_to = [l.tolist() for l in [x for x in units_trade_city_to]]
        units_trade_city_from = [l.tolist() for l in [x for x in units_trade_city_from]]
        units_trade_yields = [l.tolist() for l in [x for x in units_trade_yields]]
        units_engaged = [l.tolist() for l in [x for x in units_engaged]]

        techs = np.array(self.technologies)
        techs = [l.tolist() for l in [x for x in techs]]

        policies = np.array(self.policies)
        policies = [l.tolist() for l in [x for x in policies]]

        religious_tenets = np.array(self.religious_tenets)
        religious_tenets = [l.tolist() for l in [x for x in religious_tenets]]
        
        wonder_accel = np.array(self.player_cities["wonder_accel"])
        wonder_accel_religion = np.array(self.player_cities["religion_info"]["wonder_accel"])
        wonder_accel_culture = np.array(self.culture_info["wonder_accel"])
        wonder_accel = wonder_accel + wonder_accel_religion + wonder_accel_culture - 2
        wonder_accel = [l.tolist() for l in [x for x in wonder_accel]]

        bldg_accel = np.array(self.player_cities["bldg_accel"])
        bldg_accel_culture = np.array(self.culture_info["bldg_accel"])
        bldg_accel += bldg_accel_culture - 1
        bldg_accel = [l.tolist() for l in [x for x in bldg_accel]]

        military_bldg_accel = np.array(self.player_cities["military_bldg_accel"])
        military_bldg_accel_culture = np.array(self.culture_info["military_bldg_accel"])
        military_bldg_accel += military_bldg_accel_culture - 1
        military_bldg_accel = [l.tolist() for l in [x for x in military_bldg_accel]]
        
        religion_bldg_accel = np.array(self.player_cities["religion_bldg_accel"])
        religion_bldg_accel_culture = np.array(self.culture_info["religion_bldg_accel"])
        religion_bldg_accel += religion_bldg_accel_culture - 1
        religion_bldg_accel = [l.tolist() for l in [x for x in religion_bldg_accel]]
        
        culture_bldg_accel = np.array(self.player_cities["culture_bldg_accel"])
        culture_bldg_accel_culture = np.array(self.culture_info["culture_bldg_accel"])
        culture_bldg_accel += culture_bldg_accel_culture - 1
        culture_bldg_accel = [l.tolist() for l in [x for x in culture_bldg_accel]]

        sea_bldg_accel = np.array(self.player_cities["sea_bldg_accel"])
        sea_bldg_accel_culture = np.array(self.culture_info["sea_bldg_accel"])
        sea_bldg_accel += sea_bldg_accel_culture - 1
        sea_bldg_accel = [l.tolist() for l in [x for x in sea_bldg_accel]]

        science_bldg_accel = np.array(self.player_cities["science_bldg_accel"])
        science_bldg_accel_culture = np.array(self.culture_info["science_bldg_accel"])
        science_bldg_accel += science_bldg_accel_culture - 1
        science_bldg_accel = [l.tolist() for l in [x for x in science_bldg_accel]]

        econ_bldg_accel = np.array(self.player_cities["economy_bldg_accel"])
        econ_bldg_accel_culture = np.array(self.culture_info["econ_bldg_accel"])
        econ_bldg_accel += econ_bldg_accel_culture - 1
        econ_bldg_accel = [l.tolist() for l in [x for x in econ_bldg_accel]]
        
        unit_accel = np.array(self.player_cities["unit_accel"])
        unit_accel = [l.tolist() for l in [x for x in unit_accel]]
        
        yield_accel = np.array(self.player_cities["citywide_yield_accel"])
        yield_accel = [l.tolist() for l in [x for x in yield_accel]]

        border_accel = np.array(self.player_cities["border_growth_accel"])
        border_accel = [l.tolist() for l in [x for x in border_accel]]

        city_religious_tenets = np.array(self.player_cities["religion_info"]["religious_tenets_per_city"])
        city_religious_tenets = [l.tolist() for l in [x for x in city_religious_tenets]]

        city_religious_pop = np.array(self.player_cities["religion_info"]["religious_population"])
        city_religious_pop = [l.tolist() for l in [x for x in city_religious_pop]]

        gw_slots = np.array(self.player_cities["gw_slots"])
        gw_slots = [l.tolist() for l in [x for x in gw_slots]]

        specialist_slots = np.array(self.player_cities["specialist_slots"])
        specialist_slots = [l.tolist() for l in [x for x in specialist_slots]]

        bldg_maintenance = np.array(self.player_cities["bldg_maintenance"])
        bldg_maintenance = [l.tolist() for l in [x for x in bldg_maintenance]]

        unit_xp_add = np.array(self.player_cities["unit_xp_add"])
        unit_xp_add = [l.tolist() for l in [x for x in unit_xp_add]]

        can_trade_food = np.array(self.player_cities["can_trade_food"])
        can_trade_food = [l.tolist() for l in [x for x in can_trade_food]]

        can_trade_prod = np.array(self.player_cities["can_trade_prod"])
        can_trade_prod = [l.tolist() for l in [x for x in can_trade_prod]]

        defense = np.array(self.player_cities["defense"])
        defense = [l.tolist() for l in [x for x in defense]]

        hp = np.array(self.player_cities["hp"])
        hp = [l.tolist() for l in [x for x in hp]]
        
        trade_gold_add_owner = np.array(self.player_cities["trade_gold_add_owner"])
        trade_gold_add_owner = [l.tolist() for l in [x for x in trade_gold_add_owner]]
        
        trade_gold_add_dest = np.array(self.player_cities["trade_gold_add_dest"])
        trade_gold_add_dest = [l.tolist() for l in [x for x in trade_gold_add_dest]]
        
        trade_land_dist_mod = np.array(self.player_cities["trade_land_dist_mod"])
        trade_land_dist_mod = [l.tolist() for l in [x for x in trade_land_dist_mod]]
        
        trade_sea_dist_mod = np.array(self.player_cities["trade_sea_dist_mod"])
        trade_sea_dist_mod = [l.tolist() for l in [x for x in trade_sea_dist_mod]]
        
        naval_movement_add = np.array(self.player_cities["naval_movement_add"])
        naval_movement_add = [l.tolist() for l in [x for x in naval_movement_add]]

        naval_sight_add = np.array(self.player_cities["naval_sight_add"])
        naval_sight_add = [l.tolist() for l in [x for x in naval_sight_add]]
        
        gp_accel = np.array(self.player_cities["great_person_accel"])
        gp_accel = [l.tolist() for l in [x for x in gp_accel]]

        mounted_accel = np.array(self.player_cities["mounted_accel"])
        mounted_accel = [l.tolist() for l in [x for x in mounted_accel]]

        land_unit_accel = np.array(self.player_cities["land_unit_accel"])
        land_unit_accel = [l.tolist() for l in [x for x in land_unit_accel]]

        sea_unit_accel = np.array(self.player_cities["sea_unit_accel"])
        sea_unit_accel = [l.tolist() for l in [x for x in sea_unit_accel]]

        tech_steal_reduce_accel = np.array(self.player_cities["tech_steal_reduce_accel"])
        tech_steal_reduce_accel = [l.tolist() for l in [x for x in tech_steal_reduce_accel]]
        
        gw_tourism_accel = np.array(self.player_cities["gw_tourism_accel"])
        gw_tourism_accel = [l.tolist() for l in [x for x in gw_tourism_accel]]

        culture_to_tourism = np.array(self.player_cities["culture_to_tourism"])
        culture_to_tourism = [l.tolist() for l in [x for x in culture_to_tourism]]
        
        air_unit_capacity = np.array(self.player_cities["air_unit_capacity"])
        air_unit_capacity = [l.tolist() for l in [x for x in air_unit_capacity]]

        spaceship_prod_accel = np.array(self.player_cities["spaceship_prod_accel"])
        spaceship_prod_accel = [l.tolist() for l in [x for x in spaceship_prod_accel]]
        
        city_connection_gold_accel = np.array(self.player_cities["city_connection_gold_accel"])
        city_connection_gold_accel = [l.tolist() for l in [x for x in city_connection_gold_accel]]

        armored_accel = np.array(self.player_cities["armored_accel"])
        armored_accel = [l.tolist() for l in [x for x in armored_accel]]

        improvement_map = np.array(self.improvement_map)
        improvement_map = [l.tolist() for l in [x for x in improvement_map]]
        
        road_map = np.array(self.road_map)
        road_map = [l.tolist() for l in [x for x in road_map]]

        culture_total = np.array(self.culture_total)
        culture_total = [l.tolist() for l in [x for x in culture_total]]
        tourism_total = np.array(self.tourism_total)
        tourism_total = [l.tolist() for l in [x for x in tourism_total]]

        is_constructing = np.array(self.player_cities["is_constructing"])
        is_constructing = [l.tolist() for l in [x for x in is_constructing]]
        prod_reserves = np.array(self.player_cities["prod_reserves"])
        prod_reserves = [l.tolist() for l in [x for x in prod_reserves]]

        cs_religious_population = self.citystate_info["religious_population"]
        cs_religious_population = [l.tolist() for l in [x for x in cs_religious_population]]
        cs_relationships = self.citystate_info["relationships"]
        cs_relationships = [l.tolist() for l in [x for x in cs_relationships]]
        cs_influence = self.citystate_info["influence_level"]
        cs_influence = [l.tolist() for l in [x for x in cs_influence]]
        cs_type = self.citystate_info["cs_type"]
        cs_type = [l.tolist() for l in [x for x in cs_type]]
        cs_quest = self.citystate_info["quest_type"]
        cs_quest = [l.tolist() for l in [x for x in cs_quest]]
        cs_culture_tracker = self.citystate_info["culture_tracker"]
        cs_culture_tracker = [l.tolist() for l in [x for x in cs_culture_tracker]]
        cs_faith_tracker = self.citystate_info["faith_tracker"]
        cs_faith_tracker = [l.tolist() for l in [x for x in cs_faith_tracker]]
        cs_tech_tracker = self.citystate_info["tech_tracker"]
        cs_tech_tracker = [l.tolist() for l in [x for x in cs_tech_tracker]]
        cs_trade_tracker = self.citystate_info["trade_tracker"]
        cs_trade_tracker = [l.tolist() for l in [x for x in cs_trade_tracker]]
        cs_religion_tracker = self.citystate_info["religion_tracker"]
        cs_religion_tracker = [l.tolist() for l in [x for x in cs_religion_tracker]]
        cs_wonder_tracker = self.citystate_info["wonder_tracker"]
        cs_wonder_tracker = [l.tolist() for l in [x for x in cs_wonder_tracker]]
        cs_resource_tracker = self.citystate_info["resource_tracker"]
        cs_resource_tracker = [l.tolist() for l in [x for x in cs_resource_tracker]]

        fow = np.array(self.visibility_map)
        fow = [l.tolist() for l in [x for x in [y for y in fow]]]

        trade_ledger = np.array(self.trade_ledger)
        trade_ledger = [l.tolist() for l in [x for x in trade_ledger]]
        trade_length_ledger = np.array(self.trade_length_ledger)
        trade_length_ledger = [l.tolist() for l in [x for x in trade_length_ledger]]
        trade_gpt_adj = np.array(self.trade_gpt_adjustment)
        trade_gpt_adj = [l.tolist() for l in [x for x in trade_gpt_adj]]
        trade_resource_adj = np.array(self.trade_resource_adjustment)
        trade_resource_adj = [l.tolist() for l in [x for x in trade_resource_adj]]

        have_met = np.array(self.have_met)
        have_met = [l.tolist() for l in [x for x in have_met]]

        at_war = np.array(self.at_war)
        at_war = [l.tolist() for l in [x for x in at_war]]
        
        unit_health = np.array(self.units["health"])
        unit_health = [l.tolist() for l in [x for x in unit_health]]

        has_sacked = np.array(self.has_sacked)
        has_sacked = [l.tolist() for l in [x for x in has_sacked]]

        treasury = np.array(self.treasury)
        treasury = [l.tolist() for l in [x for x in treasury]]

        happiness = np.array(self.happiness)
        happiness = [l.tolist() for l in [x for x in happiness]]

        resources_owned = np.array(self.player_cities["resources_owned"])
        resources_owned = [l.tolist() for l in [x for x in resources_owned]]

        combat_bonus_accel = np.array(self.units["combat_bonus_accel"])
        combat_bonus_accel = [l.tolist() for l in [x for x in combat_bonus_accel]]

        gpps = np.array(self.gpps)
        gpps = [l.tolist() for l in [x for x in gpps]]
        gp_threshold = np.array(self.gp_threshold)
        gp_threshold = [l.tolist() for l in [x for x in gp_threshold]]

        golden_age_turns = np.array(self.golden_age_turns)
        golden_age_turns = [l.tolist() for l in [x for x in golden_age_turns]]

        num_delegates = np.array(self.num_delegates)
        num_delegates = [l.tolist() for l in [x for x in num_delegates]]

        gamestate = {
            "num_delegates": num_delegates,
            "golden_age_turns": golden_age_turns,
            "gpps": gpps,
            "gpp_threshold": gp_threshold,
            "combat_bonus_accel": combat_bonus_accel,
            "resources_owned": resources_owned,
            "treasury": treasury,
            "happiness": happiness,
            "has_sacked": has_sacked,
            "unit_health": unit_health,
            "at_war": at_war,
            "have_met": have_met,
            "units_engaged": units_engaged,
            "trade_ledger": trade_ledger,
            "trade_length_ledger": trade_length_ledger,
            "trade_gpt_adj": trade_gpt_adj,
            "trade_resource_adj": trade_resource_adj,
            "fog_of_war": fow,
            "cs_religious_population": cs_religious_population,
            "cs_relationships": cs_relationships,
            "cs_influence": cs_influence,
            "cs_type": cs_type,
            "cs_quest": cs_quest,
            "cs_culture_tracker": cs_culture_tracker,
            "cs_faith_tracker": cs_faith_tracker,
            "cs_tech_tracker": cs_tech_tracker,
            "cs_trade_tracker": cs_trade_tracker,
            "cs_religion_tracker": cs_religion_tracker,
            "cs_wonder_tracker": cs_wonder_tracker,
            "cs_resource_tracker": cs_resource_tracker,
            "is_constructing": is_constructing,
            "prod_reserves": prod_reserves,
            "culture_total": culture_total,
            "tourism_total": tourism_total,
            "improvement_map": improvement_map,
            "road_map": road_map,
            "armored_accel": armored_accel,
            "city_connection_gold_accel": city_connection_gold_accel,
            "naval_movement_add": naval_movement_add,
            "naval_sight_add": naval_sight_add,
            "spaceship_prod_accel": spaceship_prod_accel,
            "air_unit_capacity": air_unit_capacity,
            "culture_to_tourism": culture_to_tourism,
            "gw_tourism_accel": gw_tourism_accel,
            "sea_unit_accel": sea_unit_accel,
            "tech_steal_reduce_accel": tech_steal_reduce_accel,
            "land_unit_accel": land_unit_accel,
            "mounted_accel": mounted_accel,
            "gp_accel": gp_accel,
            "trade_gold_add_owner": trade_gold_add_owner,
            "trade_gold_add_dest": trade_gold_add_dest,
            "trade_land_dist_mod": trade_land_dist_mod,
            "trade_sea_dist_mod": trade_sea_dist_mod,
            "defense": defense,
            "hp": hp,
            "can_trade_food": can_trade_food,
            "can_trade_prod": can_trade_prod,
            "unit_xp_add": unit_xp_add,
            "bldg_maintenance": bldg_maintenance,
            "terrain": terrain_lol,
            "rivers": rivers_lolol,
            "lakes": lakes_lol,
            "elevation": elevation_lol,
            "features": features_lol,
            "nw": nw_lol,
            "units_type": units_type,
            "units_military": units_military,
            "units_rowcol": units_rowcol,
            "units_trade_player_to": units_trade_player_to,
            "units_trade_city_to": units_trade_city_to,
            "units_trade_city_from": units_trade_city_from,
            "units_trade_yields": units_trade_yields,
            "player_rowcols": player_rowcols,
            "player_ownership": player_ownership_lol,
            "player_ownership_borders": player_ownership_borders_lol,
            "player_worked_slots": worked_slots_lol,
            "player_yields": yields_lol,
            "player_pops": pop_lol,
            "player_buildings": bldgs_lol,
            "cs_rowcols": cs_rowcols,
            "cs_ownership": cs_ownership_lol,
            "cs_ownership_borders": cs_ownership_borders_lol,
            "all_resource_map": all_resource_map_lol,
            "gt_yield_map": yield_map_lolol,
            "player_yield_map": player_yield_map_lololol,
            "movement_cost_map": movement_cost_lol,
            "techs": techs,
            "policies": policies,
            "religious_tenets": religious_tenets,
            "wonder_accel": wonder_accel,
            "bldg_accel": bldg_accel,
            "military_bldg_accel": military_bldg_accel,
            "culture_bldg_accel": culture_bldg_accel,
            "sea_bldg_accel": sea_bldg_accel,
            "science_bldg_accel": science_bldg_accel,
            "religion_bldg_accel": religion_bldg_accel,
            "econ_bldg_accel": econ_bldg_accel,
            "unit_accel": unit_accel,
            "yield_accel":  yield_accel,
            "border_accel": border_accel,
            "city_religious_tenets": city_religious_tenets,
            "gw_slots": gw_slots,
            "specialist_slots": specialist_slots,
            "city_religious_pop": city_religious_pop,
        }
        
        print("Compressing save file...")
        atomic_write_ndjson_gzip(iter_gamestate_lines(gamestate), filename + ".ndjson.gz")
        print("Saving complete.")
