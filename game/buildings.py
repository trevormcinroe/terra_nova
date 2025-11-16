from functools import partial
import jax.numpy as jnp
import jax
import enum
from game.constants import DESERT_IDX, JUNGLE_IDX, MOUNTAIN_IDX, OASIS_IDX, TO_ZERO_OUT_FOR_BUILDINGS_STEP, TO_ZERO_OUT_FOR_BUILDINGS_STEP_ONLY_MAPS, TO_ZERO_OUT_FOR_BUILDINGS_STEP_SANS_MAPS, TUNDRA_IDX, make_update_fn
from game.resources import RESOURCE_TO_IDX
from game.techs import Technologies
import sys
from flax.struct import dataclass
from dataclasses import fields, replace


class GameBuildings(enum.IntEnum):
    """
   gold_maintenance, conquest_prob, water, river, freshwater, mountain, nearby_mountain_required, hill, flat, cost, faith_cost, holy_city, free_techs, free_policies, free_great_people, defense, religious_pressure, extra_spies, trade_recipient_bonus, trade_target_bonus, trade_num_bonus, trade_land_distance_mod, trade_sea_distance_mod, trade_land_gold_mod, trade_sea_gold_mod, trade_allows_food, trade_allows_production, tourism_landmark_pct, tourism_gw_mod, prereq_tech, specialist_type, specialist_count, gw_count, great_people_rate_change, mountain_science_yield, yield_city_add, yield_city_pct, yield_pop_pct, gw_type, prereq_building, resource_prereq, world_wonder, national_wonder, building_type, requires_coastal

   Types:
   1. Regular
   2. Military
   3. Religious
   4. Culture
   5. Economic
   6. Sea
   7. Science
   8. National Wonder
   9. World Wonder
   10. Spaceship parts
    """
    courthouse = 0, 4, 0.0, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.mathematics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 2, False

    seaport = 1, 2, 0.66, True, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.navigation._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["harbor"], [], False, False, [], 6, True

    stable = 2, 1, 0.0, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.horseback_riding._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["horses"], RESOURCE_TO_IDX["sheep"], RESOURCE_TO_IDX["cow"]], False, False, [], 1, False

    watermill = 3, 2, 0.66, False, True, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.the_wheel._value_, None, 0, 0, 0, 0, [2, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    circus = 4, 0, 0.66, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.trapping._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["horses"], RESOURCE_TO_IDX["ivory"]], False, False, [], 1, False

    forge = 5, 1, 0.66, False, False, False, False, False, False, False, 80.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.metal_casting._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["iron"]], False, False, [], 1, False

    windmill = 6, 2, 0.66, False, False, False, False, False, False, False, 166.66666666666666, 320, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.economics._value_, "SPECIALIST_ENGINEER", 1, 0, 0, 0, [0, 2, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [], 1, False

    hydro_plant = 7, 3, 0.66, False, True, False, False, False, False, False, 280.0, 520, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.electricity._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["aluminium"]], False, False , [], 1, False

    solar_plant = 8, 3, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.ecology._value_, None, 0, 0, 0, 0, [0, 5, 0, 0, 0, 0, 0, 0], [1, 1.15, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["factory"], [], False, False, [], 1, False

    mint = 9, 0, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.currency._value_, None, 0, 0, 0, 0, [0, 0, 2, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["gold"], RESOURCE_TO_IDX["silver"], RESOURCE_TO_IDX["copper"]], False, False , [], 5, False

    observatory = 10, 1, 0.66, False, False, False, False, False, False, False, 133.33333333333334, 260, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.astronomy._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 4, 0, 0, 0], [1, 1, 1, 1, 1.0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 7, False

    monastery = 11, 0, 0.66, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 2, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    garden = 12, 1, 0.66, False, False, True, False, False, False, False, 80.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.drama._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    lighthouse = 13, 1, 0.66, True, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.optics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [],  6, True

    harbor = 14, 2, 0.66, True, False, False, False, False, False, False, 80.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 100, False, False, 0, 0, Technologies.compass._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 6,  True

    colosseum = 15, 1, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.construction._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    theatre = 16, 2, 0.66, False, False, False, False, False, False, False, 120.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.printing_press._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["colosseum"], [], False, False, [], 4, False

    stadium = 17, 2, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.refrigeration._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["theatre"], [], False, False , [], 1, False

    monument = 18, 1, 0.0, False, False, False, False, False, False, False, 26.666666666666668, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 4, False

    temple = 19, 2, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.philosophy._value_, None, 0, 0, 0, 0, [0, 0, 0, 2, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["shrine"], [], False, False , [], 3, False

    opera_house = 20, 1, 1.0, False, False, False, False, False, False, False, 133.33333333333334, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.acoustics._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "music", ["amphitheater"], [], False, False, [], 4, False

    museum = 21, 1, 1.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.archaeology._value_, None, 0, 2, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", ["amphitheater"], [], False, False , [], 4, False

    broadcast_tower = 22, 3, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.radio._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "music", ["opera_house"], [], False, False , [], 4, False

    barracks = 23, 1, 0.0, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.bronze_working._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [], 2, False

    armory = 24, 1, 0.0, False, False, False, False, False, False, False, 106.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.steel._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["barracks"], [], False, False, [], 2, False

    military_academy = 25, 1, 0.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.military_science._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["armory"], [], False, False , [], 2, False

    arsenal = 26, 0, 0.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 900, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.metallurgy._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["castle"], [], False, False , [], 2, False

    walls = 27, 0, 0.0, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.masonry._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    castle = 28, 0, 0.0, False, False, False, False, False, False, False, 106.66666666666667, 0, False, 0, 0, 0, 700, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.chivalry._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["walls"], [], False, False, [], 1, False

    military_base = 29, 0, 0.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 1200, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.replaceable_parts._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["arsenal"], [], False, False, [], 1, False

    granary = 30, 1, 0.66, False, False, False, False, False, False, False, 40.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True, False, 0, 0, Technologies.pottery._value_, None, 0, 0, 0, 0, [2, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    hospital = 31, 2, 0.66, False, False, False, False, False, False, False, 240.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.biology._value_, None, 0, 0, 0, 0, [5, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["aqueduct"], [], False, False, [], 1, False

    medical_lab = 32, 3, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.electronics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["hospital"], [], False, False, [], 7, False

    workshop = 33, 2, 0.66, False, False, False, False, False, False, False, 80.0, 200, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, True, 0, 0, Technologies.metal_casting._value_, "SPECIALIST_ENGINEER", 1, 0, 0, 0, [0, 2, 0, 0, 0, 0, 0, 0], [1, 1.1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    factory = 34, 3, 0.66, False, False, False, False, False, False, False, 240.0, 450, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.industrialization._value_, "SPECIALIST_ENGINEER", 2, 0, 0, 0, [0, 4, 0, 0, 0, 0, 0, 0], [1, 1.1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["workshop"], [RESOURCE_TO_IDX["coal"]], False, False, [], 1, False

    nuclear_plant = 35, 3, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.nuclear_fission._value_, None, 0, 0, 0, 0, [0, 5, 0, 0, 0, 0, 0, 0], [1, 1.15, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["factory"], [RESOURCE_TO_IDX["uranium"]], False, False, [], 1, False

    spaceship_factory = 36, 3, 0.66, False, False, False, False, False, False, False, 240.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.robotics._value_, None, 0, 0, 0, 0, [0, 3, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["factory"], [], False, False, [], 1, False

    market = 37, 0, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.currency._value_, "SPECIALIST_MERCHANT", 1, 0, 0, 0, [0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1.25, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [],  5, False

    bank = 38, 0, 0.66, False, False, False, False, False, False, False, 133.33333333333334, 0, False, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.banking._value_, "SPECIALIST_MERCHANT", 1, 0, 0, 0, [0, 0, 2, 0, 0, 0, 0, 0], [1, 1, 1.25, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["market"], [], False, False, [], 5, False

    stock_exchange = 39, 0, 0.66, False, False, False, False, False, False, False, 240.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.electricity._value_, "SPECIALIST_MERCHANT", 2, 0, 0, 0, [0, 0, 3, 0, 0, 0, 0, 0], [1, 1, 1.25, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["bank"], [], False, False , [], 5, False

    library = 40, 1, 0.66, False, False, False, False, False, False, False, 50.0, 60, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.writing._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1.5, 1, 1, 1], "None", [], [], False, False, [], 7, False

    university = 41, 2, 0.66, False, False, False, False, False, False, False, 106.66666666666667, 220, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.education._value_, "SPECIALIST_SCIENTIST", 2, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1.33, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["library"], [], False, False, [], 7, False

    public_school = 42, 3, 0.66, False, False, False, False, False, False, False, 200.0, 380, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.scientific_theory._value_, "SPECIALIST_SCIENTIST", 1, 0, 0, 0, [0, 0, 0, 0, 3, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1.5, 1, 1, 1], "None", ["university"], [], False, False, [],  7, False

    laboratory = 43, 3, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 630, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.plastic._value_, "SPECIALIST_SCIENTIST", 1, 0, 0, 0, [0, 0, 0, 0, 4, 0, 0, 0], [1, 1, 1, 1, 1.5, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["public_school"], [], False, False, [],  7, False

    palace = 44, 0, 0.0, False, False, False, False, False, False, False, -0.6666666666666666, 0, False, 0, 0, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 1, 0, 0, [0, 3, 3, 0, 3, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [], False, True, [],  1, False

    heroic_epic = 45, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.iron_working._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", ["barracks"], [], False, True, [], 8, False

    national_college = 46, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.philosophy._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 3, 1, 0, 0], [1, 1, 1, 1, 1.5, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 8, False

    national_epic = 47, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.drama._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", ["monument"], [], False, True, [], 8, False

    circus_maximus = 48, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.horseback_riding._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["circus"], [],  False, True, [], 8, False

    national_treasury = 49, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 4, 2, 1, 0, 0, 0, 0, False, False, 0, 0, Technologies.guilds._value_, None, 0, 0, 0, 0, [0, 0, 4, 0, 0, 0, 0, 0], [1, 1, 1.1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["market"], [], False, True, [], 8, False
 
    ironworks = 50, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.machinery._value_, None, 0, 0, 0, 0, [0, 8, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["workshop"], [],  False, True, [], 8, False

    oxford_university = 51, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.education._value_, None, 0, 2, 0, 0, [0, 0, 0, 0, 3, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", [], [],  False, True, [],  8, False
    
    hermitage = 52, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.architecture._value_, None, 0, 3, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", ["opera_house"], [],  False, True, [], 8, False

    great_lighthouse = 53, 0, 1.0, True, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.optics._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, True

    stonehenge = 54, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.calendar._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 6, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    great_library = 55, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.writing._value_, "SPECIALIST_SCIENTIST", 0, 2, 1, 0, [0, 0, 0, 0, 3, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", [], [], True, False, [], 9, False 

    pyramid = 56, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.masonry._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [6], 9, False

    colossus = 57, 0, 1.0, True, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, False, False, 0, 0, Technologies.iron_working._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 5, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, True
 
    oracle = 58, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.philosophy._value_, "SPECIALIST_SCIENTIST", 0, 0, 1, 0, [0, 0, 0, 0, 0, 3, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [],  9, False

    hanging_garden = 59, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.mathematics._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [6, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False , [0], 9, False

    great_wall = 60, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.engineering._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False

    angkor_wat = 61, 0, 1.0, False, False, False, False, False, False, False, 266.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.education._value_, "SPECIALIST_SCIENTIST", 0, 0, 1, 0, [0, 0, 0, 0, 5, 5, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    hagia_sophia = 62, 0, 1.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.theology._value_, None, 0, 0, 0, 0, [0, 0, 0, 3, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [],  9, False

    chichen_itza = 63, 0, 1.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.civil_service._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 4, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    machu_pichu = 64, 0, 1.0, False, False, False, False, True, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.guilds._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 5, 2, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    notre_dame = 65, 0, 1.0, False, False, False, False, False, False, False, 266.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.physics._value_, "SPECIALIST_ARTIST", 0, 0, 1, 0, [0, 0, 0, 4, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    porcelain_tower = 66, 0, 1.0, False, False, False, False, False, False, False, 416.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.architecture._value_, "SPECIALIST_SCIENTIST", 0, 0, 2, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [48], 9, False

    himeji_castle = 67, 0, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.gunpowder._value_, "SPECIALIST_ENGINEER", 0, 0, 2, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    sistine_chapel = 68, 0, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.acoustics._value_, None, 0, 2, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [], True, False, [], 9, False

    kremlin = 69, 0, 1.0, False, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.railroad._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    forbidden_palace = 70, 0, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.banking._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [24], 9, False
 
    taj_mahal = 71, 0, 1.0, False, False, False, False, False, False, False, 416.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.architecture._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 4, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    big_ben = 72, 0, 1.0, False, False, False, False, False, False, False, 500.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.industrialization._value_, "SPECIALIST_MERCHANT", 0, 0, 2, 0, [0, 0, 4, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [36], 9, False
 
    louvre = 73, 0, 1.0, False, False, False, False, False, False, False, 500.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.archaeology._value_, None, 0, 4, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [], True, False, [42], 9, False

    brandenburg_gate = 74, 0, 1.0, False, False, False, False, False, False, False, 500.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.military_science._value_, "SPECIALIST_SCIENTIST", 0, 0, 2, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False , [], 9, False

    statue_of_liberty = 75, 0, 1.0, False, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.replaceable_parts._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 6, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False

    cristo_redentor = 76, 0, 1.0, False, False, False, False, False, False, False, 802.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.plastic._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 5, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [],  9, False
 
    eiffel_tower = 77, 0, 1.0, False, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.radio._value_, "SPECIALIST_MERCHANT", 0, 0, 2, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    pentagon = 78, 0, 1.0, False, False, False, False, False, False, False, 802.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.combined_arms._value_, "SPECIALIST_MERCHANT", 0, 0, 2, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    sydney_opera_house = 79, 0, 1.0, False, False, False, False, False, False, False, 833.3333333333334, 0, False, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.ecology._value_, None, 0, 2, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "music", [], [], True, False, [], 9, False
 
    aqueduct = 80, 1, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.engineering._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False
 
    stone_works = 81, 1, 0.66, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.calendar._value_, None, 0, 0, 0, 0, [0, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["marble"], RESOURCE_TO_IDX["stone"], RESOURCE_TO_IDX["obsidian"]], False, False, [], 1, False

    statue_zeus = 82, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.bronze_working._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 3, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    temple_artemis = 83, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.archery._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 3, 3, 0, 0, 3, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [12], 9, False

    mausoleum_halicarnassus = 84, 0, 1.0, False, False, False, False, False, False, False, 123.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.masonry._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    amphitheater = 85, 1, 1.0, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.drama._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", ["monument"], [], False, False, [], 4, False

    shrine = 86, 1, 0.66, False, False, False, False, False, False, False, 26.666666666666668, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.pottery._value_, None, 0, 0, 0, 0, [0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    recycling_center = 87, 3, 0.66, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.ecology._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [], 1, False

    bomb_shelter = 88, 1, 0.66, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.telecom._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    constable = 89, 1, 0.0, False, False, False, False, False, False, False, 106.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.banking._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    police_station = 90, 1, 0.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.electricity._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["constable"], [], False, False, [], 1, False

    intelligence_agency = 91, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.radio._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["constable"], [], False, True , [], 8, False

    alhambra = 92, 0, 1.0, False, False, False, False, False, False, False, 266.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.chivalry._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    cn_tower = 93, 0, 1.0, False, False, False, False, False, False, False, 802.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.radar._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    hubble = 94, 0, 1.0, False, False, False, False, False, False, False, 833.3333333333334, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.satellites._value_, "SPECIALIST_SCIENTIST", 0, 0, 3, 0, [0, 0, 0, 0, 10, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False

    leaning_tower = 95, 0, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.printing_press._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    mosque_of_djenne = 96, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.philosophy._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 6, 0, 3, 3, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [18], 9, False

    neuschwanstein = 97, 0, 1.0, False, False, False, False, True, False, False, 500.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.rifling._value_, "SPECIALIST_MERCHANT", 0, 0, 1, 0, [0, 0, 6, 0, 0, 4, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    petra = 98, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, False, False, 0, 0, Technologies.currency._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    terracotta_army = 99, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.construction._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    great_firewall = 100, 0, 1.0, False, False, False, False, False, False, False, 833.3333333333334, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.computers._value_, "SPECIALIST_SCIENTIST", 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    cathedral = 101, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 5, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [], False, False, [], 3, False

    mosque = 102, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 3, 0, 3, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    pagoda = 103, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 200, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 2, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    grand_temple = 104, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, True, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.theology._value_, None, 0, 0, 0, 0, [0, 0, 0, 8, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["temple"], [], False, True, [], 3, False

    tourist_center = 105, 2, 0.0, False, False, False, False, False, False, False, 266.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 100, 100, Technologies.telecom._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 8, False

    writers_guild = 106, 1, 0.0, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.drama._value_, "SPECIALIST_WRITER", 2, 0, 1, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    artists_guild = 107, 1, 0.0, False, False, False, False, False, False, False, 100.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.guilds._value_, "SPECIALIST_ARTIST", 2, 0, 2, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    musicians_guild = 108, 1, 0.0, False, False, False, False, False, False, False, 133.33333333333334, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.acoustics._value_, "SPECIALIST_MUSICIAN", 2, 0, 3, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    hotel = 109, 0, 0.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 50, 50, Technologies.refrigeration._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 4, False

    caravansary = 110, 0, 0.66, False, False, False, False, False, False, False, 60.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 200, 0, False, False, 0, 0, Technologies.horseback_riding._value_, None, 0, 0, 0, 0, [0, 0, 2, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 5, False

    airport = 111, 5, 0.66, False, False, False, False, False, False, False, 266.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 50, 50, Technologies.radar._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 1, False

    uffizi = 112, 0, 1.0, False, False, False, False, False, False, False, 416.6666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.architecture._value_, None, 0, 3, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [],  True, False, [30], 9, False

    globe_theater = 113, 0, 1.0, False, False, False, False, False, False, False, 333.3333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.printing_press._value_, "SPECIALIST_WRITER", 0, 2, 1, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "literature", [], [],  True, False, [], 9, False

    broadway = 114, 0, 1.0, False, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.radio._value_, None, 0, 3, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "music", [], [],  True, False, [], 9, False

    red_fort = 115, 0, 1.0, False, False, False, False, False, False, False, 416.6666666666667, 0, False, 0, 0, 0, 1200, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.metallurgy._value_, "SPECIALIST_SCIENTIST", 0, 0, 1, 0, [0, 0, 0, 0, 0, 8, 4, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    prora_resort = 116, 0, 1.0, False, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.flight._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    borobudur = 117, 0, 1.0, False, False, False, False, False, False, False, 200.0, 0, True, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.theology._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 5, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    parthenon = 118, 0, 1.0, False, False, False, False, False, False, False, 166.66666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.drama._value_, None, 0, 1, 0, 0, [0, 0, 0, 0, 0, 4, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "art_artifact", [], [],  True, False, [], 9, False

    international_space_station = 119, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], True, False, [], 9, False
 
    gurdwara = 120, 0, 0.0, False, False, False, False, False, False, False, -0.6666666666666666, 270, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    synagogue = 121, 0, 0.0, False, False, False, False, False, False, False, -0.6666666666666666, 200, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 1, 0, 2, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [], 3, False

    conservatory = 122, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, "SPECIALIST_MUSICIAN", 1, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False , [], 3, False

    vihara = 123, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, "SPECIALIST_MERCHANT", 1, 0, 0, 0, [0, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1.15, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    mandir = 124, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 270, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 3, False

    stpeters = 125, 0, 0.0, False, False, False, False, False, False, False, 200, 0, True, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.education._value_, None, 0, 0, 0, 0, [0, 0, 0, 8, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    althing = 126, 0, 1.0, False, False, False, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.metal_casting._value_, "SPECIALIST_SCIENTIST", 0, 0, 1, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False

    gemcutter = 127, 1, 0.66, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.machinery._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["gems"], RESOURCE_TO_IDX["jade"], RESOURCE_TO_IDX["amber"], RESOURCE_TO_IDX["lapis"]], False, False, [], 4, False

    textile = 128, 2, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.economics._value_, None, 0, 0, 0, 0, [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["silk"], RESOURCE_TO_IDX["cotton"], RESOURCE_TO_IDX["dyes"], RESOURCE_TO_IDX["furs"]], False, False, [], 1, False

    censer = 129, 1, 0.66, False, False, False, False, False, False, False, 50.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.theology._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["incense"], RESOURCE_TO_IDX["perfume"], RESOURCE_TO_IDX["tobacco"]], False, False, [], 4, False

    brewery = 130, 0, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.guilds._value_, None, 0, 0, 0, 0, [0, 0, 2, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["wine"], RESOURCE_TO_IDX["salt"], RESOURCE_TO_IDX["coffee"], RESOURCE_TO_IDX["tea"], RESOURCE_TO_IDX["sugar"]], False, False, [], 5, False

    grocer = 131, 1, 0.66, False, False, False, False, False, False, False, 66.66666666666667, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.banking._value_, None, 0, 0, 0, 0, [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["chocolate"], RESOURCE_TO_IDX["spices"], RESOURCE_TO_IDX["truffles"], RESOURCE_TO_IDX["citrus"], RESOURCE_TO_IDX["olives"], RESOURCE_TO_IDX["coconut"]], False, False, [], 1, False

    refinery = 132, 0, 0.66, False, False, False, False, False, False, False, 133.33333333333334, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.biology._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["oil"]], False, False, [],  1, False

    grand_stele = 133, 0, 0.0, False, False, False, False, False, False, False, 83.33333333333333, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.theology._value_, None, 0, None, 0, 0, [0, 0, 0, 4, 0, 4, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 8, False

    panama = 134, 0, 1.0, True, False, False, False, False, False, False, 706.6666666666666, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400, False, False, 0, 0, Technologies.railroad._value_, "SPECIALIST_MERCHANT", 0, 0, 2, 0, [0, 0, 3, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, True

    artist_house = 135, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, None, None, 0, 0, [0, 0, 0, 2, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    writer_house = 136, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, None, None, 0, 0, [0, 0, 0, 1, 0, 3, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    music_house = 137, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 150, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, "SPECIALIST_MUSICIAN", 1, None, 0, 0, [0, 0, 0, 2, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, True, [], 4, False

    lake_wonder = 138, 0, 1.0, False, False, True, False, False, False, False, 200.0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.civil_service._value_, "SPECIALIST_ENGINEER", 0, 0, 1, 0, [0, 0, 0, 4, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [],  True, False, [], 9, False
    
    # SPACE SHIP PARTS
    apollo_program = 139, 0, 0.0, False, False, False, False, False, False, False, 1005, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.rocketry._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False

    booster_1 = 140, 0, 0.0, False, False, False, False, False, False, False, 1005, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.advanced_ballistics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False

    booster_2 = 141, 0, 0.0, False, False, False, False, False, False, False, 1005, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.advanced_ballistics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False

    booster_3 = 142, 0, 0.0, False, False, False, False, False, False, False, 1005, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.advanced_ballistics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False
    
    engine = 143, 0, 0.0, False, False, False, False, False, False, False, 502.5, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.particle_physics._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False
    
    cockpit = 144, 0, 0.0, False, False, False, False, False, False, False, 502.5, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.satellites._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False
    
    stasis_chamber = 145, 0, 0.0, False, False, False, False, False, False, False, 502.5, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.nanotechnology._value_, None, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", ["apollo_program"], [RESOURCE_TO_IDX["aluminium"]], False, True, [], 10, False

    # Fine Arts buildings (conservatory is number 122)
    gallery = 146, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 200, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 2, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 4, False

    scriptorium = 147, 0, 1.0, False, False, False, False, False, False, False, -0.6666666666666666, 200, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, Technologies.agriculture._value_, None, 0, 0, 0, 0, [0, 0, 0, 2, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], "None", [], [], False, False, [], 4, False
    

    def __new__(cls, value: int, gold_maintenance: int, conquest_prob: float, water: bool, river: bool, freshwater: bool, mountain, nearby_mountain_required, hill, flat, cost, faith_cost, holy_city, free_techs, free_policies, free_great_people, defense, religious_pressure, extra_spies, trade_recipient_bonus, trade_target_bonus, trade_num_bonus, trade_land_distance_mod, trade_sea_distance_mod, trade_land_gold_mod, trade_sea_gold_mod, trade_allows_food, trade_allows_production, tourism_landmark_pct, tourism_gw_mod, prereq_tech, specialist_type, specialist_count, gw_count, great_people_rate_change, mountain_science_yield, yield_city_add, yield_city_pct, yield_pop_pct, gw_type, prereq_building, resource_prereq, world_wonder, nat_wonder, policy_prereq, building_type, coastal_prereq):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.gold_maintenance = gold_maintenance
        obj.conquest_prob = conquest_prob
        obj.water = water
        obj.river = river
        obj.freshwater = freshwater
        obj.mountain = mountain
        obj.nearby_mountain_required = nearby_mountain_required
        obj.hill = hill
        obj.flat = flat
        obj.cost = cost
        obj.faith_cost = faith_cost
        obj.holy_city = holy_city
        obj.free_techs = free_techs
        obj.free_policies = free_policies
        obj.free_great_people = free_great_people
        obj.defense = defense
        obj.religious_pressure = religious_pressure
        obj.extra_spies = extra_spies
        obj.trade_recipient_bonus = trade_recipient_bonus
        obj.trade_target_bonus = trade_target_bonus
        obj.trade_num_bonus = trade_num_bonus
        obj.trade_land_distance_mod = trade_land_distance_mod
        obj.trade_sea_distance_mod = trade_sea_distance_mod
        obj.trade_land_gold_mod = trade_land_gold_mod
        obj.trade_sea_gold_mod = trade_sea_gold_mod
        obj.trade_allows_food = trade_allows_food
        obj.trade_allows_production = trade_allows_production
        obj.tourism_landmark_pct = tourism_landmark_pct
        obj.tourism_gw_mod = tourism_gw_mod
        obj.prereq_tech = prereq_tech
        obj.specialist_type = specialist_type
        obj.specialist_count = specialist_count
        obj.gw_count = gw_count
        obj.great_people_rate_change = great_people_rate_change
        obj.mountain_science_yield = mountain_science_yield
        obj.yield_city_add = yield_city_add
        obj.yield_city_pct = yield_city_pct
        obj.yield_pop_pct = yield_pop_pct
        obj.gw_type = gw_type
        obj.prereq_building = prereq_building
        obj.resource_prereq = resource_prereq
        obj.world_wonder = world_wonder
        obj.nat_wonder = nat_wonder
        obj.prereq_pol = policy_prereq
        obj.building_type = building_type
        obj.coastal_prereq = coastal_prereq
        return obj


"""
Now we can add individual building functions. Some buildings will effect the yields of certain types of tiles (e.g., resources, mountains.) Other buildings increase 
things like the number of trade routes the civ can maintain. 

To make these functions general (i.e., can be used within a jax.lax.switch statement), they need to have the same signature.

These functions will be called upon building completion, which happens at the beginning of the turn.
All of these functions are called from within a single-game context. In primitives.GameState.step_cities()
"""

@dataclass
class GameStateMinimal:
    player_cities: jnp.ndarray
    idx_to_hex_rowcol: jnp.ndarray
    landmask_map: jnp.ndarray
    all_resource_map: jnp.ndarray  
    visible_resources_map_players: jnp.ndarray
    lake_map: jnp.ndarray
    feature_map: jnp.ndarray
    edge_river_map: jnp.ndarray
    terrain_map: jnp.ndarray
    elevation_map: jnp.ndarray
    all_resource_type_map: jnp.ndarray

@dataclass
class CitiesMinimal:
    """Minimal cities dataclass containing only fields that can be updated by switch functions"""
    additional_yield_map: jnp.ndarray
    building_yields: jnp.ndarray
    gw_slots: jnp.ndarray
    growth_carryover: jnp.ndarray
    bldg_accel: jnp.ndarray
    specialist_slots: jnp.ndarray
    bldg_maintenance: jnp.ndarray
    unit_xp_add: jnp.ndarray
    can_trade_food: jnp.ndarray
    can_trade_prod: jnp.ndarray
    citywide_yield_accel: jnp.ndarray
    defense: jnp.ndarray
    trade_gold_add_owner: jnp.ndarray
    trade_land_dist_mod: jnp.ndarray
    great_person_accel: jnp.ndarray
    great_person_points: jnp.ndarray
    mounted_accel: jnp.ndarray
    land_unit_accel: jnp.ndarray
    trade_sea_dist_mod: jnp.ndarray
    can_city_connect_over_water: jnp.ndarray
    tech_steal_reduce_accel: jnp.ndarray
    sea_unit_accel: jnp.ndarray
    gw_tourism_accel: jnp.ndarray
    culture_to_tourism: jnp.ndarray
    air_unit_capacity: jnp.ndarray
    spaceship_prod_accel: jnp.ndarray
    trade_gold_add_dest: jnp.ndarray
    naval_movement_add: jnp.ndarray
    naval_sight_add: jnp.ndarray
    city_connection_gold_accel: jnp.ndarray
    armored_accel: jnp.ndarray
    ranged_accel: jnp.ndarray
    ranged_xp_add: jnp.ndarray
    wonder_accel: jnp.ndarray
    potential_owned_rowcols: jnp.ndarray
    ownership_map: jnp.ndarray
    city_rowcols: jnp.ndarray
    buildings_owned: jnp.ndarray
    population: jnp.ndarray
    religion_info: jnp.ndarray

# Pre-compute field names at module level (compile-time constants)
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


def apply_minimal_update(full_game, minimal_update: CitiesMinimal):
    """Apply minimal cities update to full game state using generic set_from_subset."""
    
    # Update cities fields
    updated_cities = set_from_subset(full_game.player_cities, minimal_update, _CITIES_MINIMAL_FIELDS)
    
    # Update game state with new cities
    updated_game = replace(full_game, player_cities=updated_cities)
    
    return updated_game


def create_minimal_update(game,
                          idx_to_hex_rowcol=None, landmask_map=None, all_resource_map=None,
                          ownership_map=None, visible_resources_map_players=None, lake_map=None,
                          feature_map=None, edge_river_map=None, terrain_map=None, elevation_map=None,
                          all_resource_type_map=None,
                        religion_info=None,
                         additional_yield_map=None, building_yields=None, gw_slots=None,
                         growth_carryover=None, bldg_accel=None, specialist_slots=None,
                         bldg_maintenance=None, unit_xp_add=None, can_trade_food=None,
                         can_trade_prod=None, citywide_yield_accel=None, defense=None,
                         trade_gold_add_owner=None, trade_land_dist_mod=None,
                         great_person_accel=None, great_person_points=None, mounted_accel=None, land_unit_accel=None,
                         trade_sea_dist_mod=None, can_city_connect_over_water=None, 
                         tech_steal_reduce_accel=None, sea_unit_accel=None, 
                         gw_tourism_accel=None, culture_to_tourism=None, air_unit_capacity=None,
                         spaceship_prod_accel=None, trade_gold_add_dest=None, 
                         naval_movement_add=None, naval_sight_add=None, 
                         city_connection_gold_accel=None, armored_accel=None, 
                         ranged_accel=None, ranged_xp_add=None, wonder_accel=None, potential_owned_rowcols=None,
                          city_rowcols=None, buildings_owned=None, population=None) -> GameStateMinimal:
    """
    Create a CitiesMinimal object. If an argument is None, use the value from game.player_cities.
    This ensures only explicitly updated fields are changed, others preserve original values.
    """
    _cities_min = CitiesMinimal(
        additional_yield_map=additional_yield_map if additional_yield_map is not None else game.player_cities.additional_yield_map,
        building_yields=building_yields if building_yields is not None else game.player_cities.building_yields,
        gw_slots=gw_slots if gw_slots is not None else game.player_cities.gw_slots,
        growth_carryover=growth_carryover if growth_carryover is not None else game.player_cities.growth_carryover,
        bldg_accel=bldg_accel if bldg_accel is not None else game.player_cities.bldg_accel,
        specialist_slots=specialist_slots if specialist_slots is not None else game.player_cities.specialist_slots,
        bldg_maintenance=bldg_maintenance if bldg_maintenance is not None else game.player_cities.bldg_maintenance,
        unit_xp_add=unit_xp_add if unit_xp_add is not None else game.player_cities.unit_xp_add,
        can_trade_food=can_trade_food if can_trade_food is not None else game.player_cities.can_trade_food,
        can_trade_prod=can_trade_prod if can_trade_prod is not None else game.player_cities.can_trade_prod,
        citywide_yield_accel=citywide_yield_accel if citywide_yield_accel is not None else game.player_cities.citywide_yield_accel,
        defense=defense if defense is not None else game.player_cities.defense,
        trade_gold_add_owner=trade_gold_add_owner if trade_gold_add_owner is not None else game.player_cities.trade_gold_add_owner,
        trade_land_dist_mod=trade_land_dist_mod if trade_land_dist_mod is not None else game.player_cities.trade_land_dist_mod,
        great_person_accel=great_person_accel if great_person_accel is not None else game.player_cities.great_person_accel,
        great_person_points=great_person_points if great_person_points is not None else game.player_cities.great_person_points,
        mounted_accel=mounted_accel if mounted_accel is not None else game.player_cities.mounted_accel,
        land_unit_accel=land_unit_accel if land_unit_accel is not None else game.player_cities.land_unit_accel,
        trade_sea_dist_mod=trade_sea_dist_mod if trade_sea_dist_mod is not None else game.player_cities.trade_sea_dist_mod,
        can_city_connect_over_water=can_city_connect_over_water if can_city_connect_over_water is not None else game.player_cities.can_city_connect_over_water,
        tech_steal_reduce_accel=tech_steal_reduce_accel if tech_steal_reduce_accel is not None else game.player_cities.tech_steal_reduce_accel,
        sea_unit_accel=sea_unit_accel if sea_unit_accel is not None else game.player_cities.sea_unit_accel,
        gw_tourism_accel=gw_tourism_accel if gw_tourism_accel is not None else game.player_cities.gw_tourism_accel,
        culture_to_tourism=culture_to_tourism if culture_to_tourism is not None else game.player_cities.culture_to_tourism,
        air_unit_capacity=air_unit_capacity if air_unit_capacity is not None else game.player_cities.air_unit_capacity,
        spaceship_prod_accel=spaceship_prod_accel if spaceship_prod_accel is not None else game.player_cities.spaceship_prod_accel,
        trade_gold_add_dest=trade_gold_add_dest if trade_gold_add_dest is not None else game.player_cities.trade_gold_add_dest,
        naval_movement_add=naval_movement_add if naval_movement_add is not None else game.player_cities.naval_movement_add,
        naval_sight_add=naval_sight_add if naval_sight_add is not None else game.player_cities.naval_sight_add,
        city_connection_gold_accel=city_connection_gold_accel if city_connection_gold_accel is not None else game.player_cities.city_connection_gold_accel,
        armored_accel=armored_accel if armored_accel is not None else game.player_cities.armored_accel,
        ranged_accel=ranged_accel if ranged_accel is not None else game.player_cities.ranged_accel,
        ranged_xp_add=ranged_xp_add if ranged_xp_add is not None else game.player_cities.ranged_xp_add,
        wonder_accel=wonder_accel if wonder_accel is not None else game.player_cities.wonder_accel,
        potential_owned_rowcols=potential_owned_rowcols if potential_owned_rowcols is not None else game.player_cities.potential_owned_rowcols,
        ownership_map=ownership_map if ownership_map is not None else game.player_cities.ownership_map,
        city_rowcols=city_rowcols if city_rowcols is not None else game.player_cities.city_rowcols,
        buildings_owned=buildings_owned if buildings_owned is not None else game.player_cities.buildings_owned,
        population=population if population is not None else game.player_cities.population,
        religion_info=religion_info if religion_info is not None else game.player_cities.religion_info,
    )

    return GameStateMinimal(
        _cities_min,
        idx_to_hex_rowcol=idx_to_hex_rowcol if idx_to_hex_rowcol is not None else game.idx_to_hex_rowcol,
        landmask_map=landmask_map if landmask_map is not None else game.landmask_map,
        all_resource_map=all_resource_map if all_resource_map is not None else game.all_resource_map,
        visible_resources_map_players=visible_resources_map_players if visible_resources_map_players is not None else game.visible_resources_map_players,
        lake_map=lake_map if lake_map is not None else game.lake_map,
        feature_map=feature_map if feature_map is not None else game.feature_map,
        edge_river_map=edge_river_map if edge_river_map is not None else game.edge_river_map,
        terrain_map=terrain_map if terrain_map is not None else game.terrain_map,
        elevation_map=elevation_map if elevation_map is not None else game.elevation_map,
        all_resource_type_map=all_resource_type_map if all_resource_type_map is not None else game.all_resource_type_map,
    )



def add_bldg_maintenance(game, city_int, player_id, amt):
    new_bldg_maintenance = game.player_cities.bldg_maintenance[player_id[0], city_int] + amt
    return game.replace(
        player_cities=game.player_cities.replace(
            bldg_maintenance=game.player_cities.bldg_maintenance.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_bldg_maintenance)
        )
    )

def add_bldg_yields(game, city_int, player_id, to_add):
    city_yields = game.player_cities.building_yields[player_id[0], city_int]
    new_city_yields = city_yields + to_add
    return game.replace(
        player_cities=game.player_cities.replace(
            building_yields=game.player_cities.building_yields.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_city_yields),
        )
    )

def add_gw_slots(game, city_int, player_id, gw_add):
    """[writing, art, music, artifact]"""
    new_gw = game.player_cities.gw_slots[player_id[0], city_int] + gw_add.astype(jnp.uint8)
    return game.replace(
        player_cities=game.player_cities.replace(
            gw_slots=game.player_cities.gw_slots.at[jnp.index_exp[player_id[0], city_int]].set(new_gw)
        )
    )

def add_specialist_slots(game, city_int, player_id, specialist_add):
    """[artist, musician, writer, engineer, merchant, scientist]"""
    new_gp = game.player_cities.specialist_slots[player_id[0], city_int] + specialist_add.astype(jnp.uint8)
    return game.replace(
        player_cities=game.player_cities.replace(
            specialist_slots=game.player_cities.specialist_slots.at[jnp.index_exp[player_id[0], city_int]].set(new_gp)
        )
    )

def add_yield_multipliers(game, city_int, player_id, to_add):
    new_yield_multipler = game.player_cities.citywide_yield_accel[player_id[0], city_int] + to_add
    return game.replace(
        player_cities=game.player_cities.replace(
            citywide_yield_accel=game.player_cities.citywide_yield_accel.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_yield_multipler)
        )
    )

def add_great_person_points(game, city_int, player_id, to_add):
    """[artist, musician, writer, engineer, merchant, scientist]"""
    new_gpps = game.player_cities.great_person_points.at[player_id[0], city_int].add(to_add)
    return game.replace(player_cities=game.player_cities.replace(great_person_points=new_gpps))

def add_building_indicator(buildings_owned, city_int, player_id, building_idx):
    return buildings_owned.at[jnp.index_exp[player_id[0], city_int, building_idx]].set(1)

def add_building_indicator_minimal(buildings_owned, city_int, player_id, building_idx):
    return buildings_owned[player_id[0], city_int].at[building_idx].set(1)

def add_tile_yields(game, player_id, city_int, bool_map_generator, to_add):
    """When we are adding values to tile yields, add to additional_yield_map

    * bool_map_generator(game, player_id, city_int) => (bool_map, game_map_rowcols, bool_city_center, game_map_rowcols_city_center)
        bool_map is (36,), which represents all tiles EXCEPT city center
        bool_city_center is (,)
        game_map_rowcols (36, 2)
        game_map_rowcols_city_center (2,)
    
    """
    bool_map, game_map_rowcols, bool_city_center, game_map_rowcols_city_center = bool_map_generator(game, player_id, city_int)

    # additional_yield_map (6, 42, 66, 7)
    _additional_yield_map = game.player_cities.additional_yield_map[player_id[0]]
    
    # (36,) => (36,7)
    city_ring_to_set = bool_map[:, None] * to_add[None] 
    city_center_to_set = bool_city_center * to_add

    _additional_yield_map = _additional_yield_map.at[game_map_rowcols[:, 0], game_map_rowcols[:, 1]].add(city_ring_to_set)
    _additional_yield_map = _additional_yield_map.at[
        game_map_rowcols_city_center[0], game_map_rowcols_city_center[1]
    ].add(city_center_to_set)

    return game.replace(player_cities=game.player_cities.replace(
        additional_yield_map=game.player_cities.additional_yield_map.at[player_id[0]].set(_additional_yield_map)
    ))

def no_change(game, city_int, player_id):
    return game

def _cathedral(game, city_int, player_id):
    """
    +5 culture
    +1 gw art slot
    """
    grabbed = jnp.array([0, 0, 0, 0, 5, 0, 0, 0])
    game = add_bldg_yields(game, city_int, player_id, grabbed)

    gw_add = jnp.array([0, 1, 0, 0])
    game = add_gw_slots(game, city_int, player_id, gw_add)
    return game
    

def _gurdwara(game, city_int, player_id):
    """
    Each religion in this city +2 science, +2 faith
    """
    n_rels = (game.player_cities.religion_info.religious_population[player_id[0], city_int] > 0).sum()
    to_add = jnp.array([0, 0, 0, 2, 0, 2, 0, 0]) * n_rels
    game = add_bldg_yields(game, city_int, player_id, to_add)
    return game 


def _mandir(game, city_int, player_id):
    """
    +1 prod, +1 food to all lux
    """

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        this_city_resources = game.all_resource_type_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == 1

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_type_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == 1

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center
    
    to_add = jnp.array([1, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _monastery(game, city_int, player_id):
    """
    +1 faith, +1 culture to incense and wine
    +2 faith, +2 culture to city
    """

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        incense = RESOURCE_TO_IDX["incense"]
        wine = RESOURCE_TO_IDX["wine"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == incense) | (this_city_resources == wine)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == incense) | (this_city_center_resources == wine)

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 2, 2, 0, 0, 0]))
    return game 

def _mosque(game, city_int, player_id):
    """
    +1 happiness, +1 culture, +3 faith to city
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 3, 1, 0, 1, 0]))
    return game

def _pagoda(game, city_int, player_id):
    """
    +1 happiness, +1 culture, +2 faith to city
    10% growth carried over
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 2, 1, 0, 1, 0]))
    new_growth_carryover = game.player_cities.growth_carryover[player_id[0], city_int] + 0.1
    return game.replace(player_cities=game.player_cities.replace(
        growth_carryover=game.player_cities.growth_carryover.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_growth_carryover)
    ))

def _synagogue(game, city_int, player_id):
    """
    +1 culture, +2 faith, +1 hammer to city
    +15% boost to building prod
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 1, 0, 2, 1, 0, 0, 0]))
    new_bldg_accel = game.player_cities.bldg_accel[player_id[0], city_int] + 0.15
    return game.replace(player_cities=game.player_cities.replace(
        bldg_accel=game.player_cities.bldg_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_bldg_accel)
    ))

def _vihara(game, city_int, player_id):
    """
    +1 happiness, +1 faith to city
    +15% boost to gold output
    +1 SPECIALIST_MERCHANT slot
    [artist, musician, writer, engineer, merchant, scientist]
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0, 1, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0.15, 0, 0, 0, 0, 0]))
    return game 

def _barracks(game, city_int, player_id):
    """
    +15 XP for all units
    +1 bldg_maintenance
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))

def _granary(game, city_int, player_id):
    """
    +2 food to city
    +1 food to wheat, banana, deer, bison
    bonus for food trade
    +1 bldg_maintenance
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([2, 0, 0, 0, 0, 0, 0, 0]))
    game = add_bldg_maintenance(game, city_int, player_id, 1)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        wheat = RESOURCE_TO_IDX["wheat"]
        banana = RESOURCE_TO_IDX["banana"]
        deer = RESOURCE_TO_IDX["deer"]
        bison = RESOURCE_TO_IDX["bison"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_res = (this_city_resources == wheat) | (this_city_resources == banana) | (this_city_resources == deer) | (this_city_resources == bison)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_res = (this_city_center_resources == wheat) | (this_city_center_resources == banana) | (this_city_center_resources == deer) | (this_city_center_resources == bison)

        return this_city_currently_owned * this_city_res, game_map_rowcols, this_city_center_res, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    return game.replace(player_cities=game.player_cities.replace(
        can_trade_food=game.player_cities.can_trade_food.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(1)
    ))


def _library(game, city_int, player_id):
    """
    +50% science per pop
    +1 bldg_maintenance
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0.5, 0, 0]))
    return game

def _monument(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    return game

def _palace(game, city_int, player_id):
    """
    +1 culture, +3 gold, +3 science, +3 prod
    +250 defense
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 3, 3, 0, 1, 3, 0, 0]))
    new_defense = game.player_cities.defense[player_id[0], city_int] + 250

    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _shrine(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +1 faith
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0, 0, 0]))
    return game


def _stone_works(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +1 happiness, +1 prod
    +1 prod to marble, obsidian, stone
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 1, 0, 0, 0, 0, 1, 0]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        marble = RESOURCE_TO_IDX["marble"]
        obsidian = RESOURCE_TO_IDX["obsidian"]
        stone = RESOURCE_TO_IDX["stone"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == marble) | (this_city_resources == obsidian) | (this_city_resources == stone)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == marble) | (this_city_center_resources == obsidian) | (this_city_center_resources == stone)

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _walls(game, city_int, player_id):
    """
    +500 defense
    +50 hp
    """
    new_defense = game.player_cities.defense[player_id[0], city_int] + 500

    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _watermill(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 food, +1 prod to city
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([2, 1, 0, 0, 0, 0, 0, 0]))
    return game


def _amphitheater(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture for city
    +1 gw slot writing
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([1, 0, 0, 0]))
    return game

def _aqueduct(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +40% food carryover
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_growth_carryover = game.player_cities.growth_carryover[player_id[0], city_int] + 0.4
    return game.replace(
        player_cities=game.player_cities.replace(
            growth_carryover=game.player_cities.growth_carryover.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_growth_carryover)
        )
    )

def _caravansary(game, city_int, player_id):
    """
    +2 gold to city
    +2 gold from trade-routes that go to another civ (not cs)
    +50% range to land trade routes
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 2, 0, 0, 0, 0, 0]))
    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 2
    new_trade_land_dist_mod = game.player_cities.trade_land_dist_mod[player_id[0], city_int] + 0.5
    return game.replace(
        player_cities=game.player_cities.replace(
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            trade_land_dist_mod=game.player_cities.trade_land_dist_mod.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_land_dist_mod)
        )
    )

def _colosseum(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 happiness to city
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game

def _courthouse(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 happiness to city
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game

def _garden(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +20% great people gen
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_gp_mult = game.player_cities.great_person_accel[player_id[0], city_int] + 0.2
    return game.replace(
        player_cities=game.player_cities.replace(
            great_person_accel=game.player_cities.great_person_accel.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_gp_mult)
        )
    )

def _lighthouse(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +1 food for ocean
    +1 prod for all sea resources
    +1 food for fish
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        this_city_ocean = game.landmask_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_ocean = game.landmask_map[this_city_center[0], this_city_center[1]] == 0

        return this_city_currently_owned * this_city_ocean, game_map_rowcols, this_city_center_ocean, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        this_city_ocean = game.landmask_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0
        this_city_resources = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        
        this_city_res = this_city_ocean & this_city_resources

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_ocean = game.landmask_map[this_city_center[0], this_city_center[1]] == 0
        this_city_resources = game.visible_resources_map_players[player_id[0], this_city_center[0], this_city_center[1]] > 0
        
        this_city_center_res = this_city_ocean & this_city_resources
        return this_city_currently_owned * this_city_res, game_map_rowcols, this_city_center_res, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        fish = RESOURCE_TO_IDX["fish"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_res = this_city_resources == fish

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_res = this_city_center_resources == fish

        return this_city_currently_owned * this_city_res, game_map_rowcols, this_city_center_res, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _market(game, city_int, player_id):
    """
    +1 gold for city
    +25% gold output
    +1 SPECIALIST_MERCHANT slot
    +1 gold for traderoutes (both sender and owner)
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 1, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0.25, 0, 0, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0,  0, 1, 0]))
    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 1
    new_trade_gold_add_dest = game.player_cities.trade_gold_add_dest[player_id[0], city_int] + 1
    return game.replace(
        player_cities=game.player_cities.replace(
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            trade_gold_add_dest=game.player_cities.trade_gold_add_dest.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_dest),
        )
    )

def _mint(game, city_int, player_id):
    """
    +2 gold to city
    +2 gold to gold, silver, and copper
    """

    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 2, 0, 0, 0, 0, 0]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        gold = RESOURCE_TO_IDX["gold"]
        silver = RESOURCE_TO_IDX["silver"]
        copper = RESOURCE_TO_IDX["copper"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == gold) | (this_city_resources == silver) | (this_city_resources == copper)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == gold) | (this_city_center_resources == silver) | (this_city_center_resources == copper)

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game 


def _stable(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +15% prod mounted units
    +1 prod to horses, sheep, cattle
    +1 food maize
    """

    game = add_bldg_maintenance(game, city_int, player_id, 1)
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        horses = RESOURCE_TO_IDX["horses"]
        sheep = RESOURCE_TO_IDX["sheep"]
        cattle = RESOURCE_TO_IDX["cow"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == horses) | (this_city_resources == sheep) | (this_city_resources == cattle)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == horses) | (this_city_center_resources == sheep) | (this_city_center_resources == cattle)

        can_see_resource_bool = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            player_id[0], this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * this_city_lux * can_see_resource_bool, game_map_rowcols, this_city_center_lux * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        maize = RESOURCE_TO_IDX["maize"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == maize

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == maize

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    new_mounted_accel = game.player_cities.mounted_accel[player_id[0], city_int] + 0.15 

    return game.replace(player_cities=game.player_cities.replace(
        mounted_accel=game.player_cities.mounted_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_mounted_accel)
    ))

def _temple(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 faith
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 2, 0, 0, 0, 0]))
    return game

def _writers_guild(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 SPECIALIST_WRITER
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 2, 0, 0, 0]))
    return game

def _armory(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +15 xp for all units
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))

def _artists_guild(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 SPECIALIST_ARTIST
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([2, 0, 0, 0, 0, 0]))
    return game

def _brewery(game, city_int, player_id):
    """
    +2 gold
    +2 gold to wine, coffee, tea, sugar, salt
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 2, 0, 0, 0, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        wine = RESOURCE_TO_IDX["wine"]
        coffee = RESOURCE_TO_IDX["coffee"]
        tea = RESOURCE_TO_IDX["tea"]
        sugar = RESOURCE_TO_IDX["sugar"]
        salt = RESOURCE_TO_IDX["salt"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == wine) | (this_city_resources == coffee) | (this_city_resources == tea) | (this_city_resources == sugar) | (this_city_resources == salt)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == wine) | (this_city_center_resources == coffee) | (this_city_center_resources == tea) | (this_city_center_resources == sugar) | (this_city_center_resources == salt)

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _castle(game, city_int, player_id):
    """
    +700 defense
    +25 hp
    """
    new_defense = game.player_cities.defense[player_id[0], city_int] + 700

    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _censer_maker(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture
    +1 culture to incense, tobacco, perfume
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        tobacco = RESOURCE_TO_IDX["tobacco"]
        incense = RESOURCE_TO_IDX["incense"]
        perfume = RESOURCE_TO_IDX["perfume"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == tobacco) | (this_city_resources == incense) | (this_city_resources == perfume)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == tobacco) | (this_city_center_resources == incense) | (this_city_center_resources == perfume) 

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _forge(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +15% prod to land units
    +1 prod iron
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        iron = RESOURCE_TO_IDX["iron"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == iron

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == iron

        can_see_resource_bool = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            player_id[0], this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * this_city_lux * can_see_resource_bool, game_map_rowcols, this_city_center_lux * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    new_land_unit_accel = game.player_cities.land_unit_accel[player_id[0], city_int] + 0.15

    return game.replace(player_cities=game.player_cities.replace(
        land_unit_accel=game.player_cities.land_unit_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_land_unit_accel)
    ))


def _gemcutter(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture
    +1 culture to jade, gems, lapis, amber
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        jade = RESOURCE_TO_IDX["jade"]
        gems = RESOURCE_TO_IDX["gems"]
        lapis = RESOURCE_TO_IDX["lapis"]
        amber = RESOURCE_TO_IDX["amber"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == jade) | (this_city_resources == gems) | (this_city_resources == lapis) | (this_city_resources == amber)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == jade) | (this_city_center_resources == gems) | (this_city_center_resources == lapis) | (this_city_center_resources == amber) 

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _harbor(game, city_int, player_id):
    """
    +2 bldg_maintenance
    forms city connect with cap over water
    +25% trade route range (nerfed -- only applied to sea in the game, 
    but we have simplified away sea interactions. Ergo, this applies to all routes.)
    +1 gold trade with another civ
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    new_trade_sea_dist = game.player_cities.trade_sea_dist_mod[player_id[0], city_int] + 0.25
    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 1
    new_connect_over_water = game.player_cities.can_city_connect_over_water.at[player_id[0], city_int].set(1)

    return game.replace(
        player_cities=game.player_cities.replace(
            trade_sea_dist_mod=game.player_cities.trade_sea_dist_mod.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_sea_dist),
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            can_city_connect_over_water=new_connect_over_water
        )
    )

def _university(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +33% science mod
    +2 SPECIALIST_SCIENTIST 
    +2 science on jungle
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0.33, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 2]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        jungle = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == JUNGLE_IDX

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        center_jungle = game.feature_map[this_city_center[0], this_city_center[1]] == JUNGLE_IDX

        return this_city_currently_owned * jungle, game_map_rowcols, center_jungle, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 0, 2, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _workshop(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 prod
    +10% prod
    Can send production over trade routes
    +1 SPECIALIST_ENGINEER
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 2, 0, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0.1, 0, 0, 0, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game.replace(player_cities=game.player_cities.replace(
        can_trade_prod=game.player_cities.can_trade_prod.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(1)
    ))

def _arsenal(game, city_int, player_id):
    """
    +900 defense
    +25 hp
    """
    new_defense = game.player_cities.defense[player_id[0], city_int] + 900

    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _bank(game, city_int, player_id):
    """
    +2 gold for city
    +25% gold output
    +1 SPECIALIST_MERCHANT slot
    +1 gold for traderoutes (both sender and owner)
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 2, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0.25, 0, 0, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))

    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 1
    new_trade_gold_add_dest = game.player_cities.trade_gold_add_dest[player_id[0], city_int] + 1

    return game.replace(
        player_cities=game.player_cities.replace(
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            trade_gold_add_dest=game.player_cities.trade_gold_add_dest.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_dest)
        )
    )

def _constabulary(game, city_int, player_id):
    """
    -25% steal rate
    """
    new_reduce_accel = game.player_cities.tech_steal_reduce_accel[player_id[0], city_int] - 0.25
    return game.replace(player_cities=game.player_cities.replace(
        tech_steal_reduce_accel=game.player_cities.tech_steal_reduce_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_reduce_accel)
    ))


def _grocer(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +1 food
    +1 food to truffles, citrus, olives, coconut, chocolate, spices
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([1, 0, 0, 0, 0, 0, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        truffles = RESOURCE_TO_IDX["truffles"]
        citrus = RESOURCE_TO_IDX["citrus"]
        olives = RESOURCE_TO_IDX["olives"]
        coconut = RESOURCE_TO_IDX["coconut"]
        chocolate = RESOURCE_TO_IDX["chocolate"]
        spices = RESOURCE_TO_IDX["spices"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == truffles) | (this_city_resources == citrus) | (this_city_resources == olives) | (this_city_resources == coconut) | (this_city_resources == chocolate) | (this_city_resources == spices)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == truffles) | (this_city_center_resources == citrus) | (this_city_center_resources == olives) | (this_city_center_resources == coconut) | (this_city_center_resources == chocolate) | (this_city_center_resources == spices)

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _musicians_guild(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 SPECIALIST_MUSICIAN
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 2, 0, 0, 0, 0]))
    return game

def _observatory(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +4 science
    +1 science on tundra
    +4 science mountains
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 4, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        jungle = game.terrain_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == TUNDRA_IDX

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        center_jungle = game.terrain_map[this_city_center[0], this_city_center[1]] == TUNDRA_IDX

        return this_city_currently_owned * jungle, game_map_rowcols, center_jungle, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 0, 1, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        jungle = game.elevation_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == MOUNTAIN_IDX

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        center_jungle = game.elevation_map[this_city_center[0], this_city_center[1]] == MOUNTAIN_IDX

        return this_city_currently_owned * jungle, game_map_rowcols, center_jungle, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 0, 4, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _opera_house(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture
    +1 gw music
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 0, 1, 0]))
    return game

def _seaport(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +1 prod, +1 gold from sea resources
    +15% prod on sea units
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        this_city_ocean = game.landmask_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0
        this_city_res = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        this_city_bool = this_city_ocean & this_city_res

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_ocean = game.landmask_map[this_city_center[0], this_city_center[1]] == 0
        this_city_center_res = game.all_resource_map[this_city_center[0], this_city_center[1]] > 0
        this_city_center_bool = this_city_center_ocean & this_city_center_res
        
        can_see_resource_bool = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            player_id[0], this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * this_city_bool * can_see_resource_bool, game_map_rowcols, this_city_center_bool * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 1, 1, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    
    # Removed sea units from game. By moving this bonus to land units, coastal cities become 
    # more valuable.
    new_sea_unit_accel = game.player_cities.land_unit_accel[player_id[0], city_int] + 0.15

    return game.replace(player_cities=game.player_cities.replace(
        land_unit_accel=game.player_cities.land_unit_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_sea_unit_accel)
    ))



def _textile_mill(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +1 prod
    +2 prod to dyes, furs, silk, cotton
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 1, 0, 0, 0, 0, 0, 0]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        dyes = RESOURCE_TO_IDX["dyes"]
        furs = RESOURCE_TO_IDX["furs"]
        silk = RESOURCE_TO_IDX["silk"]
        cotton = RESOURCE_TO_IDX["cotton"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = (this_city_resources == dyes) | (this_city_resources == furs) | (this_city_resources == silk) | (this_city_resources == cotton)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = (this_city_center_resources == dyes) | (this_city_center_resources == furs) | (this_city_center_resources == silk) | (this_city_center_resources == cotton) 

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 2, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _windmill(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 prod
    +10% prod on blds
    +1 SPECIALIST_ENGINEER
    +1 prod on lake, oasis
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 2, 0, 0, 0, 0, 0, 0]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        lakes = game.lake_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 1
        oasis = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == OASIS_IDX
        this_city_bool = lakes | oasis

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        
        lakes = game.lake_map[this_city_center[0], this_city_center[1]] == 1
        oasis = game.feature_map[this_city_center[0], this_city_center[1]] == OASIS_IDX
        this_city_center_bool = lakes | oasis

        return this_city_currently_owned * this_city_bool, game_map_rowcols, this_city_center_bool, this_city_center
    
    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))

    new_bldg_accel = game.player_cities.bldg_accel[player_id[0], city_int] + 0.1

    return game.replace(player_cities=game.player_cities.replace(
        bldg_accel=game.player_cities.bldg_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_bldg_accel)
    ))


def _zoo(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 happiness to city
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game

def _circus(game, city_int, player_id):
    """
    +2 happiness to city
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game

def _factory(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +4 prod
    +10% prod
    +2 SPECIALIST_ENGINEER
    +2 gold rubber
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 4, 0, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0.1, 0, 0, 0, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 2, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        rubber = RESOURCE_TO_IDX["rubber"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == rubber 

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == rubber

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center

    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _hospital(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +5 food
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([5, 0, 0, 0, 0, 0, 0, 0]))
    return game

def _hydroplant(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +1 prod on all river tiles
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        river = game.edge_river_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]].sum(-1) > 0

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        river_center = game.edge_river_map[this_city_center[0], this_city_center[1]].sum(-1) > 0

        return this_city_currently_owned * river, game_map_rowcols, river_center, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    return game


def _military_academy(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +15 XP units
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))


def _museum(game, city_int, player_id):
    """
    +1 bldg_maintenance
    +2 culture
    +2 gw art
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 2, 0, 0]))
    return game

def _oil_refinery(game, city_int, player_id):
    """
    +3 gold, +3 prod on oil
    """
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        oil = RESOURCE_TO_IDX["oil"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == oil

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == oil
        
        can_see_resource_bool = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            player_id[0], this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * this_city_lux * can_see_resource_bool, game_map_rowcols, this_city_center_lux * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 3, 3, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _police_station(game, city_int, player_id):
    """
    -25% spy rate
    """
    new_tech_steal_reduce_accel = game.player_cities.tech_steal_reduce_accel[player_id[0], city_int] - 0.25
    return game.replace(player_cities=game.player_cities.replace(
        tech_steal_reduce_accel=game.player_cities.tech_steal_reduce_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_tech_steal_reduce_accel)
    ))


def _public_school(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +50% science mod
    +1 SPECIALIST_SCIENTIST
    +1
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 3, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0.5, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    return game


def _stock_exchange(game, city_int, player_id):
    """
    +3 gold for city
    +25% gold output
    +2 SPECIALIST_MERCHANT slot
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 3, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0.25, 0, 0, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0]))
    return game


def _broadcast_tower(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +2 culture
    +1 gw music
    +15% culture
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 0, 1, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0.15, 0, 0, 0]))
    return game

def _hotel(game, city_int, player_id):
    """
    50% culture from world/natural wonders, improvements added to tourism
    +12.5% tourism from gw
    +1 culture from lake and oasis
    """
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        lakes = game.lake_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 1
        oasis = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == OASIS_IDX
        this_city_bool = lakes | oasis

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        
        lakes = game.lake_map[this_city_center[0], this_city_center[1]] == 1
        oasis = game.feature_map[this_city_center[0], this_city_center[1]] == OASIS_IDX
        this_city_center_bool = lakes | oasis

        return this_city_currently_owned * this_city_bool, game_map_rowcols, this_city_center_bool, this_city_center
    
    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)

    new_gw_tourism_accel = game.player_cities.gw_tourism_accel[player_id[0], city_int] + 0.125
    new_culture_to_tourism = game.player_cities.culture_to_tourism[player_id[0], city_int] + 0.5

    return game.replace(player_cities=game.player_cities.replace(
        gw_tourism_accel=game.player_cities.gw_tourism_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_gw_tourism_accel),
        culture_to_tourism=game.player_cities.culture_to_tourism.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_culture_to_tourism)
    ))

def _medical_lab(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +25% food carryover
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    new_growth_carryover = game.player_cities.growth_carryover[player_id[0], city_int] + 0.25

    return game.replace(
        player_cities=game.player_cities.replace(
            growth_carryover=game.player_cities.growth_carryover.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_growth_carryover)
        )
    )


def _military_base(game, city_int, player_id):
    """
    +1200 defense
    +25 hp
    """
    new_defense = game.player_cities.defense[player_id[0], city_int] + 1200

    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _research_lab(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +4 science
    +50% science
    +1 SPECIALIST_SCIENTIST
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 4, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0.5, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    return game


def _stadium(game, city_int, player_id):
    """
    +2 bldg_maintenance
    +2 happiness to city
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game

def _airport(game, city_int, player_id):
    """
    +5 bldg_maintenance
    50% of culture from world/natural wonders, improvements to tourism
    +12.5% tourism from gw
    6=>10 air unit capacity
    """
    game = add_bldg_maintenance(game, city_int, player_id, 5)
    new_gw_tourism_accel = game.player_cities.gw_tourism_accel[player_id[0], city_int] + 0.125
    new_culture_to_tourism = game.player_cities.culture_to_tourism[player_id[0], city_int] + 0.5

    # We set this to 4, as 6 is added via the reset in "..._to_appropriate_fields"
    new_air_unit_capacity = game.player_cities.air_unit_capacity.at[jnp.index_exp[player_id[0], city_int]].set(4)
    return game.replace(player_cities=game.player_cities.replace(
        gw_tourism_accel=game.player_cities.gw_tourism_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_gw_tourism_accel),
        culture_to_tourism=game.player_cities.culture_to_tourism.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_culture_to_tourism),
        air_unit_capacity=new_air_unit_capacity
    ))

def _nuclear_plant(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +5 prod
    +15% prod
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 5, 0, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0.15, 0, 0, 0, 0, 0, 0]))
    return game

def _recycling_plant(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +3 prod on aluminium
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        iron = RESOURCE_TO_IDX["aluminium"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == iron

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == iron

        can_see_resource_bool = game.visible_resources_map_players[player_id[0], game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            player_id[0], this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * this_city_lux * can_see_resource_bool, game_map_rowcols, this_city_center_lux * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 3, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _solar_plant(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +5 prod
    +15% prod
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 5, 0, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0.15, 0, 0, 0, 0, 0, 0]))
    return game

def _bomb_shelter(game, city_int, player_id):
    """
    +1 bldg_maintenance
    We've removed nukes from the game, so instead we're upping the 
    defenses of a city if it has shelters.
    """
    game = add_bldg_maintenance(game, city_int, player_id, 1)
    new_defense = game.player_cities.defense[player_id[0], city_int] + 1000
    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )


def _spaceship_factory(game, city_int, player_id):
    """
    +3 bldg_maintenance
    +3 prod
    +50% prod for spaceship
    """
    game = add_bldg_maintenance(game, city_int, player_id, 3)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 3, 0, 0, 0, 0, 0, 0]))
    new_spaceship_prod_accel = game.player_cities.spaceship_prod_accel[player_id[0], city_int] + 0.5
    return game.replace(player_cities=game.player_cities.replace(
        spaceship_prod_accel=game.player_cities.spaceship_prod_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_spaceship_prod_accel)
    ))


def _heroic_epic(game, city_int, player_id):
    """
    +1 culture
    +1 gw writing
    +15 XP all non-air
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([1, 0, 0, 0]))
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))

def _national_college(game, city_int, player_id):
    """
    +3 science, +1 culture
    +50% science
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 3, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0.5, 0, 0]))
    return game


def _national_epic(game, city_int, player_id):
    """
    +1 culture
    +1 gw writing
    +25% great person
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([1, 0, 0, 0]))
    new_gp_mult = game.player_cities.great_person_accel[player_id[0], city_int] + 0.25
    return game.replace(
        player_cities=game.player_cities.replace(
            great_person_accel=game.player_cities.great_person_accel.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_gp_mult)
        )
    )


def _circus_maximus(game, city_int, player_id):
    """
    +1 culture, +5 happiness
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 5, 0]))
    return game

def _national_treasury(game, city_int, player_id):
    """
    This is the East India Company
    +4 gold
    +10% gold
    +1 trade route
    +4 gold for trade route owner, +2 for destination
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 4, 0, 0, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0.1, 0, 0, 0, 0, 0]))

    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 4
    new_trade_gold_add_dest = game.player_cities.trade_gold_add_dest[player_id[0], city_int] + 2

    return game.replace(
        player_cities=game.player_cities.replace(
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            trade_gold_add_dest=game.player_cities.trade_gold_add_dest.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_dest)
        )
    )


def _ironworks(game, city_int, player_id):
    """
    +8 prod
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 8, 0, 0, 0, 0, 0, 0]))
    return game

def _oxford_university(game, city_int, player_id):
    """
    free tech
    """
    # If we add one to the outer GameState 
    return game

def _hermitage(game, city_int, player_id):
    """
    +2 culture
    +50% culture
    +3 gw art
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0.5, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 3, 0, 0]))
    return game

def _great_lighthouse(game, city_int, player_id):
    """
    +1 culture
    +1 SPECIALIST_MERCHANT points
    +1 gold all water
    +1 movement +1 sight naval units
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        this_city_ocean = game.landmask_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0
        this_city_lakes = game.lake_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 1
        this_city_water = this_city_ocean | this_city_lakes

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_ocean = game.landmask_map[this_city_center[0], this_city_center[1]] == 0
        this_city_center_lakes = game.lake_map[this_city_center[0], this_city_center[1]] == 1
        this_city_center_water = this_city_center_ocean | this_city_center_lakes

        return this_city_currently_owned * this_city_water, game_map_rowcols, this_city_center_water, this_city_center

    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))

    new_naval_movement_add = game.player_cities.naval_movement_add[player_id[0], city_int] + 1
    new_naval_sight_add = game.player_cities.naval_sight_add[player_id[0], city_int] + 1
    return game.replace(player_cities=game.player_cities.replace(
            naval_movement_add=game.player_cities.naval_movement_add.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_naval_movement_add),
            naval_sight_add=game.player_cities.naval_sight_add.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_naval_sight_add)
        ))

def _stonehenge(game, city_int, player_id):
    """
    +1 culture, +6 faith
    +1 SPECIALIST_ENGINEER points
    -25% culture & gold costs for new tiles in all cities
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 6, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game

def _great_library(game, city_int, player_id):
    """
    +1 culture, +3 science
    +1 SPECIALIST_SCIENTIST point
    +2 gw writing
    +1 free tech
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 3, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([2, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    return game

def _pyramid(game, city_int, player_id):
    """
    Pyramids (the wonder)
    +1 culture
    +1 SPECIALIST_ENGINEER points
    Tile improvement speed -25%
    +2 free workers
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game


def _colossus(game, city_int, player_id):
    """
    +1 culture, +5 gold
    +1 SPECIALIST_MERCHANT points
    +1 trade route
    +1 free cargo ship
    +2 gold for traderoute owner, +1 gold for destination
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 5, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))
    new_trade_gold_add_owner = game.player_cities.trade_gold_add_owner[player_id[0], city_int] + 2
    new_trade_gold_add_dest = game.player_cities.trade_gold_add_dest[player_id[0], city_int] + 1
    return game.replace(
        player_cities=game.player_cities.replace(
            trade_gold_add_owner=game.player_cities.trade_gold_add_owner.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_owner),
            trade_gold_add_dest=game.player_cities.trade_gold_add_dest.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_trade_gold_add_dest),
        )
    )


def _oracle(game, city_int, player_id):
    """
    +3 culture
    +1 SPECIALIST_SCIENTIST point
    +1 free social policy
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 3, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    return game


def _hanging_garden(game, city_int, player_id):
    """
    +1 culture, +6 food
    +1 free garden
    +1 SPECIALIST_ENGINEER points
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([6, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game

def _great_wall(game, city_int, player_id):
    """
    +1 culture
    +1 SPECIALIST_ENGINEER point
    Free wall in all cities
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game

def _angkor_wat(game, city_int, player_id):
    """
    +5 culture, +3 science
    +1 SPECIALIST_SCIENTIST point
    Free univerity
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 5, 3, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    return game

def _hagia_sophia(game, city_int, player_id):
    """
    +3 faith
    Free great prophet
    Free temple
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 3, 0, 0, 0, 0]))
    return game

def _chichen_itza(game, city_int, player_id):
    """
    +4 happiness, +1 culture
    +1 SPECIALIST_ENGINEER point
    +50% length golden age
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 4, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game


def _machu_pichu(game, city_int, player_id):
    """
    +2 faith, +5 gold
    +1 SPECIALIST_MERCHANT point
    +25% gold from city connections
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 5, 2, 0, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))
    new_connection_gold_accel = game.player_cities.city_connection_gold_accel[player_id[0], city_int] + 0.25
    return game.replace(player_cities=game.player_cities.replace(
        city_connection_gold_accel=game.player_cities.city_connection_gold_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_connection_gold_accel)
    ))


def _notre_dame(game, city_int, player_id):
    """
    +10 happiness, +4 faith
    +1 SPECIALIST_ARTIST point
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 4, 0, 0, 10, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([1, 0, 0, 0, 0, 0]))
    return game


def _porcelain_tower(game, city_int, player_id):
    """
    +1 culture
    +2 SPECIALIST_SCIENTIST points
    +3 science, +3 culture to all lux
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 2]))
    
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        this_city_resources = game.all_resource_type_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_lux = this_city_resources == 1

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_type_map[this_city_center[0], this_city_center[1]]
        this_city_center_lux = this_city_center_resources == 1

        return this_city_currently_owned * this_city_lux, game_map_rowcols, this_city_center_lux, this_city_center
    
    to_add = jnp.array([0, 0, 0, 0, 3, 3, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _himeji_castle(game, city_int, player_id):
    """
    +1 culture
    +2 SPECIALIST_ENGINEER points
    +15% combat strength for all units in frendily territory (to add in expansion?)
    Free castle
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 2, 0, 0]))
    return game

def _sistine_chapel(game, city_int, player_id):
    """
    +1 culture
    +25% culture in *all* cities
    +2 gw art
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([2, 0, 0, 0]))
    return game

def _kremlin(game, city_int, player_id):
    """
    +1 culture
    +50% prod for armored units
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    new_armored_accel = game.player_cities.armored_accel[player_id[0], city_int] + 0.5
    return game.replace(player_cities=game.player_cities.replace(
        armored_accel=game.player_cities.armored_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_armored_accel)
    ))


def _forbidden_palace(game, city_int, player_id):
    """
    +1 culture
    +2 delegates to world congress
    -10% unhappiness from citizens in unoccupied cities
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    return game

def _taj_mahal(game, city_int, player_id):
    """
    +4 happiness, +1 culture
    starts golden age
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 4, 0]))
    return game

def _big_ben(game, city_int, player_id):
    """
    +1 culture, +8 gold
    +2 SPECIALIST_MERCHANT points
    -12% cost of purchasing with gold
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 8, 0, 1, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0]))
    return game

def _louvre(game, city_int, player_id):
    """
    +2 culture
    +4 gw art
    1 free great artist
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([4, 0, 0, 0]))
    return game

def _brandenburg_gate(game, city_int, player_id):
    """
    +1 culture
    +2 SPECIALIST_SCIENTIST points
    1 free great general
    +15 XP for **all** units
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 2]))
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))

def _statue_of_liberty(game, city_int, player_id):
    """
    +1 culture, +6 happiness
    +6 population
    1 free social policy
    """
    #game = add_building_indicator(game, city_int, player_id, GameBuildings["statue_of_liberty"]._value_)
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 6, 0]))
    new_pop = game.player_cities.population[player_id[0], city_int] + 6
    return game.replace(
        player_cities=game.player_cities.replace(
            population=game.player_cities.population.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_pop)
        )
    )

def _cristo_redentor(game, city_int, player_id):
    """
    +5 culture
    -10% culture cost of new policies
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 5, 0, 0, 0]))
    return game

def _eiffel_tower(game, city_int, player_id):
    """
    +1 culture, +5 happiness
    +2 SPECIALIST_MERCHANT points
    +12 tourism
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 5, 12]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0]))
    return game

def _pentagon(game, city_int, player_id):
    """
    +1 culture
    +2 SPECIALIST_MERCHANT points
    +15 unit xp add
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0]))
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))

def _sydney_opera_house(game, city_int, player_id):
    """
    +50% culture
    1 free social policy
    +2 gw music (+2 theming bonus if same civ but diff era)
    """
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0.5, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 0, 2, 0]))
    return game

def _statue_zeus(game, city_int, player_id):
    """
    +2 culture
    +15 combat strength to all units
    +10% production toward **all** military
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    new_mounted_accel = game.player_cities.mounted_accel[player_id[0], city_int] + 0.10 
    new_land_unit_accel = game.player_cities.land_unit_accel[player_id[0], city_int] + 0.10
    new_ranged_accel = game.player_cities.ranged_accel[player_id[0], city_int] + 0.10
    
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15

    return game.replace(player_cities=game.player_cities.replace(
        mounted_accel=game.player_cities.mounted_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_mounted_accel),
        land_unit_accel=game.player_cities.land_unit_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_land_unit_accel),
        ranged_accel=game.player_cities.ranged_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_ranged_accel),
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))


def _temple_artemis(game, city_int, player_id):
    """
    +3 culture, +3 gold, +3 prod
    +1 SPECIALIST_ENGINEER point
    +15% production ranged units
    +15 XP ranged units
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 3, 3, 0, 3, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    new_ranged_accel = game.player_cities.ranged_accel[player_id[0], city_int] + 0.15
    new_ranged_xp_add = game.player_cities.ranged_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        ranged_accel=game.player_cities.ranged_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_ranged_accel),
        ranged_xp_add=game.player_cities.ranged_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_ranged_xp_add)
    ))


def _mausoleum_halicarnassus(game, city_int, player_id):
    """
    +1 culture
    +1 SPECIALIST_MERCHANT point
    +100 gold each gp expended
    +2 gold marble, obsidian, stone
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        marble = RESOURCE_TO_IDX["marble"]
        obsidian = RESOURCE_TO_IDX["obsidian"]
        stone = RESOURCE_TO_IDX["stone"]

        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        this_city_res = (this_city_resources == marble) | (this_city_resources == obsidian) | (this_city_resources == stone)

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        this_city_center_res = (this_city_center_resources == marble) | (this_city_center_resources == obsidian) | (this_city_center_resources == stone) 

        return this_city_currently_owned * this_city_res, game_map_rowcols, this_city_center_res, this_city_center

    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _intelligence_agency(game, city_int, player_id):
    """
    +1 spy
    +1 level on all spies
    15% reduction in enemy spy effectiveness
    """
    return game

def _alhambra(game, city_int, player_id):
    """
    +1 culture
    non-air military get Drill 1
    Free castle
    +20% culture
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_yield_multipliers(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0.2, 0, 0, 0]))
    new_unit_xp_add = game.player_cities.unit_xp_add[player_id[0], city_int] + 15
    return game.replace(player_cities=game.player_cities.replace(
        unit_xp_add=game.player_cities.unit_xp_add.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_unit_xp_add)
    ))


def _cn_tower(game, city_int, player_id):
    """
    +1 SPECIALIST_MERCHANT point
    +1 population 
    +1 happiness
    Free broadcast_tower
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 1, 0]))
    return game

def _hubble(game, city_int, player_id):
    """
    +10 science
    +3 SPECIALIST_SCIENTIST points
    Free recycling_center
    +200% prod to spaceship parts
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 10, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 3]))
    new_spaceship_prod_accel = game.player_cities.spaceship_prod_accel[player_id[0], city_int] + 2.0
    return game.replace(player_cities=game.player_cities.replace(
        spaceship_prod_accel=game.player_cities.spaceship_prod_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_spaceship_prod_accel)
    ))

def _leaning_tower(game, city_int, player_id):
    """
    +1 culture
    +20% great person in **all** cities
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    return game

def _mosque_of_djenne(game, city_int, player_id):
    """
    +3 happiness, +3 culture, +6 faith
    +1 SPECIALIST_ENGINEER point
    Missionaries from this city can spread 3 times
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 6, 3, 0, 3, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game

def _neuschwanstein(game, city_int, player_id):
    """
    +2 happiness, +4 culture, +6 gold
    +1 SPECIALIST_MERCHANT point
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 6, 0, 4, 0, 2, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0]))
    return game

def _petra(game, city_int, player_id):
    """
    +1 culture
    +1 SPECIALIST_ENGINEER point
    +1 food, +1 prod all desert *except* flood plains
    +1 trade route
    Free trade caravan
    +6 culture at archaeology
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 7, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))


    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        all_desert = game.terrain_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == DESERT_IDX
        not_river = game.edge_river_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]].sum(-1) == 0
        bool_mask = all_desert & not_river

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        all_desert = game.terrain_map[this_city_center[0], this_city_center[1]] == DESERT_IDX
        not_river = game.edge_river_map[this_city_center[0], this_city_center[1]].sum(-1) == 0
        bool_mask_center = all_desert & not_river

        return this_city_currently_owned * bool_mask, game_map_rowcols, bool_mask_center, this_city_center

    to_add = jnp.array([1, 1, 0, 0, 0, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game


def _terracotta_army(game, city_int, player_id):
    """
    +1 culture
    copy of each military unit (lol)
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    return game

def _great_firewall(game, city_int, player_id):
    """
    99.9% reduction in enemy spy effectiveness in city
    -25% effectiveness of all other spies in all other cities
    Negates other players' internet-based tourism bonus
    """
    new_reduce_accel = game.player_cities.tech_steal_reduce_accel[player_id[0], city_int] - 0.999
    return game.replace(player_cities=game.player_cities.replace(
        tech_steal_reduce_accel=game.player_cities.tech_steal_reduce_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_reduce_accel)
    ))

def _grand_temple(game, city_int, player_id):
    """
    +8 faith
    double religious pressure from city
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 8, 0, 0, 0, 0]))
    return game

def _tourist_center(game, city_int, player_id):
    """
    National Visitor Center
    +2 bldg_maintenance
    100% of culture from world/natural wonders, improvements added to tourism of city
    1.5x tourism output fro gq
    """
    game = add_bldg_maintenance(game, city_int, player_id, 2)
    new_gw_tourism_accel = game.player_cities.gw_tourism_accel[player_id[0], city_int] + 0.5
    new_culture_to_tourism = game.player_cities.culture_to_tourism[player_id[0], city_int] + 1.0

    return game.replace(player_cities=game.player_cities.replace(
        gw_tourism_accel=game.player_cities.gw_tourism_accel.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_gw_tourism_accel),
        culture_to_tourism=game.player_cities.culture_to_tourism.at[
            jnp.index_exp[player_id[0], city_int]
        ].set(new_culture_to_tourism)
    ))

def _uffizi(game, city_int, player_id):
    """
    +2 culture
    +3 gw art
    1 free great artist
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 3, 0, 0]))
    return game


def _globe_theater(game, city_int, player_id):
    """
    +2 culture
    +1 SPECIALIST_WRITER point
    +2 gw writing
    Free gp writer
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([2, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 1, 0, 0, 0]))
    return game

def _broadway(game, city_int, player_id):
    """
    +2 culture
    +3 gw music
    Free gp musician
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 0, 3, 0]))
    return game

def _red_fort(game, city_int, player_id):
    """
    +4 happiness, +8 culture
    +1200 defense
    +1 SPECIALIST_SCIENTIST point
    +25% more defense in all cities
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 8, 0, 4, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))
    new_defense = game.player_cities.defense[player_id[0], city_int] + 1200
    return game.replace(
        player_cities=game.player_cities.replace(
            defense=game.player_cities.defense.at[
                jnp.index_exp[player_id[0], city_int]
            ].set(new_defense),
        )
    )

def _prora_resort(game, city_int, player_id):
    """
    +2 happiness
    +1 happiness for every other social policy
    1 free social policy
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 2, 0]))
    return game


def _borobudur(game, city_int, player_id):
    """
    +5 faith
    +1 SPECIALIST_ENGINEER point
    3 free Missionaries
    Free garden
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 5, 0, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))
    return game

def _parthenon(game, city_int, player_id):
    """
    +4 culture
    +1 gw art
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 4, 0, 0, 0]))
    game = add_gw_slots(game, city_int, player_id, jnp.array([0, 1, 0, 0]))
    return game

def _international_space_station(game, city_int, player_id):
    """
    +1 production from scientists
    +1 science from engineers
    +33% more science from great scientist bulbs
    may only be built collaboratively
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 8, 0, 4, 0]))
    return game

def _conservatory(game, city_int, player_id):
    """
    From "Fine Arts" social policy -- first 4 cities
    +1 SPECIALIST_MUSICIAN slot
    +1 tourism
    """
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 1, 0, 0, 0, 0]))
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 0, 0, 1]))
    return game

def _stpeters(game, city_int, player_id):
    """
    +8 faith, +4 happiness
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 8, 0, 4, 0]))
    return game

def _althing(game, city_int, player_id):
    """
    +1 culture
    +1 SPECIALIST_SCIENTIST point
    +1 prod, +1 culture from tundra
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 0, 1]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        jungle = game.terrain_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == TUNDRA_IDX

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        center_jungle = game.terrain_map[this_city_center[0], this_city_center[1]] == TUNDRA_IDX

        return this_city_currently_owned * jungle, game_map_rowcols, center_jungle, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 1, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game 


def _grand_stele(game, city_int, player_id):
    """+3 culture, +4 faith"""
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    return game

def _panama(game, city_int, player_id):
    """
    +1 culture, +3 gold
    +2 SPECIALIST_MERCHANT points
    +8 gold from external sea trade
    Free great merchant
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 3, 0, 1, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 0, 2, 0]))
    return game

def _artist_house(game, city_int, player_id):
    """
    +2 faith, +1 culture
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 1, 2, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([1, 0, 0, 0, 0, 0]))
    return game

def _writer_house(game, city_int, player_id):
    """
    +1 faith, +3 culture
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 1, 3, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 1, 0, 0, 0]))
    return game

def _music_house(game, city_int, player_id):
    """
    +2 faith, +2 culture
    +1 musician specialist
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 2, 2, 0, 0, 0]))
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 1, 0, 0, 0, 0]))
    return game

def _lake_wonder(game, city_int, player_id):
    """
    Huey Teocalli
    +4 faith
    +1 SPECIALIST_ENGINEER point
    +2 food, +2 culture from lakes
    """
    game = add_bldg_yields(game, city_int, player_id, jnp.array([0, 0, 0, 4, 0, 0, 0, 0]))
    game = add_great_person_points(game, city_int, player_id, jnp.array([0, 0, 0, 1, 0, 0]))

    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities.ownership_map[player_id[0], city_int][
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        lakes = game.lake_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 1
        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        lakes_center = game.lake_map[this_city_center[0], this_city_center[1]] == 1

        return this_city_currently_owned * lakes, game_map_rowcols, lakes_center, this_city_center
    
    to_add = jnp.array([2, 0, 0, 0, 2, 0, 0])
    game = add_tile_yields(game, player_id, city_int, bool_map_generator, to_add)
    return game

def _apollo_program(game, city_int, player_id):
    return game

def _booster_1(game, city_int, player_id):
    return game

def _booster_2(game, city_int, player_id):
    return game

def _booster_3(game, city_int, player_id):
    return game

def _engine(game, city_int, player_id):
    return game

def _cockpit(game, city_int, player_id):
    return game

def _stasis_chamber(game, city_int, player_id):
    return game

def _gallery(game, city_int, player_id):
    """
    +1 happiness
    +1 artist slot
    """
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    game = add_bldg_yields(game, city_int,  player_id, to_add)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([1, 0, 0, 0, 0, 0]))
    return game

def _scriptorium(game, city_int, player_id):
    """
    +1 culture
    +1 writer slot
    """
    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0, 0])
    game = add_bldg_yields(game, city_int,  player_id, to_add)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 0, 1, 0, 0, 0]))
    return game

def _conservatory(game, city_int, player_id):
    """
    +1 tourism
    +1 musician slot
    """
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 0, 1])
    game = add_bldg_yields(game, city_int,  player_id, to_add)
    game = add_specialist_slots(game, city_int, player_id, jnp.array([0, 1, 0, 0, 0, 0]))
    return game


ALL_BLDG_FINISHERS = [no_change]

ALL_BLDG_NAMES = ["_" + x.name for x in GameBuildings]

for bldg_type in ALL_BLDG_NAMES:
    if bldg_type == "_hydro_plant":
        fn = getattr(sys.modules[__name__], "_hydroplant")
    elif bldg_type == "_theatre":
        fn = getattr(sys.modules[__name__], "_zoo")
    elif bldg_type == "_laboratory":
        fn = getattr(sys.modules[__name__], "_research_lab")
    elif bldg_type == "_recycling_center":
        fn = getattr(sys.modules[__name__], "_recycling_plant")
    elif bldg_type == "_constable":
        fn = getattr(sys.modules[__name__], "_constabulary")
    elif bldg_type == "_textile":
        fn = getattr(sys.modules[__name__], "_textile_mill")
    elif bldg_type == "_censer":
        fn = getattr(sys.modules[__name__], "_censer_maker")
    elif bldg_type == "_refinery":
        fn = getattr(sys.modules[__name__], "_oil_refinery")
    else:
        fn = getattr(sys.modules[__name__], bldg_type)
    
    ALL_BLDG_FINISHERS.append(fn)

print("Success.")
qqq

# we must use the x=x trick in the lambda fn's args
ALL_BLDG_COST = jnp.array([x.cost for x in GameBuildings])
NUM_BLDGS = len(GameBuildings)

BLDG_IS_WORLD_WONDER_FN = [lambda _: x.world_wonder for x in GameBuildings]
BLDG_IS_NAT_WONDER_FN = [lambda _: x.nat_wonder for x in GameBuildings]
BLDG_IS_WORLD_WONDER = jnp.array([x.world_wonder for x in GameBuildings])
BLDG_IS_NAT_WONDER = jnp.array([x.nat_wonder for x in GameBuildings])
ALL_BLDG_TYPES = jnp.array([x.building_type for x in GameBuildings])

# This is always used as masked by NATW | WW, so we can  safely skip the culture
# output from regular buildings
BLDG_CULTURE = jnp.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # monastery (12)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # barracks (24)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # nuclear_plant (36)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,  # national_epic (48)
    1, 0, 0, 0, 2, 1, 1, 1, 1, 1, 3, 1,  # hanging_garden (60)
    1, 5, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,  # taj_mahal (72)
    1, 2, 1, 1, 5, 1, 1, 0, 0, 0, 2, 3,  # temple_artemis (84)
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,  # leaning_tower (96)
    3, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  # artists_guild (108)
    0, 0, 0, 0, 2, 2, 2, 8, 0, 0, 4, 0,  # international_space_station (120)
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # grocer (132)
    0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # engine (144)
    0, 0, 0, 0
])
BLDG_IS_NAT_OR_WORLD_WONDER = BLDG_IS_NAT_WONDER | BLDG_IS_WORLD_WONDER


def _check_prereq(techs: jnp.ndarray,
                  city_owned_res: jnp.ndarray,
                  city_bldgs: jnp.ndarray,
                  policies: jnp.ndarray,
                  is_coastal: jnp.ndarray,
                  river_access: jnp.ndarray,
                  req_indices_tech: tuple[int, ...], 
                  req_indices_res: tuple[int, ...],
                  req_indices_bldg: tuple[str, ...],
                  req_indices_pol: tuple[str, ...],
                  req_coastal: bool,
                  req_river: bool,
                  cost: float) -> bool:
    """
    techs: (len(Technologies),)
    city_owned_res: (len(RESOURCE_TO_IDX),)
    city_bldgs: (len(GameBuildings),)
    """
    if not req_indices_tech:  # 0 prereqs ⇒ always OK
        tech_prereq = True
    else:
        req = jnp.asarray(req_indices_tech, dtype=jnp.int32)
        tech_prereq = jnp.all(techs[req] == 1)

    if len(req_indices_res) == 0:
        res_prereq = True
    else:
        # The resources are +1, so need to -1 
        req = jnp.asarray(req_indices_res, dtype=jnp.int32)
        res_prereq = jnp.any(city_owned_res[req - 1] > 0)

    if len(req_indices_bldg) == 0:
        bldg_prereq = True
    else:
        req_indices_bldg = [GameBuildings[x]._value_ for x in req_indices_bldg]
        req = jnp.asarray(req_indices_bldg, dtype=jnp.int32)
        bldg_prereq = jnp.all(city_bldgs[req] == 1)

    if len(req_indices_pol) == 0:
        pol_prereq = True
    else:
        req = jnp.asarray(req_indices_pol, dtype=jnp.int32)
        pol_prereq = jnp.all(policies[req] == 1)
    
    if req_coastal:
        coastal_prereq = is_coastal == 1
    else:
        coastal_prereq = True

    if req_river:
        river_prereq = river_access
    else:
        river_prereq = True
    
    # Need this one for buy-only buildings
    can_build = cost > 0

    return tech_prereq & res_prereq & bldg_prereq & can_build & pol_prereq & coastal_prereq & river_prereq


ALL_BLDG_PREREQ_FN = []
for bldg in GameBuildings:
    assert isinstance(bldg.prereq_building, list), f"{bldg}"
    fn = partial(_check_prereq, 
                 req_indices_tech=bldg.prereq_tech,
                 req_indices_res=bldg.resource_prereq,
                 req_indices_bldg=bldg.prereq_building,
                 req_indices_pol=bldg.prereq_pol,
                 req_coastal=bldg.coastal_prereq,
                 req_river=bldg.river,
                 cost=bldg.cost)
    
    ALL_BLDG_PREREQ_FN.append(fn)


def zero_out_fields_for_building_update(pytree, names_to_zero, idx_0, idx_1):
    """
    Returns a new pytree where specified fields have element [idx_0, idx_1] set to 0.
    Works in jitted context as long as names_to_zero is known at compile time.

    Args:
        pytree: flax.struct.dataclass instance
        names_to_zero: list of field names (must be static for JIT)
        idx_0, idx_1: integer indices

    Returns:
        New pytree with indexed elements zeroed in specified fields.
    """
    return type(pytree)(**{
        f.name: (
            getattr(pytree, f.name).at[idx_0, idx_1].set(
                0 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 0
            )
            if f.name in names_to_zero
            else getattr(pytree, f.name)
        )
        for f in fields(pytree)
    })

def add_one_to_appropriate_fields(pytree, relevant_fields, idx_0, idx_1, all_cities=False):
    """
    Fields that are multipliers (e.g., any "accel") need a base one 1 added to them.
    Set all_cities=True when you want to vectorize the addition over all cities at once.
    """
    if all_cities:
        return type(pytree)(**{
            f.name: (
                getattr(pytree, f.name).at[idx_0].add(
                    1 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 6 if ("air_unit_capacity" in f.name) else 0
                )
                if f.name in relevant_fields
                else getattr(pytree, f.name)
            )
            for f in fields(pytree)
        })

    else:
        return type(pytree)(**{
            f.name: (
                getattr(pytree, f.name).at[idx_0, idx_1].add(
                    1 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 6 if ("air_unit_capacity" in f.name) else 0
                )
                if f.name in relevant_fields
                else getattr(pytree, f.name)
            )
            for f in fields(pytree)
        })

def extract_fields(pytree, names_to_extract, player_id, city_int, all_fns=False):
    """
    Extract a dict of selected fields from a dataclass-based pytree.
    To be called within a vmap-over-building-idx context

    Works under JAX JIT as long as `names_to_extract` is static.

    Returns:
        A flat dict {field_name: field_value} for the selected fields.
    """
    if all_fns:
        return {
            f.name: getattr(pytree, f.name)[:, player_id[0], city_int]
            for f in fields(pytree)
            if f.name in names_to_extract
        }
    else:
        return {
            f.name: getattr(pytree, f.name)[player_id[0], city_int]
            for f in fields(pytree)
            if f.name in names_to_extract
        }

def extract_map_fields(pytree, names_to_extract, player_id, all_fns=False):
    """
    Extract a dict of selected fields from a dataclass-based pytree.
    To be called within a vmap-over-building-idx context

    Works under JAX JIT as long as `names_to_extract` is static.

    Returns:
        A flat dict {field_name: field_value} for the selected fields.
    """
    if all_fns:
        return {
            f.name: getattr(pytree, f.name)[:, player_id[0]]
            for f in fields(pytree)
            if f.name in names_to_extract
        }
    else:
        return {
            f.name: getattr(pytree, f.name)[player_id[0]]
            for f in fields(pytree)
            if f.name in names_to_extract
        }

player_city_update_fn_nonmaps = make_update_fn(TO_ZERO_OUT_FOR_BUILDINGS_STEP_SANS_MAPS, only_maps=False)
player_city_update_fn_maps = make_update_fn(TO_ZERO_OUT_FOR_BUILDINGS_STEP_ONLY_MAPS, only_maps=True)


def apply_buildings_per_city_minimal(game, player_id, city_int):
    _buildings_owned = game.player_cities.buildings_owned[player_id[0], city_int]
    _arange_vmap_helper = jnp.arange(0, len(GameBuildings)) + 1
    _buildings_owned_idx = _buildings_owned * _arange_vmap_helper
    
    # The names to zero-out
    _player_cities = zero_out_fields_for_building_update(game.player_cities, TO_ZERO_OUT_FOR_BUILDINGS_STEP, player_id, city_int)
    # I am little unsure about the indexing for the map-shape... just for safety:
    _player_cities = _player_cities.replace(additional_yield_map=_player_cities.additional_yield_map * 0)
    game = game.replace(player_cities=_player_cities)

    # Running an experiment now on memory usage. Instead of sending the entire gamestate object into the building 
    # finisher functions, let's send a minimal subset of the fields. This should drastically reduce the memory 
    # requirements. This is mainly due to jax's lack of batched cond, which results in each branch creating 
    # a duplicate of the gamestate  in memory. 
    minimal_game = create_minimal_update(game)
    
    has_this_bldg_bool = game.player_cities.buildings_owned[player_id[0], city_int]
    
    outs = [f(minimal_game, city_int, player_id) for f in ALL_BLDG_FINISHERS[1:]]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outs)

    # If we minimize the size of the pytree, perhaps throughput will improve?
    def weighted_sum_dict(field_dict, weights):
        return jax.tree_util.tree_map(
            lambda field_array: jnp.einsum('i,i...->...', weights, field_array),
            field_dict
        )
    _out2 = extract_fields(stacked.player_cities, TO_ZERO_OUT_FOR_BUILDINGS_STEP_SANS_MAPS, player_id, city_int, all_fns=True)
    out_maps = extract_map_fields(stacked.player_cities, TO_ZERO_OUT_FOR_BUILDINGS_STEP_ONLY_MAPS, player_id, all_fns=True)
    out = {**_out2, **out_maps}
    out = weighted_sum_dict(out, has_this_bldg_bool)
    return out
