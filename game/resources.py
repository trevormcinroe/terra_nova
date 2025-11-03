import jax
from jax._src.core import Value
import jax.numpy as jnp
import numpy as np
import enum

from .techs import Technologies

# This is the order referenced in the renderer, so it would be convenient if we could stick
# to this index order all throughout the game. We can just do idx + 1 to allow 0s to mean 
# "no resource" on the resource map
ALL_RESOURCES = [
    "dyes",
    "copper",
    "deer",
    "ivory",
    "silver",
    "jewelry",
    "uranium",
    "lapis",
    "gems",
    "iron",  # 10
    "wine",
    "cow",
    "coconut",
    "wheat",
    "oil",
    "marble",
    "tobacco",
    "maize",
    "whales",
    "olives",  # 20
    "truffles",
    "bison",
    "sugar",
    "horses",
    "citrus",
    "cotton",
    "salt",
    "gold",
    "aluminium",
    "incense",  # 30
    "coffee",
    "crabs",
    "silk",
    "perfume",
    "glass",
    "spices",
    "amber",
    "chocolate",
    "rubber",
    "coal",  # 40
    "sheep",
    "coral",
    "furs",
    "porcelain",
    "fish",
    "tea",
    "hardwood",
    "obsidian",
    "banana",
    "jade",  # 50
    "pearls",
    "stone",
]

IS_LUX = jnp.array([
    1,
    1,
    0,
    1, 
    1,
    1,
    0,
    1,  
    1, 
    0,  # 10
    1, 
    0,
    1, 
    0, 
    0, 
    1, 
    1, 
    0,
    1,
    1,  # 20
    1,  
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,  # 30
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,  # 40
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,  # 50
    1,
    0
])

# Now we need to map from luxuries to the improvement type that gives the player
# happiness. For non-lux, we can just do 0. Here, we add +1 to save on cycles
class FauxImprovements(enum.IntEnum):
    """
    name: id, tech prereq, terrain prereq, res prereq, must be on resources

    For oil well, let's just use "mine" on top of oil :)
    """
    farm = 0, Technologies.agriculture, [], [], False
    pasture = 1, Technologies.animal_husbandry, [], [], True
    mine = 2, Technologies.mining, [], [], False
    fishing_boat = 3, Technologies.sailing, [], [], True
    plantation = 4, Technologies.calendar, [], [], True
    camp = 5, Technologies.trapping, [], [], True
    quarry = 6, Technologies.masonry, [], [], True
    lumber_mill = 7, Technologies.construction, [], [], False
    fort = 8, Technologies.engineering, [], [], False
    trading_post = 9, Technologies.guilds, [], [], False
    road = 10, Technologies.the_wheel, [], [], False
    chop_forest =  11, Technologies.mining, [], [], False
    chop_jungle = 12, Technologies.bronze_working, [], [], False
    clear_marsh = 13, Technologies.masonry, [], [], False

    def __new__(cls, value: int, tech_prereq, terrain_prereq, res_prereq, requires_res):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.tech_prereq = tech_prereq
        obj.terrain_prereq = terrain_prereq
        obj.res_prereq = res_prereq
        obj.requires_res = requires_res
        return obj


RESOURCE_TO_IMPROVEMENT = jnp.array([
    99,  # This is for NO improvement
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.camp + 1,
    FauxImprovements.camp + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,  # 10
    FauxImprovements.plantation + 1,
    FauxImprovements.pasture + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.farm + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.quarry + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.farm + 1,
    FauxImprovements.fishing_boat + 1,
    FauxImprovements.plantation + 1,  # 20
    FauxImprovements.camp + 1,
    FauxImprovements.camp + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.pasture + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.plantation + 1,  # 30
    FauxImprovements.plantation + 1,
    FauxImprovements.fishing_boat + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,  # 40
    FauxImprovements.pasture + 1,
    FauxImprovements.fishing_boat + 1,
    FauxImprovements.camp + 1,
    FauxImprovements.quarry + 1,
    FauxImprovements.fishing_boat + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.lumber_mill + 1,
    FauxImprovements.quarry + 1,
    FauxImprovements.plantation + 1,
    FauxImprovements.mine + 1,  # 50
    FauxImprovements.fishing_boat + 1,
    FauxImprovements.quarry + 1
])


RESOURCE_TO_IDX = {
    ALL_RESOURCES[i]: i + 1 for i in range(len(ALL_RESOURCES))
}

# [revealed, improvable]
ALL_RESOURCES_TECH = {
    "dyes": [Technologies.agriculture, Technologies.calendar],
    "copper": [Technologies.agriculture, Technologies.mining],
    "deer": [Technologies.agriculture, Technologies.trapping],
    "ivory": [Technologies.agriculture, Technologies.trapping],
    "silver": [Technologies.agriculture, Technologies.mining],
    "jewelry": [Technologies.agriculture, Technologies.mining],
    "uranium": [Technologies.atomic_theory, Technologies.mining],
    "lapis": [Technologies.agriculture, Technologies.mining],
    "gems": [Technologies.agriculture, Technologies.mining],
    "iron": [Technologies.mining, Technologies.mining],
    "wine": [Technologies.agriculture, Technologies.calendar],
    "cow": [Technologies.agriculture, Technologies.animal_husbandry],
    "coconut": [Technologies.agriculture, Technologies.calendar],
    "wheat": [Technologies.agriculture, Technologies.agriculture],
    "oil": [Technologies.biology, Technologies.mining],
    "marble": [Technologies.agriculture, Technologies.masonry],
    "tobacco": [Technologies.agriculture, Technologies.calendar],
    "maize": [Technologies.agriculture, Technologies.agriculture],
    "whales": [Technologies.agriculture, Technologies.sailing],
    "olives": [Technologies.agriculture, Technologies.calendar],
    "truffles": [Technologies.agriculture, Technologies.trapping],
    "bison": [Technologies.agriculture, Technologies.trapping],
    "sugar": [Technologies.agriculture, Technologies.calendar],
    "horses": [Technologies.animal_husbandry, Technologies.animal_husbandry],
    "citrus": [Technologies.agriculture, Technologies.calendar],
    "cotton": [Technologies.agriculture, Technologies.calendar],
    "salt": [Technologies.agriculture, Technologies.mining],
    "gold": [Technologies.agriculture, Technologies.mining],
    "aluminium": [Technologies.electricity, Technologies.mining],
    "incense": [Technologies.agriculture, Technologies.calendar],
    "coffee": [Technologies.agriculture, Technologies.calendar],
    "crabs": [Technologies.agriculture, Technologies.sailing],
    "silk": [Technologies.agriculture, Technologies.calendar],
    "perfume": [Technologies.agriculture, Technologies.calendar],
    "glass": [Technologies.agriculture, Technologies.mining],
    "spices": [Technologies.agriculture, Technologies.calendar],
    "amber": [Technologies.agriculture, Technologies.mining],
    "chocolate": [Technologies.agriculture, Technologies.calendar],
    "rubber": [Technologies.agriculture, Technologies.calendar],
    "coal": [Technologies.industrialization, Technologies.mining],
    "sheep": [Technologies.agriculture, Technologies.animal_husbandry],
    "coral": [Technologies.agriculture, Technologies.sailing],
    "furs": [Technologies.agriculture, Technologies.trapping],
    "porcelain": [Technologies.agriculture, Technologies.masonry],
    "fish": [Technologies.agriculture, Technologies.sailing],
    "tea": [Technologies.agriculture, Technologies.calendar],
    "hardwood": [Technologies.agriculture, Technologies.bronze_working],
    "obsidian": [Technologies.agriculture, Technologies.masonry],
    "banana": [Technologies.agriculture, Technologies.calendar],
    "jade": [Technologies.agriculture, Technologies.mining],
    "pearls": [Technologies.agriculture, Technologies.sailing],
    "stone": [Technologies.agriculture, Technologies.masonry],
}

ALL_RESOURCES_TECH = jnp.concatenate([
    jnp.array([[0, 0]]),
    jnp.array([v for k, v in ALL_RESOURCES_TECH.items()])
], axis=0)


# [food, prod, gold, faith, culture, science, happiness]
# YIELDS ARE IN TOTAL FOR THE RESOURCE, ADDITIVE FOR THE TILE
# [unimproved, improved]
RESOURCE_YIELDS = [
    # dyes
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 4]
    ]),
    # "copper",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 4]
    ]),
    # "deer",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ]),
    #"ivory",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0, 4]
    ]),
    # "silver",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 4]
    ]),
    # "jewelry", 
    jnp.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4]
    ]),
    # "uranium",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    # "lapis",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 4]
    ]),
    # "gems",
    jnp.array([
        [0, 0, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 4]
    ]),
    #"iron",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    #"wine",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0, 4]
    ]),
    #"cow",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ]),
    # "coconut",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [2, 0, 2, 0, 0, 0, 4]
    ]),
    #"wheat",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ]),
    # "oil",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0]
    ]),
    #"marble",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 4]
    ]),
    # "tobacco",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 4]
    ]),
    # "maize",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ]),
    # "whales",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 2, 0, 0, 0, 4]
    ]),
    # "olives",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [2, 0, 2, 0, 0, 0, 4]
    ]),
    # "truffles",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 2, 0, 0, 0, 4]
    ]),
    # "bison",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ]),
    # "sugar",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0, 4]
    ]),
    # "horses",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    # "citrus",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 4]
    ]),
    # "cotton",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 3, 0, 0, 0, 4]
    ]),
    # "salt",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 4]
    ]),
    # "gold",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 4]
    ]),
    # "aluminium",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    # "incense",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 4]
    ]),
    # "coffee",
    jnp.array([
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 3, 0, 0, 0, 4]
    ]),
    # "crabs",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [2, 0, 2, 0, 0, 0, 4]
    ]),
    # "silk",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 4]
    ]),
    # "perfume",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 4]
    ]),
    #"glass",
    jnp.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4]
    ]),
    # "spices",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 4, 0, 0, 0, 4]
    ]),
    # "amber",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 4]
    ]),
    # "chocolate",
    jnp.array([
        [1, 0, 1, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 4]
    ]),
    # "rubber",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 3, 0, 0, 0, 4]
    ]),
    # "coal",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    # "sheep",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ]),
    # "coral",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 3, 1, 0, 0, 4]
    ]),
    # "furs",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0, 4]
    ]),
    #"porcelain",
    jnp.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4]
    ]),
    # "fish",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ]),
    # "tea",
    jnp.array([
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 3, 0, 0, 0, 4]
    ]),
    #"hardwood",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
    # "obsidian",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 4]
    ]),
    # "banana",
    jnp.array([
        [1, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0]
    ]),
    # "jade",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 4]
    ]),
    # "pearls",
    jnp.array([
        [0, 0, 2, 0, 0, 0, 0],
        [1, 0, 3, 0, 1, 0, 4]
    ]),
    # "stone",
    jnp.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0]
    ]),
]


OCEAN_LUX = [
    "coral",
    "whales",
    "pearls",
    "crabs",
]

LAND_LUX = [
    "furs",
    "coconut",
    "cotton",
    "wine",
    "tea",
    "coffee",
    "sugar",
    "lapis",
    "jade",
    "ivory",
    "citrus",
    "chocolate",
    "copper",
    "gold",
    "silver",
    "truffles",
    "incense",
    "tobacco",
    "olives",
    "dyes",
    "silk",
    "gems",
    "marble",
    "obsidian",
    "perfume",
    "rubber",
    "salt",
    "spices",
    "amber",
    "glass",
    "jewelry",
    "porcelain",
]

LUX_BIAS_TABLE = {
    "furs": {
        "regional_bias": ["tundra", ],
        "terrain_bias": ["furs"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "coconut": {
        "regional_bias": ["jungle"],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "cotton": {
        "regional_bias": ["grass", "plains", "desert", "tundra", "jungle", "marsh"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "wine": {
        "regional_bias": ["grass", "plains"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "tea": {
        "regional_bias": ["grass", "plains", "marsh"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "coffee": {
        "regional_bias": ["grass", "plains", "marsh"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "sugar": {
        "regional_bias": ["plains", "desert", "grass", "marsh", "jungle"], # added jungle for balance
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "lapis": {
        "regional_bias": ["tundra", "hill"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "jade": {
        "regional_bias": ["tundra", "hill", "grass", "plains", "desert", "jungle"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "ivory": {
        "regional_bias": ["plains"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "citrus": {
        "regional_bias": ["grass", "jungle", "forest", "marsh"],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "chocolate": {
        "regional_bias": ["jungle", "marsh"],  # added marsh for balance?
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "copper": {
        "regional_bias": ["hill"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "jewelry": {
        "regional_bias": ["hill"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "gold": {
        "regional_bias": ["hill", "desert", "plains", "tundra"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "silver": {
        "regional_bias": ["hill", "desert"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "glass": {
        "regional_bias": ["hill", "desert"],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "truffles": {
        "regional_bias": ["forest", "marsh", "jungle"],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "incense": {
        "regional_bias": ["desert", "plains"],
        "terrain_bias": ["incense"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "tobacco": {
        "regional_bias": ["grass", "plains"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "olives": {
        "regional_bias": ["grass", "plains"],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "dyes": {
        "regional_bias": ["jungle", "forest"],
        "terrain_bias": ["dyes"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "silk": {
        "regional_bias": ["forest", "grass", "plains", "desert", "jungle"],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "amber": {
        "regional_bias": ["hills"],
        "terrain_bias": ["rock"]
    },
    "gems": {
        "regional_bias": [False],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "marble": {
        "regional_bias": [False],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "obsidian": {
        "regional_bias": [False],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "porcelain": {
        "regional_bias": [False],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "perfume": {
        "regional_bias": [False],
        "terrain_bias": ["flats"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "rubber": {
        "regional_bias": [False],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "salt": {
        "regional_bias": [False],
        "terrain_bias": ["rock"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "spices": {
        "regional_bias": [False],
        "terrain_bias": ["tree"],
        "feature_bias": [],
        "elevation_bias": []
    },
    "coral": {"terrain_bias": ["ocean"], "regional_bias": ["ocean"]},
    "whales": {"terrain_bias": ["ocean"], "regional_bias": ["ocean"]},
    "pearls": {"terrain_bias": ["ocean"], "regional_bias": ["ocean"]},
    "crabs": {"terrain_bias": ["ocean"], "regional_bias": ["ocean"]},
}


# These can be a list up to length 5!
STRATEGIC_BIAS_TABLE = {
    "iron": {
        "terrain_bias": ["hill", "plains", "grass"]
    },
    "oil": {
        "terrain_bias": ["plains", "grass", "jungle", "tundra", "ocean"]
    },
    "uranium": {
        "terrain_bias": ["hill", "plains", "grass", "tundra", "desert"]
    },
    "horses": {
        "terrain_bias": ["plains", "grass", "tundra", "desert"]
    },
    "coal": {
        "terrain_bias": ["hill", "grass", "plains", "desert", "tundra"]
    },
    "aluminium": {
        "terrain_bias": ["grass", "plains", "desert", "hill", "tundra"]
    },
    "fish": {
        "terrain_bias": []
    },
    "deer": {
        "terrain_bias": []
    },
    "wheat": {
        "terrain_bias": []
    },
    "banana": {
        "terrain_bias": []
    },
    "bison": {
        "terrain_bias": []
    },
    "cow": {
        "terrain_bias": []
    },
    "stone": {
        "terrain_bias": []
    },
    "sheep": {
        "terrain_bias": []
    },
    "hardwood": {
        "terrain_bias": []
    },
    "maize": {
        "terrain_bias": []
    },
}


def translate_terrain_bias_to_tile_samples_strategic(bias_type, landmask, features, terrain, elevation_map, potential_tiles, settler_rc, nw_map, lakes, current_resource_map, key, num_samples):
    
    # We now need to filter the list of potential_tiles (1) settler location, (2) mountain, (3) lake, (4) natural wonder, (5) ocean if bias_type != "ocean"
    filtered_rowcols = []
    for rowcol in potential_tiles:
        if current_resource_map[rowcol[0], rowcol[1]] > 0:
            continue
        if (rowcol - jnp.array(settler_rc)).sum() == 0:
            continue
        if elevation_map[rowcol[0], rowcol[1]] == 3:
            continue
        if lakes[rowcol[0], rowcol[1]] == 1:
            continue
        if nw_map[rowcol[0], rowcol[1]] > 0:
            continue
        if bias_type != "ocean":
            if landmask[rowcol[0], rowcol[1]] == 0:
                continue
        else:
            if landmask[rowcol[0], rowcol[1]] == 1:
                continue

        filtered_rowcols.append(np.array(rowcol).tolist())
    
    potential_tiles = filtered_rowcols

    found = False
    
    while not found:
        sampling_weights = [0.6, 0.3, 0.05, 0.03, 0.02]
        sample_idx = jax.random.choice(key=key, shape=(), a=jnp.arange(0, len(bias_type)), p=jnp.array(sampling_weights[:len(bias_type)])).item()
        key, _ = jax.random.split(key, 2)
        sampled_bias = bias_type[sample_idx]

        if sampled_bias == "hill":
            true_tiles = (elevation_map == 2) 
        elif sampled_bias == "plains":
            true_tiles = (terrain == 2) & (elevation_map != 2) & (elevation_map != 3) & (features != 3) & (features != 2)
        elif sampled_bias == "grass":
            true_tiles = (terrain == 1) & (elevation_map != 2) & (elevation_map != 3) & (features != 3) & (features != 2)
        elif sampled_bias == "jungle":
            true_tiles = (features == 2)
        elif sampled_bias == "ocean":
            true_tiles = ~landmask
        elif sampled_bias == "tundra":
            true_tiles = (terrain == 4) & (elevation_map != 2) & (elevation_map != 3)
        elif sampled_bias == "desert":
            true_tiles = (terrain == 3) & (elevation_map != 2)
        else:
            raise ValueError(f"Do not current support bias_type={sampled_bias}")
        
        potential_tiles_inside_level = []
        for potential_tile in potential_tiles:
            if true_tiles[potential_tile[0], potential_tile[1]]:
                potential_tiles_inside_level.append(np.array(potential_tile).tolist())

        if len(potential_tiles_inside_level) >= num_samples:
            found = True
    
    final_sampled_tiles = []
    for i in range(num_samples):
        sampled_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=len(potential_tiles_inside_level)).item()
        key, _ = jax.random.split(key)
        final_sampled_tiles.append(potential_tiles_inside_level[sampled_idx])
        del potential_tiles_inside_level[sampled_idx]
    return final_sampled_tiles
    


def translate_terrain_bias_to_tile_samples(bias_type, landmask, features, terrain, elevation_map, potential_tiles, settler_rc, nw_map, lakes, current_resource_map, key, num_samples):
    """
    Don't worry too much about the specific names used in the bias type. I chose the name (or close)
    to the name used in the source code; mainly to stop my own headaches. 

    - rock
        27 = hills open no grass no plains
        24 = hills open no tundra no desert
        36 = flat no grass no plains
        37 = flat open no tundra no desert
        5 = hills covered
        31 = flat covered
    - tree
        8 = jungle flat
        15 = forest flat no tundra
        28 = hills covered no tundra
        37 = flat open no tundra no desert
        24 = hills open no tundra no desert
        2 = marsh
    - dyes
        9 = forest flat
        8 = jungle flat
        5 = hills covered
        38 = flat open no desert
        23 = hills open no desert
        2 = marsh
    - furs
        21 = tundra flat forest
        15 = forest flat not tundra
        7 = hills forest
        38 = flat open no desert
        23 = hills open no desert
        2 = marsh
    - incense
        38 = flatland desert including floodplains
        11 = plains flat
        22 = tundra hill
        33 = flat covered tundra
        28 = hills covered tundra
        16 = grass flat open
    - flats
        11 = plains open flat
        16 = frass open flat
        33 = flat covered no tundra
        24 = hills open no tundra no desert 
        28 = hills covered no tundra
        2 = marsh
    """
    if bias_type == "rock":
        primary = (elevation_map == 2) & (terrain != 1) & (terrain != 2)
        secondary = (elevation_map == 2) & (terrain != 4) & (terrain != 3)
        tertiary = (elevation_map == 1) & (terrain != 4) & (terrain != 3)
        quaternary = (elevation_map == 1) & (features != 1) & (features != 2) & (terrain != 4) & (terrain != 3) 
        quinary = ((elevation_map == 2) & (features == 1)) | ((elevation_map == 2) & (features == 2))
        senary = ((elevation_map == 1) & (features == 1)) | ((elevation_map == 1) & (features == 2)) 
    
    elif bias_type == "tree":
        primary = (features == 2) & (elevation_map == 1)
        secondary = (features == 1) & (elevation_map == 1) & (terrain != 4)
        tertiary = (elevation_map == 2) & ((features == 1) | (features == 2)) & (terrain != 4)
        quaternary = (elevation_map == 1) & (features != 1) & (features != 1) & (terrain != 4) & (terrain != 3)
        quinary = (elevation_map == 2) & (features != 1) & (features != 1) & (terrain != 4) & (terrain != 3) 
        senary = features == 3

    elif bias_type == "dyes":
        primary = (elevation_map == 1) & (features == 1)
        secondary = (elevation_map == 1) & (features == 2)
        tertiary = (elevation_map == 2) & ((features == 1) | (features == 2))
        quaternary = (elevation_map == 1) & (features != 1) & (features != 2) & (terrain != 3)
        quinary = (elevation_map == 2) & (features != 1) & (features != 2) & (terrain != 3)
        senary = features == 3

    elif bias_type == "furs":
        primary = (elevation_map == 1) & (terrain == 4) & (features == 1)
        secondary = (elevation_map == 1) & (features == 1) & (terrain != 4)
        tertiary = (elevation_map == 2) & (features == 1)
        quaternary = (elevation_map == 1) & (features != 1) & (features != 2) & (terrain != 3)
        quinary = (elevation_map == 2) & (features != 1) & (features != 2) & (terrain != 3)
        senary = features == 3

    elif bias_type == "incense":
        primary = (elevation_map == 1) & (terrain == 3)
        secondary = (elevation_map == 1) & (terrain == 2)
        tertiary = (terrain == 4) & (elevation_map == 2)
        quaternary = (elevation_map == 1) & ((features == 1) | (features == 2)) & (terrain == 4)
        quinary = (elevation_map == 2) & ((features == 1) | (features == 2)) & (terrain == 4)
        senary = (elevation_map == 1) & (terrain == 2) & (features != 1) & (features != 2)

    elif bias_type == "flats":
        primary = (elevation_map == 1) & (terrain == 2) & (features != 1) & (features != 2)
        secondary = (elevation_map == 1) & (terrain == 1) & (features != 1) & (features != 2) 
        tertiary = (elevation_map == 1) & ((features == 1) | (features == 2)) & (terrain != 4)
        quaternary = (elevation_map == 2) & (features != 1) & (features != 2) & (terrain != 3) & (terrain != 4)
        quinary = (elevation_map == 2) & ((features == 1) | (features == 2)) & (terrain != 4)
        senary = (elevation_map == 2) & ((features == 1) | (features == 2)) & (terrain != 4)
    elif bias_type == "ocean":
        primary = ~landmask
        secondary = primary
        tertiary = primary
        quaternary = primary
        quinary = primary
        senary = primary
    else:
        raise ValueError(f"Should never be here, but you gave me bias_type={bias_type}")
    
    # We now need to filter the list of potential_tiles (1) settler location, (2) mountain, (3) lake, (4) natural wonder, (5) ocean if bias_type != "ocean"
    filtered_rowcols = []
    for rowcol in potential_tiles:
        if current_resource_map[rowcol[0], rowcol[1]] > 0:
            continue
        if (rowcol - jnp.array(settler_rc)).sum() == 0:
            continue
        if elevation_map[rowcol[0], rowcol[1]] == 3:
            continue
        if lakes[rowcol[0], rowcol[1]] == 1:
            continue
        if nw_map[rowcol[0], rowcol[1]] > 0:
            continue
        if bias_type != "ocean":
            if landmask[rowcol[0], rowcol[1]] == 0:
                continue
        else:
            if landmask[rowcol[0], rowcol[1]] == 1:
                continue

        filtered_rowcols.append(np.array(rowcol).tolist())
    
    potential_tiles = filtered_rowcols

    picked_rowcols = []
    levels = [primary, secondary, tertiary, quaternary, quinary, senary]
    for _ in range(num_samples):
        found_enough_possible = False
        
        num_retries = 0

        while not found_enough_possible:
            # (1) pick primary, secondary, etc
            picked_level = jax.random.choice(
                key=key, 
                a=jnp.arange(0, 5), 
                shape=(), 
                p=jnp.array([0.5, 0.3, 0.1, 0.06, 0.04])
            ).item()

            picked_level = levels[picked_level]
            key, _ = jax.random.split(key, 2)
            
            potential_tiles_inside_level = []
            for potential_tile in potential_tiles:
                if picked_level[potential_tile[0], potential_tile[1]]:
                    potential_tiles_inside_level.append(potential_tile)

            key, _ = jax.random.split(key, 2)
            if len(potential_tiles_inside_level) > 0:
                found_enough_possible = True

            num_retries += 1
            if num_retries > 500:
                raise ValueError("Could not find enough appropriate tiles.")

        # Here, we have found at least one tile for the given lux that satisfies one of the 
        # prefernce levels for its spawn.
        _picked_rowcol = jax.random.randint(key=key, shape=(), minval=0, maxval=len(potential_tiles_inside_level)).item()
        _picked_rowcol = potential_tiles_inside_level[_picked_rowcol]
        picked_rowcols.append(_picked_rowcol)

        # Now we need to remove this rowcol from the list of potential_tiles
        del potential_tiles[potential_tiles.index(_picked_rowcol)]
        key, _ = jax.random.split(key, 2)

    return picked_rowcols
