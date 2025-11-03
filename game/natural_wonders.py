"""
From LekMod notes (https://docs.google.com/document/d/18tsjg2C1wKA7I41GktDRr6R83eUrhn4FHi9EUEtpKvI/edit?tab=t.0):
    - Krakatoa: 2 Food, 8 Science, +5 science from Observatory
    - Gibraltar: 3 Food, 7 Gold, +5 science from Observatory
    - Kailash: 6 Faith, 2 Happiness, +5 Science from Observatory
    - Kilimanjaro: 3 Food, 2 Culture, +5 Science from Observatory, Gives Altitude Training promotion: double movement + 10% Combat Modifier on hills.
    - Sinai: 8 Faith, +5 Science from Observatory
    - Sri Pada: 2 Food, 4 Faith, 2 Happiness. Can now spawn on the mainland, +5 Science from Observatory
    - Cerro de Potosi: 1 Production, 10 Gold, +5 Science from Observatory
    - Fuji: 2 Happiness, 2 Culture, 4 Faith and +5 Science from Observatory
    - Uluru: 2 Food, 6 Faith, +5 Science from Observatory
    - Barringer Crater: 2 Production, 3 Gold, 5 Science, +5 Science from Observatory
    - Grand Mesa: 3 Food, 3 Production, 3 Gold, +5 Science from Observatory
    - Old Faithful: 4 Happiness, 4 Science, +5 Science from Observatory
    - Fountain of Youth: now also provides Fresh Water and only 6 happiness, spawns as often as Cerro de Potosi. Additionally provides the ‘Heals at double rate’ promotion, which causes the unit to heal 10 additional HP per turn.
    - Great Barrier Reef: 2 Food, 1 Production, 1 Gold, 2 Science, +1 food from Lighthouse and Treasure Fleets work on it.
    - El Dorado: 5 Culture. Now gives 150 free gold instead of 500, spawns as often as Cerro de Potosi
    - Solomon’s Mines: 6 Production
    - Lake Victoria: 6 food, 2 Gold, Canada UA, Floating Gardens and Huey Teocalli work on it.
"""
import jax.numpy as jnp

ALL_NATURAL_WONDERS = [
    "Krakatoa",
    "Gibraltar",
    "Kailash",
    "Kilimanjaro",
    "Sinai",
    "Sri Pada",
    "Cerro de Potosi",
    "Fuji",
    "Uluru",
    "Barringer Crater",
    "Grand Mesa",
    "Old Faithful",
    "Fountain of Youth",
    "Great Barrier Reef",
    "El Dorado",
    "Solomon's Mines",
    "Lake Victoria"
]

LAKE_VICTORIA_IDX = ALL_NATURAL_WONDERS.index("Lake Victoria")


# [food, prod, gold, faith, culture, science]
NW_YIELD_TABLE = {
    "Krakatoa": jnp.array([2, 0, 0, 0, 0, 8, 0]),
    "Gibraltar": jnp.array([3, 0, 7, 0, 0, 0, 0]),
    "Kailash": jnp.array([0, 0, 0, 6, 0, 0, 2]),
    "Kilimanjaro": jnp.array([3, 0, 0, 0, 2, 0, 0]),
    "Sinai": jnp.array([0, 0, 0, 8, 0, 0, 0]),
    "Sri Pada": jnp.array([2, 0, 0, 4, 0, 0, 2]),
    "Cerro de Potosi": jnp.array([0, 1, 10, 0, 0, 0, 0]),
    "Fuji": jnp.array([0, 0, 0, 4, 2, 0, 2]),
    "Uluru": jnp.array([2, 0, 0, 6, 0, 0, 0]),
    "Barringer Crater": jnp.array([0, 2, 3, 0, 0, 5, 0]),
    "Grand Mesa": jnp.array([3, 3, 3, 0, 0, 0, 0]),
    "Old Faithful": jnp.array([0, 0, 0, 0, 0, 4, 4]),
    "Fountain of Youth": jnp.array([0, 0, 0, 0, 0, 0, 6]),
    "Great Barrier Reef": jnp.array([2, 1, 1, 0, 0, 2, 0]),
    "El Dorado": jnp.array([0, 0, 0, 0, 5, 0, 0]),
    "Solomon's Mines": jnp.array([0, 6, 0, 0, 0, 0, 0]),
    "Lake Victoria": jnp.array([6, 0, 2, 0, 0, 0, 0])
}

NW_YIELD_TABLE_IDX = [v for k, v in NW_YIELD_TABLE.items()]

# Each entry is a bool
# terrain = [ocean, grassland, plains, desert, tundra, snow]
# features = [None, forest, jungle, marsh, oasis, floodplains]
# elevation = [ocean, flatland, hill, mountain]
#{
#    "ocean_land": [],
#    "terrain": [],
#    "features": [],
#    "elevation": []
#}
NW_SPAWN_CRITERIA = {
    "Krakatoa": {
        "ocean_land": [1, 0],
        "terrain": [1, 0, 0, 0, 0, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [1, 0, 0, 0]
    },
    "Gibraltar": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 1, 1],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Kailash": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 0, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Kilimanjaro": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 0, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Sinai": {
        "ocean_land": [0, 1],
        "terrain": [0, 0, 1, 1, 0, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Sri Pada": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 0, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Cerro de Potosi": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Fuji": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 0, 0, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Uluru": {
        "ocean_land": [0, 1],
        "terrain": [0, 0, 1, 1, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Barringer Crater": {
        "ocean_land": [0, 1],
        "terrain": [0, 0, 0, 1, 1, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Grand Mesa": {
        "ocean_land": [0, 1],
        "terrain": [0, 0, 1, 1, 1, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Old Faithful": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 1, 0],
        "features": [1, 1, 0, 0, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Fountain of Youth": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 1, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Great Barrier Reef": {
        "ocean_land": [1, 0],
        "terrain": [1, 0, 0, 0, 0, 0],
        "features": [1, 0, 0, 0, 0, 0],
        "elevation": [1, 0, 0, 0]
    },
    "El Dorado": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 0, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Solomon's Mines": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 1, 1, 0],
        "features": [1, 1, 1, 1, 0, 0],
        "elevation": [0, 1, 1, 1]
    },
    "Lake Victoria": {
        "ocean_land": [0, 1],
        "terrain": [0, 1, 1, 0, 0, 0],
        "features": [1, 1, 1, 1, 1, 0],
        "elevation": [0, 1, 1, 1]
    }
}
