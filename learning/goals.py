from __future__ import annotations
from flax import struct
from flax.core.frozen_dict import pop
import jax
from jax._src.dtypes import dtype
import jax.numpy as jnp
from typing import TYPE_CHECKING
from game.buildings import BLDG_IS_WORLD_WONDER

if TYPE_CHECKING:
    from game.primitives import GameState
    from learning.metrics import EpisodicMetrics




def compute_rewards(games: GameState, player_id: jnp.ndarray, episode_metrics: EpisodicMetrics) -> jnp.ndarray:
    """
    Seems like all many individually normed by "map score mod" then /= 100

    * Cities: https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L5946
        - number of cities * 6

    * Population: https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L5966
        - total population * 6

    * Land: https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L5986
        - total land (I think total tiles owned?) * 6 (based on comments in the code)

    * Wonders (NOT NORMED): https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L5998
        - number of wonders * 10

    * Policies: (NOT NORMED): https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L6014
         - number of policies * 8

    * Great works (NOT NORMED): https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L6034
        - number of great works * 1

    * Religion (NOT NORMED): https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L6050
         - return 0 if no religion OR only pantheon?
         - number of beliefs * 4
    
    * Techs (NOT NORMED): https://github.com/EnormousApplePie/Lekmod/blob/main/LEKMOD_DLL/CvGameCoreDLL_Expansion2/CvPlayer.cpp#L6078
        - number of techs * 10
    
    # TODO: look into diplomacy reward function? -- specifically, how they handle alliance rewards
    """
    games_index = jnp.arange(player_id.shape[0])

    city_reward = (games.player_cities.city_ids[games_index, player_id] > 0).sum(-1) * 6
    pop_reward = (games.player_cities.population[games_index, player_id]).sum(-1) * 6
    land_reward = (games.player_cities.ownership_map[games_index, player_id] >= 2).sum((-1, -2, -3)) * 6
    wonder_reward = (games.player_cities.buildings_owned[games_index, player_id] * BLDG_IS_WORLD_WONDER[None, None]).sum((-1, -2)) * 10
    policy_reward = (games.policies[games_index, player_id]).sum(-1) * 8
    gw_reward = (games.player_cities.gw_slots[games_index, player_id]).sum((-1, -2)) * 1
    rel_reward = (games.religious_tenets[games_index, player_id]).sum(-1) * 4
    tech_reward = ((games.technologies[games_index, player_id]).sum(-1) - 1) * 10
    
    new_score = (city_reward + pop_reward + land_reward + wonder_reward + policy_reward + gw_reward + rel_reward + tech_reward) / 10
    new_rewards, episode_metrics = episode_metrics.return_and_track_rewards(
        new_score, player_id
    )
    return new_rewards, episode_metrics
