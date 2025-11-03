from __future__ import annotations
import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING
from game.buildings import GameBuildings
from game.constants import DIPLO_VICTORY_THRESHOLD, WC_MEETING_FREQ
from game.techs import Technologies 

if TYPE_CHECKING:
    from game.primitives import GameState
    from learning.obs_spaces import ObservationSpace
    from learning.metrics import EpisodicMetrics


def reset_episode(games: GameState, obs_space: ObservationSpace, episode_metrics: EpisodicMetrics, player_id: jnp.ndarray):
    """
    Called in a vmap-over-games context
    """
    step_bool = (games.current_step >= 300) & (player_id == 5)
    
    # tourism total > culture total
    # Self-comparisons do not matter
    self_mask = jnp.eye(N=6) == 1
    tourism_victory = (games.tourism_total > games.culture_total) | self_mask
    tourism_victory = tourism_victory.all(-1).any()

    # Diplomatic victory
    # The world congress begins when (1) someone has printing press and (2) someone 
    # has met everyone else
    # (num_games,)
    # Need to be careful on has met. Players default meet self.
    wc_tech_prereq = games.technologies[:, Technologies["printing_press"]._value_] == 1
    wc_meet_prereq = games.have_met[:, :6] == 1
    wc_meet_prereq = (wc_meet_prereq | self_mask).all(-1)
    
    wc_formed = (wc_tech_prereq & wc_meet_prereq).any()
    wc_meeting_freq = (games.current_step % WC_MEETING_FREQ == 0)[0]
    wc_is_meeting = wc_formed & wc_meeting_freq

    # (12, 6) => (6,)
    cs = (games.citystate_info.relationships == 2).sum(0)
    diplomacy_victory = ((games.num_delegates + cs) >= DIPLO_VICTORY_THRESHOLD).any() & wc_is_meeting

    # (6, 5, ...) => (6,) for each
    # When added together, ==6, then sum. If > 0, then someone won!
    science_victory = (
        (games.player_cities.buildings_owned[..., GameBuildings["booster_1"]._value_] == 1).any(-1) &
        (games.player_cities.buildings_owned[..., GameBuildings["booster_2"]._value_] == 1).any(-1) &
        (games.player_cities.buildings_owned[..., GameBuildings["booster_3"]._value_] == 1).any(-1) &
        (games.player_cities.buildings_owned[..., GameBuildings["engine"]._value_] == 1).any(-1) &
        (games.player_cities.buildings_owned[..., GameBuildings["cockpit"]._value_] == 1).any(-1) &
        (games.player_cities.buildings_owned[..., GameBuildings["stasis_chamber"]._value_] == 1).any(-1)
    )
    science_victory = science_victory.any()

    # (6, 6) => (6,)
    warfare_victory = ((games.has_sacked == 1) | self_mask).all(-1).any()
    return step_bool | tourism_victory | diplomacy_victory | science_victory | warfare_victory
