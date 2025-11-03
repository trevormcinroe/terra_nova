from flax.struct import dataclass
import jax.numpy as jnp

from game.buildings import GameBuildings
from game.techs import Technologies
from game.constants import DIPLO_VICTORY_THRESHOLD, WC_MEETING_FREQ 

@dataclass
class TerraNovaEpisodeMetrics:
    culture_victory: jnp.ndarray
    diplomatic_victory: jnp.ndarray
    science_victory: jnp.ndarray
    domination_victory: jnp.ndarray
    in_episode: jnp.ndarray
    contains_an_episode: jnp.ndarray
    previous_step_rewards: jnp.ndarray
    
    @classmethod
    def create(cls, num_parallel_envs, num_episodes_to_track):
        """
        The build_simulator function will handle auto-distributing this to match up with 
        each of the parallel environments!
        """
        return cls(
            culture_victory=jnp.zeros(shape=(num_parallel_envs, num_episodes_to_track), dtype=jnp.uint8), 
            diplomatic_victory=jnp.zeros(shape=(num_parallel_envs, num_episodes_to_track), dtype=jnp.uint8), 
            science_victory=jnp.zeros(shape=(num_parallel_envs, num_episodes_to_track), dtype=jnp.uint8), 
            domination_victory=jnp.zeros(shape=(num_parallel_envs, num_episodes_to_track), dtype=jnp.uint8), 
            in_episode=jnp.zeros(shape=(num_parallel_envs,), dtype=jnp.uint8),
            contains_an_episode=jnp.zeros(shape=(num_parallel_envs, num_episodes_to_track), dtype=jnp.uint8),
            previous_step_rewards=jnp.zeros(shape=(num_parallel_envs, 6))
        )
    
    def return_and_track_rewards(self, new_rewards, player_id):
        games_index = jnp.arange(self.previous_step_rewards.shape[0])
        previous_rewards = self.previous_step_rewards[games_index, player_id]
        _self = self.replace(
            previous_step_rewards=self.previous_step_rewards.at[games_index, player_id].set(new_rewards)
        )
        return new_rewards - previous_rewards, _self
        

    def step_episode(self):
        """
        This is called from a per-game perspective
        """
        max_num_episodes = self.culture_victory.shape[0]
        next_episode_int = (self.in_episode + 1) % max_num_episodes
        return self.replace(
            in_episode=next_episode_int, 
            contains_an_episode=self.contains_an_episode.at[self.in_episode].set(1),
            previous_step_rewards=self.previous_step_rewards * 0
        )

    def track_end_of_episode(self, game):
        """
        +1 to player_id to map to winnder idx. This is to differentiate from 0s init
        """
        # Tourism victory
        self_mask = jnp.eye(N=6) == 1
        tourism_victory = (game.tourism_total > game.culture_total) | self_mask
        tourism_victory = tourism_victory.all(-1)
        is_tourism_victory = tourism_victory.any()
        tourism_victor = (tourism_victory.argmax() + 1) * is_tourism_victory

        # Diplomatic victory
        wc_tech_prereq = game.technologies[:, Technologies["printing_press"]._value_] == 1
        wc_meet_prereq = game.have_met[:, :6] == 1
        wc_meet_prereq = (wc_meet_prereq | self_mask).all(-1)
        
        wc_formed = (wc_tech_prereq & wc_meet_prereq).any()
        wc_meeting_freq = (game.current_step % WC_MEETING_FREQ == 0)[0]
        wc_is_meeting = wc_formed & wc_meeting_freq

        # (12, 6) => (6,)
        cs = (game.citystate_info.relationships == 2).sum(0)
        diplomatic_victory = (game.num_delegates + cs) >= DIPLO_VICTORY_THRESHOLD
        is_diplomatic_victory = diplomatic_victory.any() & wc_is_meeting
        diplomatic_victor = (diplomatic_victory.argmax() + 1) * is_diplomatic_victory
        
        # Science victory
        # (6, 5, ...) => (6,) for each
        science_victory = (
            (game.player_cities.buildings_owned[..., GameBuildings["booster_1"]._value_] == 1).any(-1) &
            (game.player_cities.buildings_owned[..., GameBuildings["booster_2"]._value_] == 1).any(-1) &
            (game.player_cities.buildings_owned[..., GameBuildings["booster_3"]._value_] == 1).any(-1) &
            (game.player_cities.buildings_owned[..., GameBuildings["engine"]._value_] == 1).any(-1) &
            (game.player_cities.buildings_owned[..., GameBuildings["cockpit"]._value_] == 1).any(-1) &
            (game.player_cities.buildings_owned[..., GameBuildings["stasis_chamber"]._value_] == 1).any(-1)
        )
        is_science_victory = science_victory.any()
        science_victor  = (science_victory.argmax() + 1) * is_science_victory

        # Domination victory
        warfare_victory = ((game.has_sacked == 1) | self_mask).all(-1)
        is_warfare_victory = warfare_victory.any()
        warfare_victor = (warfare_victory.argmax() + 1) * is_warfare_victory

        return self.replace(
           culture_victory=self.culture_victory.at[self.in_episode].set(tourism_victor),
           diplomatic_victory=self.diplomatic_victory.at[self.in_episode].set(diplomatic_victor),
           science_victory=self.science_victory.at[self.in_episode].set(science_victor),
           domination_victory=self.domination_victory.at[self.in_episode].set(warfare_victor),
        )
