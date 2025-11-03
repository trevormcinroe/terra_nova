from functools import partial
import jax.numpy as jnp
from flax import struct
import jax.tree_util as tree
import jax


@struct.dataclass
class Actions:
    trade_acceptdeny: jnp.ndarray
    trade_ask: jnp.ndarray
    trade_offer: jnp.ndarray
    trade_counterparty:  jnp.ndarray

    policy: jnp.ndarray
    tenet: jnp.ndarray
    tech: jnp.ndarray

    units_category: jnp.ndarray
    units_hex: jnp.ndarray
    
    city_pop: jnp.ndarray
    city_construction: jnp.ndarray

    @classmethod
    def create(cls, reference_array, traj_length):
        """
        These all begin (devices, 6, trajectory, games, ...)

        Actions: (1, 2) is (num_devices, num_games) 
            (1, 2, 6) (1, 2) (1, 2) (1, 2)
            (1, 2, 2)
            (1, 2, 3)
            (1, 2, 2)
            (1, 2, 30) (1, 2, 30)
            (1, 2, 10, 36) (1, 2, 10)
        """
        D = reference_array.shape[0]
        N = reference_array.shape[1]

        accept_deny = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 6), dtype=jnp.int32), reference_array.sharding)
        ask = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N), dtype=jnp.int32), reference_array.sharding)
        offer = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N), dtype=jnp.int32), reference_array.sharding)
        counterparty = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N), dtype=jnp.int32), reference_array.sharding)
        
        policies = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 2), dtype=jnp.int32), reference_array.sharding)
        tenets = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 3), dtype=jnp.int32), reference_array.sharding)
        techs = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 2), dtype=jnp.int32), reference_array.sharding)
        
        unit_cat = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 30), dtype=jnp.int32), reference_array.sharding)
        unit_hex = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 30), dtype=jnp.int32), reference_array.sharding)
        
        city_pop = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 10, 36), dtype=jnp.int32), reference_array.sharding)
        city_const = jax.device_put(jnp.zeros(shape=(D, 6, traj_length, N, 10), dtype=jnp.int32), reference_array.sharding)

        return cls(
            trade_acceptdeny=accept_deny,
            trade_ask=ask,
            trade_offer=offer,
            trade_counterparty=counterparty,
            policy=policies,
            tenet=tenets,
            tech=techs,
            units_category=unit_cat,
            units_hex=unit_hex,
            city_pop=city_pop,
            city_construction=city_const,
        )


@struct.dataclass
class Trajectories:
    """Trajectory holder that stores observations for all 6 players.
    """
    observations: any
    dones: jnp.ndarray  # Shape: (num_devices, 6, traj_length, num_games)
    actions: Actions 
    rewards: jnp.ndarray  # Shape: (num_devices, 6, traj_length, num_games)
    current_idx: jnp.ndarray  # Track where we are for each player in each game (shape: num_devices, 6, num_games)
    traj_length: int = 0  # Store trajectory length for modulo operation
    
    @classmethod
    def create(cls, reference_observation, traj_length):
        """Create a Trajectories holder from a reference observation.

        These all begin (devices, 6, trajectory, games, ...)
        
        Args:
            reference_observation: A flax dataclass observation with arrays having shape
                                 (num_devices, num_games, ...)
            traj_length: Length of trajectory to store
        
        Returns:
            Trajectories instance with observations expanded to include player and trajectory dimensions
        """
        def expand_array(x):
            if isinstance(x, jnp.ndarray):
                # Expand along axis 1 to add player dimension
                x_expanded = jnp.expand_dims(x, axis=1)  # (devices, 1, games, ...)

                # Expand along axis 2 to add trajectory dimension
                x_expanded = jnp.expand_dims(x_expanded, axis=2)  # (devices, 1, 1, games, ...)
                
                # Tile for both players and trajectory
                #x_tiled = jnp.tile(x_expanded, (1, 6, traj_length, *([1] * (x.ndim))))  # (devices, 6, traj_length, games, ...)
                x_tiled = jnp.tile(x_expanded, (1, 6, traj_length, *([1] * (x.ndim - 1))))
                return x_tiled * 0  # Convert to zeros while preserving sharding
            return x
        
        expanded_obs = tree.tree_map(expand_array, reference_observation)
        
        # Initialize per-player, per-game indices
        # Get a sample array from the reference observation to match its shape and sharding
        sample_array = reference_observation.player_id

        # Create dones array - boolean flags for episode termination
        dones_template = jnp.expand_dims(sample_array, axis=1)  # (devices, 1, games)
        dones_template = jnp.expand_dims(dones_template, axis=2)  # (devices, 1, 1, games)
        dones_template = jnp.tile(dones_template, (1, 6, traj_length, 1))  # (devices, 6, traj_length, games)
        dones = jnp.zeros_like(dones_template, dtype=jnp.bool_)

        # Create rewards array - scalar rewards per step
        rewards = jnp.zeros_like(dones_template, dtype=jnp.float32)
        actions = Actions.create(reference_observation.player_id, traj_length)

        # Expand template to include player dimension while preserving device sharding
        shard_template = jnp.expand_dims(sample_array, axis=1)  # (devices, 1, games)
        shard_template = jnp.tile(shard_template, (1, 6, 1))  # (devices, 6, games)

        # Create current_idx with same structure and sharding, but as zeros
        current_idx = jnp.zeros_like(shard_template, dtype=jnp.int32)

        return cls(
            observations=expanded_obs, 
            current_idx=current_idx, 
            dones=dones,
            actions=actions,
            rewards=rewards,
            traj_length=traj_length
        )
    
    @partial(jax.jit, donate_argnums=(0,))
    def add_data(self, players_turn_id, obs, executed_actions, rewards, dones):
        """Add observation data to the trajectory.
        
        Args:
            players_turn_id: Array of shape (num_devices, num_games) with values 0-5
            obs: Observation with same structure as reference_observation
        
        Returns:
            Updated Trajectories instance
        """
        # Create indices for the update
        device_idx = jnp.arange(players_turn_id.shape[0])[:, None]  # (num_devices, 1)
        game_idx = jnp.arange(players_turn_id.shape[1])[None, :]  # (1, num_games)
        
        # Get the correct trajectory index for each game based on which player it is
        traj_idx_per_game = self.current_idx[device_idx, players_turn_id, game_idx].squeeze()
        
        def update_array(traj_array, obs_array):
            if isinstance(obs_array, jnp.ndarray):
                # traj_array shape: (num_devices, 6, traj_len, num_games, ...)
                # obs_array shape: (num_devices, num_games, ...)
                # Update at [device, player_id, player's_current_idx, game]
                updated = traj_array.at[device_idx, players_turn_id, traj_idx_per_game, game_idx].set(obs_array)
                return updated
            return traj_array
        
        # Update all arrays in the observation tree - map over both trees together
        updated_obs = tree.tree_map(update_array, self.observations, obs)
        
        _update_lbda = lambda x, y: x.at[device_idx, players_turn_id, traj_idx_per_game, game_idx].set(y)

        updated_actions = self.actions.replace(
            trade_acceptdeny=_update_lbda(self.actions.trade_acceptdeny, executed_actions[0][0]),
            trade_ask=_update_lbda(self.actions.trade_ask, executed_actions[0][1]),
            trade_offer=_update_lbda(self.actions.trade_offer, executed_actions[0][2]),
            trade_counterparty=_update_lbda(self.actions.trade_counterparty, executed_actions[0][3]),
            policy=_update_lbda(self.actions.policy, executed_actions[1]),
            tenet=_update_lbda(self.actions.tenet, executed_actions[2]),
            tech=_update_lbda(self.actions.tech, executed_actions[3]),
            units_category=_update_lbda(self.actions.units_category, executed_actions[4][0]),
            units_hex=_update_lbda(self.actions.units_hex, executed_actions[4][1]),
            city_pop=_update_lbda(self.actions.city_pop, executed_actions[5][0]),
            city_construction=_update_lbda(self.actions.city_construction, executed_actions[5][1]),
        )
        updated_rewards = self.rewards.at[device_idx, players_turn_id, traj_idx_per_game, game_idx].set(rewards)

        updated_dones = self.dones.at[device_idx, players_turn_id, traj_idx_per_game, game_idx].set(1 - dones[..., 0])
        
        # Increment indices for the specific player in each game that was updated
        new_idx = self.current_idx.at[device_idx, players_turn_id, game_idx].set(
            (self.current_idx[device_idx, players_turn_id, game_idx] + 1) % self.traj_length
        )
        
        return self.replace(
            observations=updated_obs, 
            actions=updated_actions,
            rewards=updated_rewards,
            dones=updated_dones,
            current_idx=new_idx
        )
