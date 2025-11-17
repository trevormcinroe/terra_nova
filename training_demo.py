import argparse
import os
import pickle
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
import optax
from functools import partial

from sim.build import build_simulator
from learning.memories import Trajectories
from learning.networks import make_terra_nova_network


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--map_folder", type=str)
parser.add_argument("--distributed_strategy", type=str)
parser.add_argument("--memory_length", type=int, default=1)
args = parser.parse_args()

all_maps = os.listdir(args.map_folder)
print(all_maps)

games = []

for game in all_maps:
    if ".gamestate" not in game:
        continue
    with open(f"{args.map_folder}/{game}", "rb") as f:
        gamestate = pickle.load(f)
    games.append(gamestate)


env_step_fn, games, obs_spaces, episode_metrics, players_turn_id, obs, GLOBAL_MESH, sharding = build_simulator(
    games, 
    args.distributed_strategy,
    jax.random.PRNGKey(args.seed),
)

trajectories = Trajectories.create(obs, args.memory_length)

pi_v = make_terra_nova_network(me_n_pma_seeds=16)
variables = pi_v.init({"params": jax.random.PRNGKey(args.seed)}, jax.tree.map(lambda x: x[0], obs), False)
params = variables["params"]

tx = optax.adam(learning_rate=1e-4)
opt_state = tx.init(params)

params = jax.tree.map(
    lambda x: jax.make_array_from_single_device_arrays(
        (len(GLOBAL_MESH.devices),) + x.shape,
        sharding,
        [
            jax.device_put(x[None], device)
            for device in GLOBAL_MESH.devices
        ],
    ), 
    params
)


gpu_axis = GLOBAL_MESH.axis_names[0]

@jax.jit
@partial(
    shard_map, 
    mesh=GLOBAL_MESH, 
    in_specs=PartitionSpec(gpu_axis, gpu_axis),
    out_specs=PartitionSpec(
        (
            (gpu_axis, gpu_axis, gpu_axis, gpu_axis),
            gpu_axis,
            gpu_axis,
            gpu_axis,
            (gpu_axis, gpu_axis),
            (gpu_axis, gpu_axis)
        ),
        gpu_axis
    )
)
def forward_pass_distributed(params, obs):
    params = jax.tree.map(lambda x: x[0], params)
    obs = jax.tree.map(lambda x: x[0], obs)

    actions, values = pi_v.apply({"params": params}, obs, training=False)
    
    actions = jax.tree.map(lambda x: x[None], actions)
    values = jax.tree.map(lambda x: x[None], values)
    return actions, values

actions, values = forward_pass_distributed(params, obs)

# Perhaps here you can initiailize your network and load your saved parameters via your custom code. 
# You can use one of the arrays from `trajectories` as your sharding reference for the parameters
# of your network.

for recording_int in range(args.num_steps):
    for agent_step in range(6):
        # NOTE: replace the following random action sampling with whatever you like. E.g., the action sampling
        # process for your control policy.
        random_actions = games.sample_actions_uniformly(games.key[0, 0])

        print("Network:")
        test = actions
        print("Trade:\n\tResponse: {test[0][0].shape}\n\tAsk: {test[0][1].shape}\n\tOffer: {test[0][2].shape}\n\tCounterparty: {test[0][3].shape}")
        print(f"Social policy: {test[1].shape}")
        print(f"Religion: {test[2].shape}")
        print(f"Tech: {test[3].shape}")
        print(f"Units:\n\tCat: {test[4][0].shape}\n\tMap: {test[4][1].shape}")
        print(f"City:\n\tPop: {test[5][0].shape}\n\tBldgs: {test[5][1].shape}")
        
        print("Primitive:")
        test = random_actions
        print("Trade:\n\tResponse: {test[0][0].shape}\n\tAsk: {test[0][1].shape}\n\tOffer: {test[0][2].shape}\n\tCounterparty: {test[0][3].shape}")
        print(f"Social policy: {test[1].shape}")
        print(f"Religion: {test[2].shape}")
        print(f"Tech: {test[3].shape}")
        print(f"Units:\n\tCat: {test[4][0].shape}\n\tMap: {test[4][1].shape}")
        print(f"City:\n\tPop: {test[5][0].shape}\n\tBldgs: {test[5][1].shape}")
        qqq
        
        games, obs_spaces, episode_metrics, new_players_turn_id, next_obs, rewards, done_flags, selected_actions = env_step_fn(
            games, random_actions, obs_spaces, episode_metrics, players_turn_id
        )
        
        trajectories = trajectories.add_data(players_turn_id, obs, selected_actions, rewards, done_flags)
        
        # We defer the overwrite of obs/player_turn values until *after* information is inserted into the memory
        players_turn_id = new_players_turn_id
        obs = next_obs
