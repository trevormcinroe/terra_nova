import argparse
import os
import pickle
import jax

from sim.build import build_simulator
from game.recorder import GameStateRecorder


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--save_filename", type=str)
parser.add_argument("--map_folder", type=str)
parser.add_argument("--device_idx", type=int)
parser.add_argument("--games_idx", type=int)
parser.add_argument("--distributed_strategy", type=str)
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

# Perhaps here you can initiailize your network and load your saved parameters via your custom code. 
# To initialize the recorder, we need to extract a single game from the bundle. This requires indexing 
# within *both* the device and game axis => games[device_idx][games_idx]
# This exact process will need to be repeated after each player takes its step within each game turn.
gamestate = jax.tree_map(lambda x: x[args.device_idx][args.games_idx], games)
recorder = GameStateRecorder.create(reference_gamestate=gamestate, num_steps=args.num_steps)
recorder = recorder.record(gamestate)


for recording_int in range(args.num_steps):
    for agent_step in range(6):
        # NOTE: replace the following random action sampling with whatever you like. E.g., the action sampling
        # process for your control policy.
        random_actions = games.sample_actions_uniformly(games.key[0, 0])
        games, obs_spaces, episode_metrics, players_turn_id, obs, rewards, done_flags, executed_actions = env_step_fn(
            games, random_actions, obs_spaces, episode_metrics, players_turn_id
        )

    gamestate = jax.tree_map(lambda x: x[args.device_idx][args.games_idx], games)
    recorder = recorder.record(gamestate)

recorder.save_replay(f"./renderer/saved_games/{args.save_filename}")
