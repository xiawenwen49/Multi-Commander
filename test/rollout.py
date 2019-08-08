from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import pickle
import gym
from gym.envs.registration import register
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts
from ray.tune.registry import register_env
import sys
sys.path.append('../')
from cityflow_env import CityFlowEnvRay
import logging
from utility import parse_roadnet


EXAMPLE_USAGE = """
    python rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --steps 100 --env CartPole-v0 --out rollouts.pkl
"""

parser = argparse.ArgumentParser()
parser.add_argument('--global_config', type=str, default='/home/leaves/workspace/Multi-Commander/config/global_config_multi.json', help='config file')
parser.add_argument('--run', type=str, default='QMIX', choices=['QMIX', 'APEX_QMIX'],
                    help='choose an algorithm')
parser.add_argument('--checkpoint', type=str, default=r'~/ray_results/QMIX/QMIX_cityflow_multi_0_mixer\=qmix_2019-08-07_18-25-5716112rhk/checkpoint_200/', help='checkpoint dir')
parser.add_argument('--ckpt_config', type=str, default='~/ray_results/QMIX/experiment_state-2019-08-07_18-25-57.json', help='checkpoint config')
parser.add_argument('--env', type=str, default='Cityflow-v0', help='environment')
parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
parser.add_argument('--num_step', type=int, default=1500,help='number of timesteps for one episode, and for inference')
parser.add_argument('--save_freq', type=int, default=50, help='model saving frequency')
parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1" # use GPU


def generate_config(args):
    with open(args.global_config) as f:
        config = json.load(f)
    with open(config['cityflow_config_file']) as f:
        cityflow_config = json.load(f)
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["num_step"] = args.num_step
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    intersection_id = list(config['lane_phase_info'].keys())
    config["intersection_id"] = intersection_id
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span
    config["thread_num"] = 1
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span
    # phase_list = config['lane_phase_info'][intersection_id]['phase']
    # logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    return config


def run(args, config_env):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.ckpt_config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
#    config = merge_dicts(config, config_env)

    ray.init()
    cls = get_agent_class(args.run)
    register_env("Cityflow-v0", lambda config: CityFlowEnvRay(config))
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    # env = CityflowGymEnv(config_env)
    env = CityFlowEnvRay(config_env)

    num_step = int(args.num_step)

    if args.out is not None:
        rollouts = []
    steps = 0
    while steps < (num_step or steps + 1):
        if args.out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_step or steps + 1):
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        if args.out is not None:
            rollouts.append(rollout)
        print('Episode reward', reward_total)
    if args.out is not None:
        pickle.dump(rollouts, open(args.out, 'wb'))





if __name__ == "__main__":
    args = parser.parse_args()
    config_env = generate_config(args)

    run(args, config_env)
