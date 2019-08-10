from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two trainers (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse
import os
import gym
import json

import ray
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from cityflow_env import CityFlowEnvRay
from utility import parse_roadnet

import getpass
USERNAME = getpass.getuser()

parser = argparse.ArgumentParser()
# parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument('--config', type=str, default='/home/{}/workspace/Multi-Commander/config/global_config_multi.json'.format(USERNAME), help='config file')
parser.add_argument('--algo', type=str, default='QMIX', choices=['QMIX', 'APEX_QMIX'],
                    help='choose an algorithm')
parser.add_argument('--rollout', type=bool, default=False, help='rollout a policy')
parser.add_argument('--ckpt', type=str, default=r'/home/{}/ray_results/QMIX/QMIX_cityflow_multi_0_mixer=qmix_2019-08-09_13-14-1143atx77h/checkpoint_800/checkpoint-800'.format(USERNAME), help='checkpoint')
parser.add_argument('--epoch', type=int, default=20, help='number of training epochs')
parser.add_argument('--num_step', type=int, default=1500,help='number of timesteps for one episode, and for inference')
parser.add_argument('--save_freq', type=int, default=50, help='model saving frequency')
parser.add_argument('--batch_size', type=int, default=32, help='model saving frequency')
parser.add_argument('--phase_time', type=int, default=15, help='consistancy time of one phase')
parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1" # use GPU

def generate_config(args):
    with open(args.config) as f:
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
    config["rollout"] = args.rollout
    config["phase_time"] = args.phase_time
    # phase_list = config['lane_phase_info'][intersection_id]['phase']
    # logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    return config


if __name__ == "__main__":
    args = parser.parse_args()
    config = generate_config(args)

    ray.init()

    # Simple environment with 4 independent cartpole entities
    register_env("cityflow_multi", lambda config: CityFlowEnvRay(config))
    multi_env = CityFlowEnvRay(config)
    intersection_id = multi_env.intersection_id

    obs_space = multi_env.observation_space
    act_space = multi_env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {
        # "ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
        # "dqn_policy": (DQNTFPolicy, obs_space, act_space, {}),
        id_: (DQNTFPolicy, obs_space, act_space, {}) for id_ in intersection_id
    }

    def policy_mapping_fn(agent_id):
        # if agent_id % 2 == 0:
        #     return "ppo_policy"
        # else:
        #     return "dqn_policy"
        return agent_id

    # ppo_trainer = PPOTrainer(
    #     env="multi_cartpole",
    #     config={
    #         "multiagent": {
    #             "policies": policies,
    #             "policy_mapping_fn": policy_mapping_fn,
    #             "policies_to_train": ["ppo_policy"],
    #         },
    #         # disable filters, otherwise we would need to synchronize those
    #         # as well to the DQN agent
    #         "observation_filter": "NoFilter",
    #     })

    dqn_trainer = DQNTrainer(
        env="cityflow_multi",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": [id_ for id_ in intersection_id]
            },
            "gamma": 0.95,
            "n_step": 3,
            "env_config":config
        })

    # disable DQN exploration when used by the PPO trainer
    # ppo_trainer.workers.foreach_worker(
    #     lambda ev: ev.for_policy(
    #         lambda pi: pi.set_epsilon(0.0), policy_id="dqn_policy"))

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.epoch):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        if (i+1) % 100 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)

        # improve the PPO policy
        # print("-- PPO --")
        # print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        # dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        # ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))