import os
import argparse
import yaml

import numpy as np
import torch
import gym

from src.environment_api import EnvironmentObjective, manipulate_reward
from src.policy_parameterizations import MLP
from src.evaluate import (
    offline_reward_evaluation,
    postprocessing_interpolate_x,
    plot_rewards_over_calls,
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate optimization methods and save figure (reward over objective calls).'
    )
    parser.add_argument('-post', '--postprocess', type=bool, default=True)
    parser.add_argument(
        '-path',
        '--path_savefig',
        type=str,
        help='Path where to save evaluation figure.',
    )
    parser.add_argument(
        '-title',
        '--figure_title',
        type=str,
        default='Evaluation Image',
        help='Title of evaluation figure mapping reward over objective calls.',
    )
    parser.add_argument('-marker', '--figure_marker', default=['o', '>'])
    parser.add_argument('-steps', '--figure_steps', type=int, default=1)
    parser.add_argument('-markevery', '--figure_markevery', type=int, default=5)
    parser.add_argument(
        '-cs',
        '--configs',
        type=str,
        help='List of paths to config files of optimization methods to evaluate.',
        nargs='*',
    )

    args = parser.parse_args()
    method_to_name = {'bga': 'GIBO', 'rs': 'ARS', 'vbo': 'Vanilla BO'}
    list_interpolated_rewards = []
    list_names_optimizer = []

    for cfg_str in args.configs:

        with open(cfg_str, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if args.postprocess:

            print('Postprocess tracked parameters over optimization procedure.')

            # Usecase 1: optimizing policy for a reinforcement learning environment.
            mlp = MLP(*cfg['mlp']['layers'], add_bias=cfg['mlp']['add_bias'])
            len_params = mlp.len_params

            # In evaluation mode manipulation of state and reward is always None.
            objective_env = EnvironmentObjective(
                env=gym.make(cfg['environment_name']),
                policy=mlp,
                manipulate_state=None,
                manipulate_reward=None,
            )

            # Load data.
            directory = cfg['out_dir']
            print(f'Load data from {directory}.')
            parameters = np.load(
                os.path.join(directory, 'parameters.npy'), allow_pickle=True
            ).item()
            calls = np.load(
                os.path.join(directory, 'calls.npy'), allow_pickle=True
            ).item()

            # Postprocess data (offline evaluation and interpolation).
            print('Postprocess data: offline evaluation and interpolation.')
            offline_rewards = offline_reward_evaluation(parameters, objective_env)
            interpolated_rewards = postprocessing_interpolate_x(
                offline_rewards, calls, max_calls=cfg['max_objective_calls']
            )

            # Save postprocessed data.
            print(f'Save postprocessed data in {directory}')
            torch.save(
                interpolated_rewards, os.path.join(directory, 'interpolated_rewards.pt')
            )
            torch.save(offline_rewards, os.path.join(directory, 'offline_rewards.pt'))

        else:
            interpolated_rewards = torch.load(
                os.path.join(cfg['out_dir'], 'interpolated_rewards.pt')
            )

        list_names_optimizer.append(method_to_name[cfg['method']])
        list_interpolated_rewards.append(interpolated_rewards)

    print('Generate figure.')

    plot_rewards_over_calls(
        rewards_optimizer=list_interpolated_rewards,
        names_optimizer=list_names_optimizer,
        title=args.figure_title,
        marker=args.figure_marker,
        steps=args.figure_steps,
        markevery=args.figure_markevery,
        path_savefig=args.path_savefig,
    )

    print(f'Save figure in {args.path_savefig}.')