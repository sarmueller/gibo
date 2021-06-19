from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List
import copy

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from src.environment_api import EnvironmentObjective


def offline_reward_evaluation(
    parameters: Dict[int, np.array],
    objective_env: EnvironmentObjective,
    verbose: bool = False,
) -> Dict[int, List[int]]:
    """Evaluate offline the parameter performance for a given environment.

    Args:
        parameters: Parameter instances to evaluate.
        objective_env: Objective environment to evaluate parameters.
        verbose: If True an output is logged.

    Returns:
        Dictionary mapping every parameter to a reward (over multiple trials/new starts).
    """
    rewards = {}
    for trial in parameters.keys():
        rewards_trial = []
        for params in parameters[trial]:
            rewards_trial.append(
                objective_env.test_params(
                    params=torch.tensor(params),
                    episodes=1,
                    render=False,
                    verbose=verbose,
                ).item()
            )
        rewards[trial] = rewards_trial
    return rewards


def postprocessing_interpolate_x(
    x: Dict[int, List], calls: Dict[int, List], max_calls: int
) -> Tensor:
    """Interpolates between given rewards at objective calls.

    Having a reward for every integer between [0, max_calls).
    By reflecting the given reward to the right until a new reward value is given.

    Args:
        x: Dictionary with rewards.
        calls: Number of objective calls at every reward.
        max_calls: Maximum number of calls (x-axis) until you want to interpolate.

    Returns:
        Interpolated rewards over objective calls
    """
    runs = list(x.keys())
    interpolation = torch.empty((len(runs), max_calls))
    for index_run, _ in enumerate(runs):
        index_x = 0
        for num_call in range(max_calls):
            if num_call == calls[index_run][index_x]:
                index_x += 1
            interpolation[index_run][num_call] = x[index_run][index_x]
    return interpolation


def plot_rewards_over_calls(
    rewards_optimizer: List[torch.tensor],
    names_optimizer: List[str],
    title: str,
    marker: List[str] = ["o", ">"],
    steps: int = 100,
    markevery: int = 5,
    path_savefig: Optional[str] = None,
):
    """Generate plot showing rewards over objective calls for multiple optimizer.

    Args:
        rewards_optimizer: List of torch tensors for every optimizer.
        title: Plot title.
        marker: Plot marker.
        steps: Number which defines the x-th reward that should be plotted.
        markevery: Number which defines the x-th reward which should be marked (after steps).
        path_savefig: Path where to save the resulting figure.
    """
    for index_optimizer, rewards in enumerate(rewards_optimizer):
        max_calls = rewards.shape[-1]
        m = torch.mean(rewards, dim=0)[::steps]
        std = torch.std(rewards, dim=0)[::steps]
        plt.plot(
            torch.linspace(0, max_calls, max_calls // steps),
            m,
            marker=marker[index_optimizer],
            markevery=markevery,
            label=names_optimizer[index_optimizer],
        )
        plt.fill_between(
            torch.linspace(0, max_calls, max_calls // steps),
            m - std,
            m + std,
            alpha=0.2,
        )
    plt.xlabel("# of evaluations")
    plt.ylabel("Average Reward")
    plt.legend(loc="lower right")
    plt.title(title)
    if path_savefig:
        plt.savefig(path_savefig, bbox_inches="tight")


def sort_rewards_global_optimization(rewards: dict) -> dict:
    """Sort out rewards that are smaller than last max (smoothing).

    Args:
        rewards: Dictionary of reward values, mapping dimension and number of
            objective to a list of rewards.

    Returns:
        Sorted rewards.
    """
    rewards_sorted = copy.deepcopy(rewards)
    for dim in rewards.keys():
        for num_objective, rewards_objective in enumerate(rewards[dim]):
            rewards_sorted[dim][num_objective] = sort_rewards_helper(rewards_objective)
    return rewards_sorted


def sort_rewards_helper(rewards: list) -> list:
    """Helper function for 'sort_rewards_global_optimization.'

    Args:
        rewards: List of rewards.

    Returns:
        Sorted rewards.
    """
    rewards_new = []
    max_r = -np.inf
    for r in rewards:
        if r > max_r:
            rewards_new.append(r)
            max_r = r
        else:
            rewards_new.append(max_r)
    return rewards_new


def postprocessing_interpolation_rewards(
    rewards: Dict, calls: Dict, calls_of_objective: int
):
    """Interpolates between given rewards at objective calls.

    Similar to 'postprocessing_interpolation_x', but with an additional
    dimension of 'num_objectives'.

    Args:
        rewards: Dictionary of reward values, mapping dimension and number of
            objective to a list of rewards.
        calls: Dictionary of number of objective calls, mapping dimension and
            number of objective to a list of calls.
        calls_of_objective: Maximum number of calls (x-axis) until you want to
        interpolate.

    Returns:
        Interpolated rewards over objective calls
    """
    dimensions = list(rewards.keys())
    num_objectives = len(rewards[dimensions[0]])
    new_rewards = torch.empty((len(dimensions), num_objectives, calls_of_objective))
    for index_dim, dim in enumerate(dimensions):
        list_rewards = rewards[dim]
        list_calls = calls[dim]
        for num_obj, _ in enumerate(list_rewards):
            index_rewards = 0
            for call in range(calls_of_objective):
                if call == list_calls[num_obj][index_rewards]:
                    index_rewards += 1
                new_rewards[index_dim][num_obj][call] = list_rewards[num_obj][
                    index_rewards
                ]
    return new_rewards


def f_max_new(f_max_old: dict, list_rewards_optimizers: list):
    """Correct f_max if one of the optimizers found larger values.

    Args:
        f_max: Dictionary of old f_max.
        list_rewards_optimizers: List of competing optimizers.

    Returns:
        New f_max dictionary.
    """
    f_max_new = copy.deepcopy(f_max_old)
    dimensions = list(f_max_old.keys())
    for rewards_optimizer in list_rewards_optimizers:
        for dim_index, dim in enumerate(dimensions):
            global_max = torch.tensor(f_max_new[dim])
            max_rewards = torch.max(rewards_optimizer[dim_index], axis=-1).values
            for i, g_m in enumerate(global_max):
                if g_m < max_rewards[i]:
                    f_max_new[dim][i] += -g_m.item() + max_rewards[i].item() + 1e-2
    return f_max_new
