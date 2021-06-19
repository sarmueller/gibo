from typing import Optional, Dict, Callable, Union, Tuple

import torch

from src.environment_api import EnvironmentObjective
from src.optimizers import (
    RandomSearch,
    VanillaBayesianOptimization,
    BayesianGradientAscent,
)


def call_counter(func) -> Callable:
    """Decorate a function and "substitute" it function with a wrapper that
        count its calls.

    Args:
        func: Function which calls should be counted.

    Returns:
        Helper function which has attributes _calls and _func.
    """

    def helper(*args, **kwargs):
        helper._calls += 1
        return func(*args, **kwargs)

    helper._func = func
    helper._calls = 0
    return helper


def loop(
    params_init: torch.tensor,
    max_iterations: Optional[int],
    max_objective_calls: Optional[int],
    objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
    Optimizer: Union[RandomSearch, BayesianGradientAscent, VanillaBayesianOptimization],
    optimizer_config: Optional[Dict],
    verbose=True,
) -> Tuple[list, list]:
    """Connects parameters with objective and optimizer.

    Args:
        params_init:
        max_iterations: Stopping criterion for optimization after maximum of
            iterations (update steps of parameters).
        max_objective_calls: Stopping criterion for optimization after maximum
            of function calls of objective function.
        objective: Objective function to be optimized (search for maximum).
        Optimizer: One of the implemented optimizers.
        optimizer_config: Configuration dictionary for optimizer.
        verbose: If True an output is logged.

    Returns:
        Tuple of
            - list of parameters history
            - list of objective function calls in every iteration
    """
    calls_in_iteration = []
    objective_w_counter = call_counter(objective)
    optimizer = Optimizer(params_init, objective_w_counter, **optimizer_config)
    if max_iterations:
        for iteration in range(max_iterations):
            if verbose:
                print(f"--- Iteration {iteration+1} ---")
            optimizer()
            calls_in_iteration.append(objective_w_counter._calls)
    elif max_objective_calls:
        iteration = 0
        while objective_w_counter._calls < max_objective_calls:
            if verbose:
                print(
                    f"--- Iteration {iteration+1} ({objective_w_counter._calls} objective calls so far) ---"
                )
            optimizer()
            iteration += 1
            calls_in_iteration.append(objective_w_counter._calls)
    if verbose:
        print(
            f"\nObjective function was called {objective_w_counter._calls} times (sample complexity).\n"
        )
    return optimizer.params_history_list, calls_in_iteration
