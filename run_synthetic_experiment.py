import os
import argparse
import yaml

import numpy as np
import torch

from src import config
from src.loop import loop
from src.synthetic_functions import (
    generate_objective_from_gp_post,
    compute_rewards,
    get_lengthscale_hyperprior,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run optimization of synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")
    parser.add_argument(
        "-cd", "--config_data", type=str, help="Path to data config file."
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Translate config dictionary.
    cfg = config.insert(cfg, config.insertion_config)

    with open(args.config_data, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)

    f_max_dict = torch.load(os.path.join(cfg_data["out_dir"], "f_max.pt"))
    train_x_dict = torch.load(os.path.join(cfg_data["out_dir"], "train_x.pt"))
    train_y_dict = torch.load(os.path.join(cfg_data["out_dir"], "train_y.pt"))
    lengthscales_dict = torch.load(os.path.join(cfg_data["out_dir"], "lengthscales.pt"))

    params_dict = {}
    calls_dict = {}
    rewards_dict = {}

    for dim in cfg_data["dimensions"]:
        print(f"\nDimension {dim}.")

        params_list = []
        rewards_list = []
        calls_list = []

        for index_objective in range(cfg_data["num_objectives"]):
            print(f"\nObjective {index_objective+1}.")
            objective = generate_objective_from_gp_post(
                train_x_dict[dim][index_objective],
                train_y_dict[dim][index_objective],
                noise_variance=cfg_data["noise_variance"],
                gp_hypers={
                    "covar_module.base_kernel.lengthscale": lengthscales_dict[dim],
                    "covar_module.outputscale": torch.tensor(
                        cfg_data["gp_hypers"]["outputscale"]
                    ),
                },
            )
            print(f"Max of objective: {f_max_dict[dim][index_objective]}.")

            hypers = None
            if "set_hypers" in cfg.keys():
                if cfg["set_hypers"]:
                    hypers = {
                        "covar_module.base_kernel.lengthscale": lengthscales_dict[dim],
                        "covar_module.outputscale": torch.tensor(
                            cfg_data["gp_hypers"]["outputscale"]
                        ),
                        "likelihood.noise": torch.tensor(cfg_data["noise_variance"]),
                    }
            elif "only_set_noise_hyper" in cfg.keys():
                if cfg["only_set_noise_hyper"]:
                    hypers = {
                        "likelihood.noise": torch.tensor(cfg_data["noise_variance"])
                    }

            cfg_dim = config.evaluate(
                cfg,
                dim_search_space=dim,
                factor_lengthscale=cfg_data["factor_lengthscale"],
                factor_N_max=5,
                hypers=hypers,
            )

            params, calls_in_iteration = loop(
                params_init=0.5 * (torch.ones(dim, dtype=torch.float32)),
                max_iterations=cfg_dim["max_iterations"],
                max_objective_calls=cfg_dim["max_objective_calls"],
                objective=objective,
                Optimizer=cfg_dim["method"],
                optimizer_config=cfg_dim["optimizer_config"],
                verbose=False,
            )

            rewards = compute_rewards(params, objective)
            print(f"Optimizer's max reward: {max(rewards)}")
            params_list.append(params)
            calls_list.append(calls_in_iteration)
            rewards_list.append(rewards)

        params_dict[dim] = params_list
        calls_dict[dim] = calls_list
        rewards_dict[dim] = rewards_list

    directory = cfg["out_dir"]
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(
        f"Save parameters, objective calls and rewards (function values) at {directory}."
    )
    np.save(os.path.join(directory, "parameters"), params_dict)
    np.save(os.path.join(directory, "calls"), calls_dict)
    np.save(os.path.join(directory, "rewards"), rewards_dict)
