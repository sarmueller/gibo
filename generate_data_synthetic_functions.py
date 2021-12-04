import os
import yaml
import argparse

import torch

from src.synthetic_functions import (
    generate_training_samples,
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate data for synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)

    train_x_dict = {}
    train_y_dict = {}
    lengthscales_dict = {}

    print(
        "Generate data (train_x, train_y, lengthscales, f_max, argmax) for synthetic test functions with domains of different dimensionality."
    )

    for dim in cfg_data["dimensions"]:
        print(f"Data for function with {dim}-dimensional domain.")
        l = get_lengthscales(dim, factor_hennig)
        m = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]),
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]),
        )
        lengthscale = m.sample((1, dim))
        train_x, train_y = generate_training_samples(
            num_objectives=cfg_data["num_objectives"],
            dim=dim,
            num_samples=cfg_data["num_samples"],
            gp_hypers={
                "covar_module.base_kernel.lengthscale": lengthscale,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers"]["outputscale"]
                ),
            },
        )
        train_x_dict[dim] = train_x
        train_y_dict[dim] = train_y
        lengthscales_dict[dim] = lengthscale

    print("Compute maxima and argmax of synthetic functions.")
    f_max_dict, argmax_dict = get_maxima_objectives(
        lengthscales=lengthscales_dict,
        noise_variance=cfg_data["noise_variance"],
        train_x=train_x_dict,
        train_y=train_y_dict,
        n_max=cfg_data["n_max"],
    )

    if not os.path.exists(cfg_data["out_dir"]):
        os.mkdir(cfg_data["out_dir"])

    path = cfg_data["out_dir"]
    print(f"Save data at {path}.")
    torch.save(train_x_dict, os.path.join(path, "train_x.pt"))
    torch.save(train_y_dict, os.path.join(path, "train_y.pt"))
    torch.save(lengthscales_dict, os.path.join(path, "lengthscales.pt"))
    torch.save(f_max_dict, os.path.join(path, "f_max.pt"))
    torch.save(argmax_dict, os.path.join(path, "argmax.pt"))
