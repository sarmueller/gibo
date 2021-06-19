from typing import Dict, Optional
import copy

import torch
import botorch
import gpytorch

from src.optimizers import (
    RandomSearch,
    CMAES,
    VanillaBayesianOptimization,
    BayesianGradientAscent,
)
from src.model import ExactGPSEModel, DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_vanilla_bo, optimize_acqf_custom_bo
from src.synthetic_functions import (
    get_lengthscales,
    factor_hennig,
    get_lengthscale_hyperprior,
)

# Dictionaries for translating config file.
# Note: can be extended
prior_dict = {
    "prior": {
        "normal": gpytorch.priors.NormalPrior,
        "gamma": gpytorch.priors.GammaPrior,
        "uniform": gpytorch.priors.UniformPrior,
        "get_lengthscale_hyperprior": get_lengthscale_hyperprior,
    }
}
constraint_dict = {
    "constraint": {
        "greather_than": gpytorch.constraints.GreaterThan,
        "less_than": gpytorch.constraints.LessThan,
        "interval": gpytorch.constraints.Interval,
        "positive": gpytorch.constraints.Positive,
    }
}
insertion_config = {
    "method": {
        "gibo": BayesianGradientAscent,
        "vbo": VanillaBayesianOptimization,
        "rs": RandomSearch,
        "cmaes": CMAES,
    },
    "optimizer_config": {
        "OptimizerTorch": {"sgd": torch.optim.SGD, "adam": torch.optim.Adam},
        "Model": {
            "derivative_gp": DerivativeExactGPSEModel,
            "plain_gp": ExactGPSEModel,
        },
        "model_config": {
            "lengthscale_constraint": constraint_dict,
            "lengthscale_hyperprior": prior_dict,
            "outputscale_constraint": constraint_dict,
            "outputscale_hyperprior": prior_dict,
            "noise_constraint": constraint_dict,
            "noise_hyperprior": prior_dict,
        },
        "optimize_acqf": {
            "bga": optimize_acqf_custom_bo,
            "vbo": optimize_acqf_vanilla_bo,
        },
        "acquisition_function": {
            "expected_improvement": botorch.acquisition.analytic.ExpectedImprovement
        },
    },
}


def insert_(dict1: dict, dict2: dict):
    """Insert dict2 into dict1.

    Caution: dict1 is manipulated!

    Args:
        dict1: Dictionary which is manipulated.
        dict2: Dictionary from which insertion information is collected.
    """
    for key, value in dict1.items():
        if key in dict2.keys():
            if isinstance(value, dict):
                insert_(value, dict2[key])
            elif value is not None:
                dict1[key] = dict2[key][value]


def insert(dict1: dict, dict2: dict):
    """Insert dict2 into dict1.

    Args:
        dict1: Dictionary which is manipulated.
        dict2: Dictionary from which insertion information is collected.

    Return:
        Copied and manipulated version of dict1.
    """
    manipulated_dict1 = copy.deepcopy(dict1)
    insert_(manipulated_dict1, dict2)

    return manipulated_dict1


def evaluate_(
    config: dict,
    dim_search_space: int,
    factor_lengthscale: Optional[int] = None,
    factor_N_max: Optional[int] = None,
    hypers: Optional[Dict] = None,
):
    """Evaluate functions- and dim_search_space-related entries in config.

    Caution: config is manipulated!

    Args:
        config: Dictionary with evaluated entries.
        dim_search_space: Dimension of search space.
        factor_lengthscale: Factor for setting lengthscale.
        factor_N_max: Factor for setting N_max in bga.
        hypers: Dict for fixing GP hyperparameters.
    """
    search_constraints = [
        "lengthscale_constraint",
        "outputscale_constraint",
        "noise_constraint",
    ]
    search_priors = [
        "lengthscale_hyperprior",
        "outputscale_hyperprior",
        "noise_hyperprior",
    ]

    for key, value in config.items():
        if key in search_constraints:
            if value["constraint"] is not None:
                config[key] = value["constraint"](**value["kwargs"])
            else:
                config[key] = None
        elif key in search_priors:
            if (value["prior"] is not None) and ("dim" in value["kwargs"].keys()):
                if value["kwargs"]["dim"] == "dim_search_space":
                    value["kwargs"]["dim"] = dim_search_space
                config[key] = value["prior"](**value["kwargs"])
            elif value["prior"] is not None:
                config[key] = value["prior"](**value["kwargs"])
            else:
                config[key] = None
        elif (key == "ard_num_dims") and (value == "dim_search_space"):
            config[key] = dim_search_space
        elif key == "bounds":
            if (value["lower_bound"] is not None) and (
                value["upper_bound"] is not None
            ):
                config[key] = torch.tensor(
                    [
                        [value["lower_bound"]] * dim_search_space,
                        [value["upper_bound"]] * dim_search_space,
                    ]
                )
            else:
                config[key] = None
        elif (key == "samples_per_iteration") and (value == "variable"):
            config[key] = 1 + dim_search_space // 8
        elif (key == "exploration_noise") and (value == "variable"):
            config[key] = (
                0.1
                * factor_lengthscale
                * get_lengthscales(dim_search_space, factor_hennig)
            )
        elif (
            (key == "set_hypers") or (key == "only_set_noise_hyper") and (value == True)
        ):
            config["optimizer_config"]["hyperparameter_config"]["hypers"] = hypers
        elif (key == "N_max") and (value == "variable"):
            config[key] = factor_N_max * dim_search_space
        elif (key == "max_samples_per_iteration") and (value == "dim_search_space"):
            config[key] = dim_search_space
        elif (key == "sigma") and (value == "variable"):
            config[key] = 0.3 * get_lengthscales(dim_search_space, factor_hennig)
        elif isinstance(value, dict):
            evaluate_(value, dim_search_space, factor_lengthscale, factor_N_max, hypers)


def evaluate(
    config: dict,
    dim_search_space: int,
    factor_lengthscale: Optional[int] = None,
    factor_N_max: Optional[int] = None,
    hypers: Optional[Dict] = None,
):
    """Evaluate functions- and dim_search_space-related entries in config.

    Args:
         config: Dictionary with evaluated entries.
         dim_search_space: Dimension of search space.
         factor_lengthscale: Factor for setting lengthscale.
         factor_N_max: Factor for setting N_max in bga.
         hypers: Dict for fixing GP hyperparameters.

     Return:
         Evaluated config.
    """
    copied_config = copy.deepcopy(config)
    evaluate_(
        config=copied_config,
        dim_search_space=dim_search_space,
        factor_lengthscale=factor_lengthscale,
        factor_N_max=factor_N_max,
        hypers=hypers,
    )
    return copied_config
