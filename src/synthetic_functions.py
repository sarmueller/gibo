from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

import numpy as np
import torch
from torch import Tensor
import botorch
import gpytorch


delta_n = lambda n: ((1 / 6) * n) ** (1 / 2) * (
    1 / 3 * (1 + 2 * (1 - 3 / (5 * n)) ** (1 / 2))
) ** (1 / 2)
delta_n_lower_bound = lambda n: 1 / 3 * n ** (1 / 2)
get_lengthscales = lambda n, factor: delta_n(n) * factor
factor_hennig = 0.1 / delta_n(2)


def irregular_grid(dim: int) -> Tensor:
    """Generate irragular grid with a quasirandom sobol sampler.

    Args:
        dim: Dimensions of grid.

    Return:
        Grid sample.
    """
    soboleng = torch.quasirandom.SobolEngine(dimension=dim)

    def sample(number: int):
        return soboleng.draw(number)

    return sample


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model with constant mean and SE-kernel.

    Attributes:
        train_x: The training features X.
        train_y: The training targets y.
        likelihood: The model's likelihood.
    """

    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        """Inits the model."""
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Args:
            x: The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def sample_from_gp_prior(
    dim: int,
    num_samples: int = 1000,
    gp_hypers: Dict[str, Tensor] = {
        "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
        "covar_module.outputscale": torch.tensor(1.0),
    },
) -> Tuple[Tensor, Tensor]:
    """Sample random points from gp prior.

    Args:
        dim: Dimension of sample grid for train_x data.
        num_samples: Number of train_x samples.
        gp_hypers: GP model hyperparameters.

    Returns:
        Trainings features and targets.
    """

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(None, None, likelihood, ard_num_dims=dim)
    model.initialize(**gp_hypers)
    train_x = irregular_grid(dim)(num_samples)
    train_y = sample_from_gp_prior_helper(model)(train_x)
    return train_x, train_y


def sample_from_gp_prior_helper(model) -> Tensor:
    """Helper function to sample from GP prior model.

    Args:
        model: GP prior model.

    Returns:
        GP prior sample.
    """

    def sample(x):
        model.train(False)
        with gpytorch.settings.prior_mode(True):
            mvn = model(x)
        return mvn.sample().flatten()

    return sample


def generate_objective_from_gp_post(
    train_x: Tensor,
    train_y: Tensor,
    noise_variance: float = 1e-6,
    gp_hypers: Dict[str, Tensor] = {
        "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
        "covar_module.outputscale": torch.tensor(1.0),
    },
) -> Callable[[Tensor], float]:
    """Generate objective function with given train_x, train_y and hyperparameters.

    Args:
        train_x: The training features X.
        train_y: The training targets y.
        noise_variance: Observation noise.
        gp_hypers: GP model hyperparameters.

    Returns:
        Objective function.
    """

    dim = train_x.shape[-1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(None, None, likelihood, ard_num_dims=dim)
    model.initialize(**gp_hypers)
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    def objective(x, observation_noise=True, requires_grad=False):
        model.eval()
        with gpytorch.settings.fast_pred_var():
            posterior = model(x)
            m = posterior.mean.flatten()
            if observation_noise:
                m += torch.randn_like(m) * np.sqrt(noise_variance)
        if requires_grad:
            return m
        else:
            return m.detach()

    return objective


def generate_training_samples(
    num_objectives: int, dim: int, num_samples: int, gp_hypers: Dict[str, Tensor]
) -> Tuple[List[Tensor], List[Tensor]]:
    """Generate training samples for `num_objectives` objectives.

    Args:
        num_objectives: Number of objectives.
        dim: Dimension of parameter space/sample grid.
        num_samples: Number of grid samples.
        gp_hypers: GP model hyperparameters.

    Returns:
        List of trainings features and targets.
    """

    train_x = []
    train_y = []
    for _ in range(num_objectives):
        x, y = sample_from_gp_prior(dim, num_samples, gp_hypers)
        train_x.append(x)
        train_y.append(y)
    return train_x, train_y


def compute_rewards(
    params: Tensor, objective: Callable[[Tensor], float], verbose: bool = False
) -> List[float]:
    """Compute rewards as return of objective function with given parameters.

    Args:
        params: Parameters as input for objective function.
        objective: Objective function.
        verbose: If True an output is logged.

    Returns:
        Rewards for parameters.
    """

    rewards = []
    for i, param in enumerate(params):
        reward = objective(param, observation_noise=False).item()
        rewards.append(reward)
        if verbose:
            print(f"Iteration {i+1}, reward {reward :.2f}.")
    return rewards


def get_maxima_objectives(
    lengthscales: Dict[int, Tensor],
    noise_variance: float,
    train_x: Dict[int, Tensor],
    train_y: Dict[int, Tensor],
    n_max: Optional[int],
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """Compute maxima of synthetic objective functions.

    Args:
        lengthscales: Hyperparameter for GP model.
        noise_variance: Observation noise.
        train_x: The training features X.
        train_y: The training targets y.
        n_max: Number of train_x samples for optimization's starting points.

    Returns:
        Maxima values (max) and positions (argmax).
    """
    f_max_dict = {}
    argmax_dict = {}
    dimensions = list(lengthscales.keys())
    number_objectives = len(train_y[dimensions[0]])
    for dim in dimensions:
        f_max_dim = []
        argmax_dim = []
        for index_objective in range(number_objectives):
            objective = generate_objective_from_gp_post(
                train_x[dim][index_objective],
                train_y[dim][index_objective],
                noise_variance=noise_variance,
                gp_hypers={
                    "covar_module.base_kernel.lengthscale": lengthscales[dim],
                    "covar_module.outputscale": torch.tensor(1.0),
                },
            )
            f = lambda x: objective(x, observation_noise=False, requires_grad=True)
            if n_max:
                _, indices_sort = torch.sort(train_y[dim][index_objective])
                init_cond = train_x[dim][index_objective][indices_sort[:n_max]]
            else:
                init_cond = train_x[dim][index_objective]
            clamped_candidates, batch_acquisition = botorch.gen_candidates_scipy(
                initial_conditions=init_cond,
                acquisition_function=f,
                lower_bounds=torch.tensor([[0.0] * dim]),
                upper_bounds=torch.tensor([[1.0] * dim]),
            )
            max_optimizer, index_optimizer = torch.max(batch_acquisition, dim=0)
            max_train_samples, index_max_train_samples = torch.max(
                train_y[dim][index_objective], dim=0
            )
            if max_optimizer > max_train_samples:
                f_max_dim.append(max_optimizer.clone().item())
                argmax_dim.append(clamped_candidates[index_optimizer])
            else:
                f_max_dim.append(max_train_samples.clone().item())
                argmax_dim.append(
                    train_x[dim][index_objective][index_max_train_samples]
                )
        f_max_dict[dim] = f_max_dim
        argmax_dict[dim] = argmax_dim
    return f_max_dict, argmax_dict


def get_lengthscale_hyperprior(dim: int, factor_lengthscale: int, gamma: float):
    """Compute hyperprior for lengthscale.

    Args:
        dim: Dimension of search space.
        factor_lengthscale: Scale for upper bound of lengthscales' sample
            distribution.
        gamma: Noise parameter for uniform sample distribution for lengthscales.

    Returns:
        Gpytorch hyperprior for lengthscales.
    """
    l = get_lengthscales(dim, factor_hennig)
    a = factor_lengthscale * l * (1 - gamma)
    b = factor_lengthscale * l * (1 + gamma)
    return gpytorch.priors.UniformPrior(a, b)
