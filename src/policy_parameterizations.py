from typing import Tuple, List, Callable, Union, Optional

import torch


class MLP:
    """Multilayer perceptrone.

    Consists of at least two layers of nodes: an input layer and an output
    layer. Optionally one can extend it with arbitrary many hidden layers.
    Except for the input nodes, each node is a neuron that can optionally use a
    nonlinear activation function.

    Attributes:
        L0: Number of input nodes. For a gym environment objective this
            corresponds to the states.
        Ls: List of numbers for nodes of optional hidden layers and the output
            layer. For a gym environment objective the last number of the list
            has to correspond to the actions.
        add_bias: If True every layer has one bias vector of the same dimension
            as the output dimension of the layer.
        nonlinearity: Opportunity to hand over a nonlinearity function.
    """

    def __init__(
        self,
        L0: int,
        *Ls: List[int],
        add_bias: bool = False,
        nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Inits MLP."""
        self.L0 = L0
        self.Ls = Ls
        self.add_bias = add_bias
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip((L0,) + Ls[:-1], Ls)
            ]
        )

        if nonlinearity is None:
            nonlinearity = lambda x: x
        self.nonlinearity = nonlinearity

    def __call__(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Maps states and parameters of MLP to its actions.

        Args:
            state: The state tensor.
            params: Parameters of the MLP.

        Returns:
            Output of the MLP/actions.
        """
        with torch.no_grad():
            params = params.view(self.len_params)
            out = state
            start, end = (0, 0)
            in_size = self.L0
            for out_size in self.Ls:
                # Linear mapping.
                start, end = end, end + in_size * out_size
                out = out @ params[start:end].view(in_size, out_size)
                # Add bias.
                if self.add_bias:
                    start, end = end, end + out_size
                    out = out + params[start:end]
                # Apply nonlinearity.
                out = self.nonlinearity(out)
                in_size = out_size
        return out

    def _select_first_layer(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function for state normalization.

        Normalization is only applied for the first layer out = A @ in + b.

        Args:
            params: Parameters of MLP.

        Returns:
            A tuple of the layer nodes A and its biases b.

        Raises:
            ValueError: If the mlp has no bias vectors in every layer.
        """
        if self.add_bias is False:
            raise ValueError(
                f"For state normalization the MLP should have biases, but add_bias is {self.add_bias}."
            )
        in_size = self.L0
        out_size = self.Ls[0]
        A = params[:, : in_size * out_size].view(-1, in_size, out_size)
        b = params[:, in_size * out_size : (in_size + 1) * out_size]
        return A, b

    def normalize_params(
        self,
        params: torch.Tensor,
        mean: Union[float, torch.Tensor],
        std: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """State normalization for a MLP.

        Only the first layer is transformed affine linear.
        For further information see thesis 4.2 Extensions.

        Args:
            params: Parameters of MLP.
            mean: Mean of states.
            std: Standard deviation of states.

        Returns:
            Normalized parameters of MLP.
        """
        params = params.clone()
        A, b = self._select_first_layer(params)
        if type(mean) is torch.Tensor:
            b += mean @ A
        else:
            b += (mean * torch.ones(A.shape[1])) @ A

        if type(std) is torch.Tensor:
            A *= std.view(-1, 1)
        else:
            A *= std

        return params

    def unnormalize_params(
        self,
        params: torch.Tensor,
        mean: Union[float, torch.Tensor],
        std: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """State unnormalization for a MLP.

        Only the first layer is transformed affine linear.
        For further information see thesis 4.2 Extensions.

        Args:
            params: Parameters of MLP.
            mean: Mean of states.
            std: Standard deviation of states.

        Returns:
            Unnormalized parameters of MLP.
        """
        params = params.clone()
        A, b = self._select_first_layer(params)
        if type(std) is torch.Tensor:
            A /= std.view(-1, 1)
        else:
            A /= std

        if type(mean) is torch.Tensor:
            b -= mean @ A
        else:
            b -= (mean * torch.ones(A.shape[1])) @ A

        return params


def discretize(function: Callable, num_actions: int):
    """Discretize output/actions of MLP.

    For instance necessary for the CartPole environment.

    Args:
        function: Mapping states with parameters to actions, e.g. MLP.
        num_actions: Number of function outputs.

    Returns:
        Function with num_actions discrete outputs.
    """

    def discrete_policy_2(state, params):
        return (function(state, params) > 0.0) * 1

    def discrete_policy_n(state, params):
        return torch.argmax(function(state, params))

    if num_actions == 2:
        discrete_policy = discrete_policy_2
    elif num_actions > 2:
        discrete_policy = discrete_policy_n
    else:
        raise (f"Argument num_actions is {num_actions} but has to be greater than 1.")
    return discrete_policy
