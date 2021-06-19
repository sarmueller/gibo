from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

from abc import ABC, abstractmethod

from time import time
import glfw

import torch
import gym
from gym import wrappers


class EnvironmentObjective:
    """API for translating an OpenAI gym environment into a black-box objective
    function for which a parameterized policy should be optimized.

    Attributes:
        env: OpenAI Gym environment.
        policy: Parameterized policy that should by optimized for env.
        manipulate_state: Function that manipluates the states of the
            environment.
        manipulate_reward: Function that manipluates the reward of the
            environment.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: Callable,
        manipulate_state: Optional[Callable] = None,
        manipulate_reward: Optional[Callable] = None,
    ):
        """Inits the translation environment to objective."""
        self.env = env
        self.policy = policy
        self.max_steps = env._max_episode_steps
        self.timesteps = 0
        self.timesteps_to_reward = {}
        shape_states = env.observation_space.shape
        dtype_states = torch.float32

        shape_actions = env.action_space.shape
        dtype_actions = torch.tensor(env.action_space.sample()).dtype

        self._last_episode_length = 0
        self._last_episode = {
            "states": torch.empty(
                (self.max_steps + 1,) + shape_states, dtype=dtype_states
            ),
            "actions": torch.empty(
                (self.max_steps,) + shape_actions, dtype=dtype_actions
            ),
            "rewards": torch.empty(self.max_steps, dtype=torch.float32),
        }

        if manipulate_reward is None:
            manipulate_reward = lambda reward, action, state, done: reward
        self.manipulate_reward = manipulate_reward

        self._manipulate_state = manipulate_state
        if manipulate_state is None:
            manipulate_state = lambda state: state

        self.manipulate_state = lambda state: manipulate_state(
            torch.tensor(state, dtype=dtype_states)
        )

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        return self.run(params)

    def _unpack_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function for get_last_episode.

        Get states, actions and rewards of last episode.

        Returns:
            Tuple of states, actions and rewards.
        """
        states = self._last_episode["states"]
        actions = self._last_episode["actions"]
        rewards = self._last_episode["rewards"]
        return states, actions, rewards

    def get_last_episode(self) -> Dict[int, torch.Tensor]:
        """Return states, actions and rewards of last episode.

        Implemented for the implementation of policy gradient methods.

        Returns:
            Dictionary of states, actions and rewards.
        """
        states, actions, rewards = self._unpack_episode()
        return {
            "states": states[: self._last_episode_length + 1].clone(),
            "actions": actions[: self._last_episode_length].clone(),
            "rewards": rewards[: self._last_episode_length].clone(),
        }

    def run(
        self, params: torch.Tensor, render: bool = False, test: bool = False
    ) -> torch.Tensor:
        """One rollout of an episodic environment with finite horizon.

        Evaluate value of current parameter constellation with sum of collected
        rewards over one rollout.

        Args:
            params: Current parameter constellation.
            render: If True render environment.
            test: If True renderer is not closed after one run.

        Returns:
            Cumulated reward.
        """
        states, actions, rewards = self._unpack_episode()
        r = 0
        states[0] = self.manipulate_state(self.env.reset())
        for t in range(self.max_steps):  # rollout
            actions[t] = self.policy(states[t], params)
            state, rewards[t], done, _ = self.env.step(actions[t].numpy())
            states[t + 1] = self.manipulate_state(state)
            r += self.manipulate_reward(
                rewards[t], actions[t], states[t + 1], done
            )  # Define as stochastic gradient ascent.
            if render:
                self.env.render()
            if done:
                break
        if not test:
            self.timesteps += t
            self.timesteps_to_reward.update({self.timesteps: rewards[:t].sum()})
        self._last_episode_length = t
        if render and not test:
            self.env.close()

        return torch.tensor([r], dtype=torch.float32)

    def test_params(
        self,
        params: torch.Tensor,
        episodes: int,
        render: bool = True,
        path_to_video: Optional[str] = None,
        verbose: bool = True,
    ):
        """Test case for quantitative evaluation of parameter perfomance on
        environment.

        Args:
            params: Current parameter constellation.
            episodes: Number of episodes.
            render: If True render environment.
            path_to_video: Path to directory if a video wants to be saved.
            verbose: If True an output is logged.

        Returns:
            Cumulated reward.
        """
        if path_to_video is not None:
            self.env = wrappers.Monitor(self.env, path_to_video, force=True)
        num_digits = len(str(episodes))
        for episode in range(episodes):
            reward = self.run(params, render=render, test=True)
            if verbose:
                print(f"episode: {episode+1:{num_digits}}, reward: {reward}")
        if render:
            try:
                glfw.destroy_window(self.env.viewer.window)
                self.env.viewer = None
            except:
                self.env.close()
        return reward


class StateManipulator(ABC):
    """Abstract class for state manipulation."""

    def __init__(self):
        pass

    def __call__(self, state):
        return self.manipulate(state)

    @abstractmethod
    def manipulate(self, state):
        pass


class StateNormalizer(StateManipulator):
    """Class for state normalization.

    Implementation of Welfords online algorithm. For further information see
        thesis appendix A.3.

    Attributes:
        eps: Small value to prevent division by zero
        normalize_params: Normalization function for policy parameters.
        unnormalize_params: Unnormalization function for policy parameters.
    """

    def __init__(
        self, eps: float = 1e-8, normalize_params=None, unnormalize_params=None
    ):
        # Init super.
        self.eps = eps
        self.steps = 0
        self._mean_of_states = 0.0
        self._sum_of_squared_errors = 0.0
        self.mean = 0.0
        self.std = 1.0
        if normalize_params is None:
            normalize_params = lambda params, mean, std: params
        self.__normalize_params = normalize_params

        if unnormalize_params is None:
            unnormalize_params = lambda params, mean, std: params
        self.__unnormalize_params = unnormalize_params

    def _get_mean_var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function for apply_update.

        Returns:
            Means and variances of states.
        """
        if self.steps <= 1:
            var = torch.ones_like(self._mean_of_states)
        else:
            # Sample variance.
            var = self._sum_of_squared_errors / (self.steps - 1)
        return self._mean_of_states, var

    def _welford_update(self, state: torch.Tensor):
        """Helper function for manipulate.

        Internally trackes mean and std according to the seen states.

        Args:
            state: New state.
        """
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.steps += 1
        delta = state - self._mean_of_states
        self._mean_of_states += delta / self.steps
        self._sum_of_squared_errors += delta * (state - self._mean_of_states)

    def manipulate(self, state: torch.Tensor) -> torch.Tensor:
        """Actually manipulate a state with the tracked mean and standard
        deviation.

        Args:
            state: State to normalize.

        Returns:
            Normalized state.
        """
        self._welford_update(state)
        normalized_state = (state - self.mean) / self.std
        return normalized_state

    def apply_update(self):
        """Updates mean and std according to the states internally tracked using
        _welford_update."""
        self.mean, var = self._get_mean_var()
        self.std = torch.sqrt(var) + self.eps

    def normalize_params(self, params: torch.Tensor):
        return self.__normalize_params(params, self.mean, self.std)

    def unnormalize_params(self, params: torch.Tensor):
        return self.__unnormalize_params(params, self.mean, self.std)


def manipulate_reward(shift: Union[int, float], scale: Union[int, float]):
    """Manipulate reward in every step with shift and scale.

    Args:
        shift: Reward shift.
        scale: Reward scale.

    Return:
        Manipulated reward.
    """
    if shift is None:
        shift = 0
    if scale is None:
        scale = 1
    return lambda reward, action, state, done: (reward - shift) / scale
