# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating passive agent cache. The 'passive agent' means that the agent does not actually
control the network caching. Instead, the environment controls the network caching by using the legacy caching
algorithms provided by original Icarus. In other words, actions from the passive agent do not have influence on the
network caching. This environment is useful when we need to compare the performances of reinforcement learning-based
control and legacy control in same scenario.
"""

import numpy as np

from icarusgym.envs.gym_env_base import GymEnvBase
from gym.spaces import Discrete, Box, Tuple


class PassiveAgentCache(GymEnvBase):
    """Class that defines the observation and action spaces of gym-type PassiveAgentCache environment.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        super().__init__(**kwargs)

    @staticmethod
    def build_obs_space(**kwargs):
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs['config']
        content_max = config['content_max']

        # An observation is a tuple that consists of three values: env_time, content, hit. 'env_time' is the current
        # time of caching simulation. 'content' is the ID of requested content. 'hit' becomes 1 when the requested
        # content is hit in the cache, 0 for the case of cache miss.
        return Tuple((Box(low=0., high=np.inf, shape=(1,), dtype=np.float),
                      Box(low=0, high=content_max, shape=(1,), dtype=np.uint32),
                      Discrete(2)))

    @staticmethod
    def build_action_space(**kwargs):
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        # An action is a meaningless 0/1 variable that does not have any influence on the actual environment.
        return Discrete(2)
