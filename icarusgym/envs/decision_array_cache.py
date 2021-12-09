# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating decision array cache. The decision array cache is implemented based on the
following reference:
A. Sadeghi et al., "Deep reinforcement learning for adaptive caching in hierarchical content delivery networks," IEEE
Trans. Cogn. Commun. Netw., vol. 5, no. 4, pp. 1024-1033, Dec. 2019.
The decision array cache environment does not provide 2-timescale formulation in the reference. If you want to use
the 2-timescale concepts, we recommend you to implement the concepts in agent scope, namely, by pre-processing
observation and information. It may be not difficult.
"""

import numpy as np

from icarusgym.envs.gym_env_base import GymEnvBase
from gym.spaces import Box


class DecisionArrayCache(GymEnvBase):
    """Class that defines the observation and action spaces of gym-type DecisionArrayCache environment.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        super().__init__(**kwargs)

    @staticmethod
    def build_obs_space(**kwargs) -> Box:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs['config']
        node_max = config['node_max']
        content_max = config['content_max']

        # The observation is an array that represents the numbers of content requests coming to each cache in the
        # network for single time-step. Namely, 'requests[n, c] = i' means that the number of requests for content c is
        # i at the cache of node n. If node n is not cache router, requests[n, c] always becomes 0 because node n does
        # not have cache. We use this somewhat wasteful definition due to implementation convenience.
        return Box(low=0, high=np.inf, shape=(node_max + 1, content_max + 1), dtype=np.uint32)

    @staticmethod
    def build_action_space(**kwargs) -> Box:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs['config']
        node_max = config['node_max']
        content_max = config['content_max']

        # The action is a 0/1-value array that indicates, for each content for each node, whether a content is cached
        # or not at a node. Namely, 'decision[n, c] = 1' means that content c is stored at the cache of node n.
        # 'decision[n, i] = 0' means otherwise. If node n is not cache router, 'requests[n, c] = 1' does not effective
        # since node n does not have cache. We use this somewhat wasteful definition due to implementation convenience.
        return Box(low=0, high=1, shape=(node_max + 1, content_max + 1), dtype=np.uint32)
