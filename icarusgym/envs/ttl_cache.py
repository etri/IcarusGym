# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating time-to-live (TTL) cache. The TTL cache environment is implemented based on the
following reference:
M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27, no. 3,
pp. 1013-1027, May 2019 (Earlier version of the paper is presented in IEEE INFOCOM 2016).
"""

import numpy as np

from icarusgym.envs.gym_env_base import GymEnvBase
from gym.spaces import Discrete, Box, Tuple


class TtlCache(GymEnvBase):
    """Class that defines the observation and action spaces of gym-type TtlCache environment.
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
        ttl_max = config['ttl_max']

        # An observation is a tuple that consists of four values: env_time, content, remaining_ttl, and hit. 'env_time'
        # is the current time of caching simulation. 'content' is the ID of requested content. 'remaining_ttl' is the
        # remaining time until when the requested is removed. 'hit' becomes 1 when the requested content is hit in the
        # cache, 0 for the case of cache miss.
        return Tuple((Box(low=0., high=np.inf, shape=(1,), dtype=np.float),
                      Box(low=0, high=content_max, shape=(1,), dtype=np.uint32),
                      Box(low=0., high=ttl_max, shape=(1,), dtype=np.float),
                      Discrete(2)))

    @staticmethod
    def build_action_space(**kwargs):
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs['config']
        ttl_max = config['ttl_max']
        cache_size_max = config['cache_size_max']

        # An action is a tuple that consists of two values: ttl and cache_size. 'ttl' is the time interval for which
        # the requested content is stored in the cache. 'cache_size' is the size of cache. When the cache size becomes
        # lower than the number of cached contents at the previous time-step, the cache evicts the contents with the
        # most lowest remaining TTLs.
        return Tuple((Box(low=0, high=ttl_max, shape=(1,), dtype=np.float),
                      Box(low=0, high=cache_size_max, shape=(1,), dtype=np.uint32)))
