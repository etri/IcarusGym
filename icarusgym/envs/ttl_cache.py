# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating time-to-live (TTL) cache. The TTL cache environment is implemented based on the
following reference:
M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27, no. 3,
pp. 1013-1027, May 2019 (Earlier version of the paper is presented in IEEE INFOCOM 2016).
"""

from typing import Optional
import numpy as np

import gymnasium
from icarusgym.envs.gym_env_base import GymEnvBase
from gymnasium.spaces import Discrete, Box, Tuple


class TtlCache(GymEnvBase):
    """Class that defines the observation and action spaces of gym-type TtlCache environment.
    """
    def __init__(self, config: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        print("###config",config)
        # Provide default config if None is passed (e.g., during RLLib environment checks)
        if config is None:
            config = {
                'content_max': 100,
                'ttl_max': float('inf'),  # Use infinity instead of 100.0
                'cache_size_max': 30.0,  # Use actual experiment value
                'config_path': '/dev/null',  # Dummy path
                'output_path': '/dev/null'   # Dummy path
            }
        super().__init__(config)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> Tuple:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        if config is None:
            # Default values when no config is provided (e.g., during RLLib environment checks)
            content_max = 100
            ttl_max = float('inf')  # Use infinity instead of 100.0
        else:
            content_max = config['content_max']
            ttl_max = config['ttl_max']

        # An observation is a flattened numpy array that consists of six values: env_time, content_id, weight, size, remaining_ttl, and hit. 
        # The GymProxy automatically flattens tuple observations to numpy arrays, so we define the space as a single Box.
        # 'env_time' is the current time of caching simulation. 'content_id' is the ID of requested content. 
        # 'weight' is the weight/importance of the content. 'size' is the size of the content.
        # 'remaining_ttl' is the remaining time until when the requested is removed. 'hit' becomes 1 when the requested 
        # content is hit in the cache, 0 for the case of cache miss.
        return Box(low=0., high=np.inf, shape=(6,), dtype=np.float64)

    @staticmethod
    def build_action_space(kwargs: Optional[dict] = None) -> Tuple:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        if config is None:
            # Default values when no config is provided (e.g., during RLLib environment checks)
            ttl_max = float('inf')  # Use infinity instead of 100.0
            cache_size_max = 30.0  # Use actual experiment value
        else:
            ttl_max = config['ttl_max']
            cache_size_max = config['cache_size_max']

        # An action is a tuple that consists of two values: ttl and cache_size. 'ttl' is the time interval for which
        # the requested content is stored in the cache. 'cache_size' is the size of cache. When the cache size becomes
        # lower than the number of cached contents at the previous time-step, the cache evicts the contents with the
        # most lowest remaining TTLs.
        return Tuple((Box(low=0, high=ttl_max, shape=(1,), dtype=np.float64),
                      Box(low=0, high=cache_size_max, shape=(1,), dtype=np.int_)))
