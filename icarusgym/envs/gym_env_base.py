# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that defines the base class of gym-type environments provided by IcarusGym.
"""

from abc import *
from gymproxy import BaseEnv
from icarusgym.actual_env import IcarusActualEnv


class GymEnvBase(BaseEnv, metaclass=ABCMeta):
    """Base class of gym-type environment classes provided by IcarusGym.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        BaseEnv.actual_env_class = IcarusActualEnv
        super().__init__(**kwargs)
