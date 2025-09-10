# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that defines the base class of gym-type environments provided by IcarusGym.
"""

from abc import *
from gymproxy import GymEnv #originall BaseEnv
from icarusgym.actual_env import IcarusActualEnv
from typing import TypeVar, Optional, Any, SupportsFloat
import numpy as np

class GymEnvBase(GymEnv, metaclass=ABCMeta):
    """Base class of gym-type environment classes provided by IcarusGym.
    """
    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        print("$$$kwargs",kwargs)

        GymEnv.actual_env_class = IcarusActualEnv
        super().__init__(kwargs)
        #self.icarus_env = IcarusActualEnv(kwargs)
