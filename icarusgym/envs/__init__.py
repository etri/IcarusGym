# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Package that includes the gym-type environments provided by IcarusGym.
"""

import gym

from .decision_array_cache import DecisionArrayCache
from .gym_env_base import GymEnvBase
from .passive_agent_cache import PassiveAgentCache
from .ttl_cache import TtlCache

gym.register(id='DecisionArrayCache-v0', entry_point='icarusgym.envs:DecisionArrayCache')
gym.register(id='PassiveAgentCache-v0', entry_point='icarusgym.envs:PassiveAgentCache')
gym.register(id='TtlCache-v0', entry_point='icarusgym.envs:TtlCache')
