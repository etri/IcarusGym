# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Example script for time-to-live (TTL) caching. The TTL cache environment is implemented based on the following
reference:
M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27, no. 3,
pp. 1013-1027, May 2019 (Earlier version of the paper is presented in IEEE INFOCOM 2016).
"""

import examples.ttl_cache.config as conf
import numpy as np

from icarusgym import *

# Setting the logger.
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')


def main():
    """Main routine for testing the TTL cache environment.
    """
    config = {'config_path': conf.CONFIG_PATH,
              'output_path': conf.OUTPUT_PATH,
              'content_max': conf.CONTENT_MAX,
              'ttl_max': conf.TTL_MAX,
              'cache_size_max': int(conf.N_CONTENTS * conf.CACHE_RATIO)}
    env = gym.make(id='TtlCache-v0', config=config)
    for i in range(0, conf.NUM_EPISODES):
        j = 0
        _ = env.reset()
        while True:
            env.render()

            # FIFO cache with the TTL value of 16.0s for every content.
            action = (np.array([16.], dtype=np.float), np.array([conf.N_CONTENTS * conf.CACHE_RATIO], dtype=np.uint32))

            # For more details of obs and action, refer icarusgym.envs.ttl_cache module.
            obs, reward, done, info = env.step(action)

            log_step(i, j, obs, reward, done, info, action)
            j = j + 1
            if done:
                break
    env.close()


def log_step(episode: int, step: int, obs: tuple, reward: float, done: bool, info: dict, action: tuple):
    """Utility function for printing logs.

    :param episode: Current episode number.
    :param step: Current time-step.
    :param obs: Observation from the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the current episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: Action to be given to the current environment.
    """
    env_time = obs[0].item()
    cid = obs[1].item()
    remaining_time = obs[2].item()
    hit = obs[3]
    step_str = '{}-th step in {}-th episode \n'.format(step, episode)
    obs_str = 'obs: {} \n'.format((env_time, cid, remaining_time, hit))
    reward_str = 'reward: {} \n'.format(reward)
    done_str = 'done: {} \n'.format(done)
    info_str = 'info: {} \n'.format(info)
    action_ttl_str = 'action: TTL: {} / '.format(action[0][0])
    action_cache_size_str = 'cache size: {} \n'.format(action[1][0])
    action_str = action_ttl_str + action_cache_size_str
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
