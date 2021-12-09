# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Example script for passive agent caching. The 'passive agent' means that the agent does not actually control the
network caching. Instead, the environment controls the network caching by using the legacy caching algorithms provided
by original Icarus. In other words, actions from the passive agent do not have influence on the network caching. This
environment is useful when we need to compare the performances of reinforcement learning-based control and legacy
control in same scenario.
"""

import examples.passive_agent_cache.config as conf

from icarusgym import *

# Setting the logger.
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')


def main():
    """Main routine for testing passive agent cache environment.
    """
    config = {'config_path': conf.CONFIG_PATH,
              'output_path': conf.OUTPUT_PATH,
              'content_max': conf.CONTENT_MAX,
              'ttl_max': conf.TTL_MAX}
    env = gym.make(id='PassiveAgentCache-v0', config=config)
    for i in range(0, conf.NUM_EPISODES):
        j = 0
        _ = env.reset()
        while True:
            env.render()

            # For more details of obs and action, refer icarusgym.envs.passive_agent_cache module.
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            log_step(i, j, obs, reward, done, info, action)
            j = j + 1
            if done:
                break
    env.close()


def log_step(episode: int, step: int, obs: tuple, reward: float, done: bool, info: dict, action: int):
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
    hit = obs[2]
    step_str = '{}-th step in {}-th episode \n'.format(step, episode)
    obs_str = 'obs: {} \n'.format((env_time, cid, hit))
    reward_str = 'reward: {} \n'.format(reward)
    done_str = 'done: {} \n'.format(done)
    info_str = 'info: {} \n'.format(info)
    action_str = 'action: {}\n'.format(action)
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
