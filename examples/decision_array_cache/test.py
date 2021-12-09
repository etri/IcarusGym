# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Example for decision array cache environment. The decision array cache environment is implemented based on the
following reference:
A. Sadeghi et al., "Deep reinforcement learning for adaptive caching in hierarchical content delivery networks," IEEE
Trans. Cogn. Commun. Netw., vol. 5, no. 4, pp. 1024-1033, Dec. 2019.
The decision array cache environment does not provide 2-timescale formulation in the reference. If you want to use
the 2-timescale concepts, we recommend you to implement the concepts in agent scope, namely, by pre-processing
observation and information. It may be not difficult.
"""

import examples.decision_array_cache.config as conf
import fnss
import networkx as nx
import numpy as np

from icarus.registry import register_topology_factory
from icarus.scenarios.topology import *
from icarusgym import *

# Setting the logger.
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Parameters for preventing from logging explosive information.
CONTENT_THR = 10
NODE_THR = 10

# Decision array for a fixed content deployment. We assume that tier-1 nodes store the contents with even numbered
# IDs, and tier-2 node stores the contents with odd numbered IDs.
FIXED_ACTION = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def main():
    """Main routine for testing decision array cache environment.
    """
    config = {'config_path': conf.CONFIG_PATH,
              'output_path': conf.OUTPUT_PATH,
              'content_max': conf.CONTENT_MAX,
              'node_max': conf.NODE_MAX,
              'ttl_max': conf.TTL_MAX}
    env = gym.make(id='DecisionArrayCache-v0', config=config)
    for i in range(0, conf.NUM_EPISODES):
        j = 0
        _ = env.reset()
        while True:
            env.render()

            # For more details of obs and action, refer icarusgym.envs.decision_array_cache module.
            action = FIXED_ACTION
            obs, reward, done, info = env.step(action)

            # Defensive codes for preventing from logging explosive information when many contents are considered.
            if conf.CONTENT_MAX <= CONTENT_THR and conf.NODE_MAX <= NODE_THR:
                log_step(i, j, obs, reward, done, info, action)

            j = j + 1
            if done:
                break
    env.close()


@register_topology_factory('CUSTOM')
def topology_custom(delay: float = 1., **kwargs) -> IcnTopology:
    """Custom topology generation routine.

    :param delay: Delay of each link in millisecond.
    :param kwargs: Dictionary of keyword arguments.
    :return: IcnTopology object. For more details about the IcnTopology class, refer icarus.scenarios.topology module.
    """

    # Defines a topology that includes 1 content server, 3 cache routers, and 2 client nodes. Node 0 is content server.
    # Node 1 is tier-2 (parent) cache router. Node 2 and 3 are tier-1 (child) cache routers. Node 4 and 5 is connected
    # to node 2 and 3, respectively.
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)])
    receivers = [4, 5]
    routers = [1, 2, 3]
    sources = [0]

    # Sets a number of property of the topology.
    topology = fnss.Topology(g)
    topology.name = "custom_topology"
    topology.graph['type'] = 'custom'
    topology.graph['icr_candidates'] = set(routers)

    # Deploys protocol stacks to the nodes according to their roles.
    for v in sources:
        fnss.add_stack(topology, v, 'source')
    for v in receivers:
        fnss.add_stack(topology, v, 'receiver')
    for v in routers:
        fnss.add_stack(topology, v, 'router')

    # Sets weights and delays on all links.
    fnss.set_weights_constant(topology, 1.0)
    fnss.set_delays_constant(topology, delay, 'ms')

    # Labels links as internal or external.
    for u, v in topology.edges():
        topology.adj[u][v]['type'] = 'internal'
    return IcnTopology(topology)


def log_step(episode: int, step: int, obs: tuple, reward: float, done: bool, info: dict, action: np.ndarray):
    """Utility function for printing logs.

    :param episode: Current episode number.
    :param step: Current time-step.
    :param obs: Observation from the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the current episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: Action to be given to the current environment.
    """
    step_str = '{}-th step in {}-th episode: \n'.format(step, episode)
    obs_str = 'obs: \n{}\n'.format(obs)
    reward_str = 'reward: {} \n'.format(reward)
    done_str = 'done: {} \n'.format(done)
    info_hits_str = 'info: hits: \n{}\n'.format(info['hits'])
    info_misses_str = 'info: misses: \n{}\n'.format(info['misses'])
    info_fetch_req_str = 'info: fetch_req_overheads: {}\n'.format(info['fetch_req_overheads'])
    info_fetch_res_str = 'info: fetch_res_overheads: {}\n'.format(info['fetch_res_overheads'])
    action_str = 'action: \n{}\n'.format(action)
    result_str = (step_str + obs_str + reward_str + done_str + info_hits_str + info_misses_str + info_fetch_req_str +
                  info_fetch_res_str + action_str)
    logger.info(result_str)


if __name__ == "__main__":
    main()
