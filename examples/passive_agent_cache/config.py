# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Configuration for passive agent cache environment example. The 'passive agent' means that the agent does not
actually control the network caching. Instead, the environment controls the network caching by using the legacy caching
algorithms provided by original Icarus. In other words, actions from the passive agent do not have influence on the
network caching. This environment is useful when we need to compare the performances of reinforcement learning-based
control and legacy control in same scenario.
"""

import numpy as np

from collections import deque
from icarus.util import Tree


# ============================== CONSTANTS FOR MAIN ROUTINE ==============================
NUM_EPISODES = 1    # Number of episodes to evaluate.
# ========================================================================================

# ============================== CONSTANTS FOR ICARUSGYM ACTUAL ENVIRONMENT ==============================
N_CONTENTS = 100                # Number of contents.
WORKLOAD_N_WARM_UP = 0          # Number of sessions during warm-up period of simulation.
WORKLOAD_N_MEASURED = 3000      # Number of sessions after warm-up period of simulation.
WORKLOAD_NAME = 'STATIONARY'    # Workload name.
ALPHA = 1.                      # Alpha parameter.
BETA = 0.                       # Beta parameter.
RATE = 1.                       # Rate parameter.
CACHE_RATIO = 0.1               # Ratio of cache per number of contents.
# ========================================================================================================

# ============================== CONSTANTS FOR ICARUSGYM ENVIRONMENT ==============================
CONFIG_PATH = 'config.py'                   # Config file name for icarus-sim.
OUTPUT_PATH = 'result.pickle'               # Output file name for icarus-sim.
TTL_MAX = np.inf                            # Maximum TTL values.
CONTENT_MAX = N_CONTENTS                    # Maximum content ID.
CACHE_SIZE_MAX = CONTENT_MAX * CACHE_RATIO  # Maximum cache size.
# =================================================================================================

# ============================== GENERAL SETTINGS FOR ICARUSGYM ACTUAL ENVIRONMENT ==============================
LOG_LEVEL = 'INFO'  # Output level of logging package.

# Indicates whether simulation is run in parallel, or not (It should be 'False' for IcarusGym).
PARALLEL_EXECUTION = False

RESULTS_FORMAT = 'PICKLE'               # Format of output file.
N_REPLICATIONS = 1                      # Number of replicated run of each simulation scenario.
CACHING_GRANULARITY = 'OBJECT'
DATA_COLLECTORS = ['CACHE_HIT_RATIO']   # Specifies data collectors active in simulation.
# ===============================================================================================================

# ============================== EXPERIMENT SETTINGS FOR ICARUSGYM ACTUAL ENVIRONMENT ==============================
# Create experiment queue object (double ended queue).
EXPERIMENT_QUEUE = deque()

# Build experiment objects with respect to varying options.
experiment = Tree()

experiment['topology']['name'] = 'PATH'             # Specifies network topology.
experiment['topology']['n'] = 3                     # Number of nodes.
experiment['topology']['delay'] = 10.               # Transmission delay of every link in topology.
experiment['workload']['name'] = WORKLOAD_NAME      # Workload name.
experiment['workload']['n_contents'] = N_CONTENTS   # Number of contents.

# Number of sessions during warm-up period of simulation. '0' means there is no warm-up period.
experiment['workload']['n_warmup'] = WORKLOAD_N_WARM_UP

# Number of sessions after warm-up period of simulation.
experiment['workload']['n_measured'] = WORKLOAD_N_MEASURED

experiment['workload']['rate'] = RATE                   # Workload generation rate.
experiment['workload']['alpha'] = ALPHA                 # Alpha parameter of workload generator.
experiment['content_placement']['name'] = 'UNIFORM'     # Specifies content source deployment.
experiment['cache_placement']['name'] = 'UNIFORM'       # Specifies cache placement.

# Size of cache in each node. If experiment['workload']['n_contents'] == 100 and
# experiment['cache_placement']['network_cache'] == 0.05, the cache size of each node becomes 5.
experiment['cache_placement']['network_cache'] = CACHE_RATIO

experiment['strategy']['name'] = 'PASSIVE_ICARUSGYM_LCE'    # Specifies cache strategy.

# Maximum content ID parameter of passive IcarusGym LCE strategy.
experiment['strategy']['content_max'] = CONTENT_MAX

# Interval of collecting statistics of Passive IcarusGym LCE strategy.
experiment['strategy']['stats_interval'] = 60.

experiment['cache_policy']['name'] = 'PERFECT_LFU'          # Specifies cache policy.
experiment['desc'] = ('Line topology with a single router - ' + 'strategy: %s / cache policy: %s' %
                      (str(experiment['strategy']['name']), str(experiment['cache_policy']['name'])))
EXPERIMENT_QUEUE.append(experiment)
# ==================================================================================================================
