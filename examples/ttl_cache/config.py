# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Configuration for time-to-live (TTL) cache environment example. The TTL cache environment is implemented based on
the following reference:
M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27, no. 3,
pp. 1013-1027, May 2019 (Earlier version of the paper is presented in IEEE INFOCOM 2016).
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
IS_RESET = False                # Whether cache policy is reset or non-reset.
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

experiment['strategy']['name'] = 'ICARUSGYM_LCE'        # Specifies cache strategy.

# Maximum content ID parameter of IcarusGym LCE strategy.
experiment['strategy']['content_max'] = CONTENT_MAX

experiment['strategy']['stats_interval'] = 60.          # Interval of collecting statistics of IcarusGym LCE strategy.
experiment['cache_policy']['name'] = 'ICARUSGYM_TTL'    # Specifies cache policy.

# Indicates TTL is reset when a content is hit. If it is true, the expiration time is extended for each cache hit.
experiment['cache_policy']['is_reset'] = IS_RESET

# Specifies extra cache size ratio. or example, suppose that original cache size is 10. If the number of content is 10,
# and extra_cache_size_ratio is set to 0.2, the extra cache size becomes 2. Therefore, the practical cache size becomes
# 12. For more details of the theoretic aspects of extra cache, please refer Section IV-C in the reference mentioned at
# the beginning of this class.
experiment['cache_policy']['extra_cache_size_ratio'] = 0.2

experiment['desc'] = ('Line topology with a single router - ' + 'strategy: %s / cache policy: %s' %
                      (str(experiment['strategy']['name']), str(experiment['cache_policy']['name'])))
EXPERIMENT_QUEUE.append(experiment)
# ==================================================================================================================
