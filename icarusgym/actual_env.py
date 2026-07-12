# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that ports the main control loop of Icarus to Gym through GymProxy.
"""

from typing import Optional
import collections
import copy
import functools
import logging
import os
import signal
import sys
import time
import traceback

from gymproxy import ActualEnv
from icarus.execution import NetworkModel, NetworkView, NetworkController, CollectorProxy
from icarus.orchestration import Orchestrator
from icarus.registry import CACHE_POLICY, CACHE_PLACEMENT, CONTENT_PLACEMENT, DATA_COLLECTOR, RESULTS_WRITER, \
                            STRATEGY, TOPOLOGY_FACTORY, WORKLOAD
from icarus.runner import handler, _validate_settings
from icarus.util import config_logging, timestr, Settings


logger = logging.getLogger('icarusgym_actual_env')  # Setting the logger.


class IcarusActualEnv(ActualEnv, Orchestrator):
    """External environment class that is inherited from BaseActualEnv class of GymProxy.
    """
    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor that prepares an execution of Icarus simulation.

        :param kwargs: Dictionary of keyword arguments.
        """
        print("@@@@kwargs",kwargs)
        env_proxy = kwargs['env_proxy']
        ActualEnv.__init__(self, env_proxy)
        config = kwargs.get('kwargs') or kwargs
        print("ACTUAL_ENV config", config)
        # Handle nested config structure
        if 'config' in config:
            config_file = config['config'].get('config_path')
        else:
            config_file = config.get('config_path') if config else None
        if config_file and not os.path.isabs(config_file):
            # Convert relative path to absolute path
            config_file = os.path.abspath(config_file)
        print("CONFIG_PATH", config_file)
        # Handle nested config structure for output path
        if 'config' in config:
            output = config['config'].get('output_path')
        else:
            output = kwargs.get('kwargs', {}).get('output_path') or kwargs.get('output_path')
        config_override = None
        settings = Settings()
        settings.read_from(config_file)
        if config_override:
            for k, v in config_override.items():
                try:
                    v = eval(v)
                except NameError:
                    pass
                settings.set(k, v)

        # Config logger.
        config_logging(settings.LOG_LEVEL if 'LOG_LEVEL' in settings else 'INFO')

        # Validate settings.
        _validate_settings(settings, freeze=True)

        orch = self
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGABRT):
            signal.signal(sig, functools.partial(handler, settings, orch, output))
        logger.info('Launching orchestrator')
        Orchestrator.__init__(self, settings)

        # Create queue of experiment configurations.
        queue = collections.deque(settings.EXPERIMENT_QUEUE)

        # Calculate number of experiments and number of processes.
        self.n_exp = len(queue) * self.settings.N_REPLICATIONS
        self.n_proc = self.settings.N_PROCESSES if self.settings.PARALLEL_EXECUTION else 1

        logger.info('Starting simulations: %d experiments, %d process(es)' % (self.n_exp, self.n_proc))
        self._experiments = []
        while queue:
            self._experiments.append(queue.popleft())
        self._output = output

    def run(self, seed_:int, kwargs: Optional[dict] = None):
        """Runs the main control loop of Icarus simulation.

        :param kwargs: Dictionary of keyword argument.
        """
        logger.info('Run started!')
        # i = int(self.seq.current() / self.settings.N_REPLICATIONS) % len(self._experiments)
        # experiment = self._experiments[i]
        # self.experiment_callback(run_scenario(self.settings, experiment, self.seq.assign(), self.n_exp))
        # if self._stop:
        #     self.stop()
        #print("num_steps:", kwargs.get('num_steps'))
        print("!!! run")
        try:
            i = int(self.seq.current() / self.settings.N_REPLICATIONS) % len(self._experiments)
        except Exception as e:
            print("Error:", e)
            i = 0   
        # i = int(self.seq.current() / self.settings.N_REPLICATIONS) % 1)
        experiment = self._experiments[i]
        self.experiment_callback(run_scenario(self.settings, experiment, self.seq.assign(), self.n_exp))

        if self._stop:
            self.stop()
               
        # import numpy as np
        # _obs = np.zeros((5, 11), dtype=np.int32)  # 더미 observation
        # _info = {}  # 더미 info
        # terminated = False
        # truncated = False
        # reward = 0
        
        # print("RUN hello2:", seed_)
        # IcarusActualEnv.set_obs_and_reward(_obs, reward, terminated, truncated, _info)

    def finish(self, kwargs=None, **kw):
        """Finishes an execution of Icarus simulation.

        :param kwargs: Dictionary of keyword arguments.
        :param kw: Additional keyword arguments.
        """
        if kwargs is None:
            kwargs = kw
        
        # Skip finish if no experiments have actually been completed (early close from RLlib env checking)
        if self.n_success == 0 and self.n_fail == 0:
            logger.info('Skipping finish() - no experiments completed yet (likely early close from env checking)')
            return
            
        logger.info('END | Planned: %d, Completed: %d, Succeeded: %d, Failed: %d',
                    self.n_exp, self.n_fail + self.n_success, self.n_success, self.n_fail)
        logger.info('Orchestrator finished')
        orch = self
        settings = self.settings
        output = self._output
        results = orch.results
        RESULTS_WRITER[settings.RESULTS_FORMAT](results, output)
        logger.info('Saved results to file %s' % os.path.abspath(output))


def run_scenario(settings, params, curr_exp, n_exp):
    """Runs a single scenario experiment.

    :param settings: The simulation settings.
    :param params: Experiment parameters tree.
    :param curr_exp: Sequence number of the experiment.
    :param n_exp: Number of scheduled experiments.
    :return: 3-tuple.
        A (params, results, duration) 3-tuple. The first element is a dictionary which stores all the attributes of the
        experiment. The second element is a dictionary which stores the results. The third element is an integer
        expressing the wall-clock duration of the experiment (in seconds).
    """
    try:
        logger.info('run_scenario started')
        start_time = time.time()

        # Get list of metrics required.
        metrics = settings.DATA_COLLECTORS

        # Copy parameters so that they can be manipulated.
        tree = copy.deepcopy(params)

        # Set topology.
        topology_spec = tree['topology']
        topology_name = topology_spec.pop('name')
        if topology_name not in TOPOLOGY_FACTORY:
            logger.error('No topology factory implementation for %s was found.' % topology_name)
            return None
        topology = TOPOLOGY_FACTORY[topology_name](**topology_spec)

        workload_spec = tree['workload']
        workload_name = workload_spec.pop('name')
        if workload_name not in WORKLOAD:
            logger.error('No workload implementation named %s was found.' % workload_name)
            return None
        workload = WORKLOAD[workload_name](topology, **workload_spec)

        # Assign cache to nodes.
        if 'cache_placement' in tree:
            cachepl_spec = tree['cache_placement']
            cachepl_name = cachepl_spec.pop('name')
            if cachepl_name not in CACHE_PLACEMENT:
                logger.error('No cache placement named %s was found.' % cachepl_name)
                return None
            network_cache = cachepl_spec.pop('network_cache')
            # Cache budget is the cumulative number of cache entries across the whole network.
            cachepl_spec['cache_budget'] = workload.n_contents * network_cache
            CACHE_PLACEMENT[cachepl_name](topology, **cachepl_spec)

        # Assign contents to sources. If there are many contents, after doing this, performing operations requiring a
        # topology deep copy, i.e. to_directed/undirected, will take long.
        contpl_spec = tree['content_placement']
        contpl_name = contpl_spec.pop('name')
        if contpl_name not in CONTENT_PLACEMENT:
            logger.error('No content placement implementation named %s was found.' % contpl_name)
            return None
        CONTENT_PLACEMENT[contpl_name](topology, workload.contents, **contpl_spec)

        # Caching and routing strategy definition.
        strategy = tree['strategy']
        if strategy['name'] not in STRATEGY:
            logger.error('No implementation of strategy %s was found.' % strategy['name'])
            return None

        # Cache eviction policy definition.
        cache_policy = tree['cache_policy']
        if cache_policy['name'] not in CACHE_POLICY:
            logger.error('No implementation of cache policy %s was found.' % cache_policy['name'])
            return None

        # Configuration parameters of network model.
        netconf = tree['netconf']

        # Text description of the scenario run to print on screen.
        scenario = tree['desc'] if 'desc' in tree else "Description N/A"

        logger.info('Experiment %d/%d | Preparing scenario: %s', curr_exp, n_exp, scenario)

        if any(m not in DATA_COLLECTOR for m in metrics):
            logger.error('There are no implementations for at least one data collector specified')
            return None

        collectors = {m: {} for m in metrics}

        logger.info('Experiment %d/%d | Start simulation', curr_exp, n_exp)
        results = exec_experiment(topology, workload, netconf, strategy, cache_policy, collectors)

        duration = time.time() - start_time
        logger.info('Experiment %d/%d | End simulation | Duration %s.', curr_exp, n_exp, timestr(duration, True))
        return params, results, duration
    except KeyboardInterrupt:
        logger.error('Received keyboard interrupt. Terminating')
        sys.exit(-signal.SIGINT)
    except Exception as e:
        err_type = str(type(e)).split("'")[1].split(".")[1]
        err_message = e.args[0]
        logger.error('Experiment %d/%d | Failed | %s: %s\n%s',
                     curr_exp, n_exp, err_type, err_message,
                     traceback.format_exc())


def exec_experiment(topology, workload, netconf, strategy, cache_policy, collectors):
    """Executes the simulation of a specific scenario.

    :param topology: The FNSS topology object modeling topology on which experiments are run.
    :param workload: An iterable object whose elements are (time, event) tuples, where time is a float type indicating
    the timestamp of the event to be executed and event is a dictionary storing all the attributes of the event to
    execute.
    :param netconf: Dictionary of attributes to initialize the network model.
    :param strategy: Strategy definition. It is tree describing the name of the strategy to use and a list of
    initialization attributes.
    :param cache_policy: Cache policy definition. It is tree describing the name of the cache policy to use and a list
    of initialization attributes.
    :param collectors: The collectors to be used. It is a dictionary in which keys are the names of collectors to use
    and values are dictionaries of attributes for the collector they refer to.
    :return: A tree with the aggregated simulation results from all collectors.
    """
    logger.info('exec_experiment started')
    model = NetworkModel(topology, cache_policy, **netconf)
    view = NetworkView(model)
    logger.info('exec_experiment started 1')
    controller = NetworkController(model)
    logger.info('exec_experiment started 2')
    collectors_inst = [DATA_COLLECTOR[name](view, **params)
                       for name, params in collectors.items()]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)
    strategy_name = strategy['name']
    logger.info('exec_experiment started 3')
    strategy_args = {k: v for k, v in strategy.items() if k != 'name'}
    print(strategy_args)
    print(view)
    print(controller)
    print("strategy_name", strategy_name)
    print("STRATEGY ",STRATEGY)
    print("**strategy_args", strategy_args)
    logger.info('exec_experiment started 3.5')
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)
    logger.info("exec_experiment started 4")
    try:
        for time_, event in workload:
            strategy_inst.process_event(time_, **event)
    except Exception as e:
        if e.args and 'StopIteration' in e.args[0]:
            pass
        else:
            if e.args and 'TerminateGymProxy' in e.args[0]:
                logger.info('Terminating IcarusGym.')
                # Normal termination - don't exit with error code
                logger.info('Releasing lock and setting event...')
                try:
                    ActualEnv.env_proxy.release_lock()
                    logger.info('Lock released successfully')
                except Exception as lock_e:
                    logger.warning(f'Error releasing lock: {lock_e}')
                
                try:
                    ActualEnv.env_proxy.set_gym_env_event()
                    logger.info('Event set successfully')
                except Exception as event_e:
                    logger.warning(f'Error setting event: {event_e}')
                
                logger.info('Returning results...')
                return collector.results()
            else:
                logger.error(traceback.format_exc())
                ActualEnv.env_proxy.release_lock()
                ActualEnv.env_proxy.set_gym_env_event()
                exit(1)
    
    # Normal completion - call finish() on caches to signal episode end
    logger.info('Workload completed normally, calling finish on caches')
    try:
        # Call finish() on all caches to send termination signal to GymProxy
        logger.info(f'Number of caches: {len(controller.model.cache)}')
        for node, cache in controller.model.cache.items():
            logger.info(f'Checking cache at node {node}, type: {type(cache)}, has finish: {hasattr(cache, "finish")}')
            if hasattr(cache, 'finish'):
                logger.info(f'Calling finish() on cache at node {node}')
                cache.finish()
            else:
                logger.info(f'Cache at node {node} does not have finish() method')
    except Exception as e:
        logger.warning(f'Error calling finish on caches: {e}')
        import traceback
        logger.warning(traceback.format_exc())
    
    return collector.results()
