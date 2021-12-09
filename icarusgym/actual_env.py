# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that ports the main control loop of Icarus to Gym through GymProxy.
"""

import collections
import copy
import functools
import logging
import os
import signal
import sys
import time
import traceback

from gymproxy import BaseActualEnv
from icarus.execution import NetworkModel, NetworkView, NetworkController, CollectorProxy
from icarus.orchestration import Orchestrator
from icarus.registry import CACHE_POLICY, CACHE_PLACEMENT, CONTENT_PLACEMENT, DATA_COLLECTOR, RESULTS_WRITER, \
                            STRATEGY, TOPOLOGY_FACTORY, WORKLOAD
from icarus.runner import handler, _validate_settings
from icarus.util import config_logging, timestr, Settings


logger = logging.getLogger('icarusgym_actual_env')  # Setting the logger.


class IcarusActualEnv(BaseActualEnv, Orchestrator):
    """External environment class that is inherited from BaseActualEnv class of GymProxy.
    """
    def __init__(self, **kwargs):
        """Constructor that prepares an execution of Icarus simulation.

        :param kwargs: Dictionary of keyword arguments.
        """
        env_proxy = kwargs['env_proxy']
        BaseActualEnv.__init__(self, env_proxy)
        config = kwargs['config']
        config_file = config['config_path']
        output = config['output_path']
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

    def run(self, **kwargs):
        """Runs the main control loop of Icarus simulation.

        :param kwargs: Dictionary of keyword argument.
        """
        i = int(self.seq.current() / self.settings.N_REPLICATIONS) % len(self._experiments)
        experiment = self._experiments[i]
        self.experiment_callback(run_scenario(self.settings, experiment, self.seq.assign(), self.n_exp))
        if self._stop:
            self.stop()

    def finish(self, **kwargs):
        """Finishes an execution of Icarus simulation.

        :param kwargs: Dictionary of keyword arguments.
        """
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
    model = NetworkModel(topology, cache_policy, **netconf)
    view = NetworkView(model)
    controller = NetworkController(model)
    collectors_inst = [DATA_COLLECTOR[name](view, **params)
                       for name, params in collectors.items()]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)
    strategy_name = strategy['name']
    strategy_args = {k: v for k, v in strategy.items() if k != 'name'}
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)
    try:
        for time_, event in workload:
            strategy_inst.process_event(time_, **event)
    except Exception as e:
        if e.args and 'StopIteration' in e.args[0]:
            pass
        else:
            if e.args and 'TerminateGymProxy' in e.args[0]:
                logger.info('Terminating IcarusGym.')
            else:
                logger.error(traceback.format_exc())
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)
    return collector.results()
