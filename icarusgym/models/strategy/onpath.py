# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that defines additional on-path caching strategy classes used for IcarusGym. The original on-path caching
strategy classes of Icarus are defined in icarus.models.strategy.onpath module.
"""

__all__ = ['IcarusGymLce', 'PeriodicUpdateEverywhere', 'PassiveIcarusGymLce']

import copy
import math
import numpy as np

from abc import ABC
from icarus.execution.network import NetworkController, NetworkView
from icarus.models import Cache
from icarus.models import LeaveCopyEverywhere, Strategy
from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links
from icarusgym.actual_env import IcarusActualEnv
from icarusgym import DecisionArrayCache, PerNodeAndContentStatsCollector
from typing import Dict


@register_strategy('ICARUSGYM_LCE')
class IcarusGymLce(LeaveCopyEverywhere):
    """Leave copy everywhere (LCE) caching strategy class adapted for IcarusGym. This strategy is same as the ordinary
    LCE strategy except for that caches are controlled by agent.
    """
    internal_collector = None

    def __init__(self, view: NetworkView, controller: NetworkController, content_max: int,
                 stats_interval: float = np.inf, **kwargs):
        """Constructor.

        :param view: View object of Icarus.
        :param controller: Controller object of Icarus.
        :param content_max: Maximum content ID.
        :param stats_interval: Interval of collecting statistics.
        :param kwargs: Dictionary of keyword arguments.
        """
        super(IcarusGymLce, self).__init__(view, controller)
        self._current_time = 0.
        self._prev_time = 0.
        self._stats_interval = stats_interval

        # Prepares an internal collector for constructing an observation and information to be given to the agent.
        self._internal_collector = PerNodeAndContentStatsCollector(view, content_max)
        self.controller.collector.collectors['cache_hit'].append(self._internal_collector)
        self.controller.collector.collectors['cache_miss'].append(self._internal_collector)
        self.__class__.internal_collector = self._internal_collector

        # Sets the node ID if a cache in the network has attribute named as node.
        for key, cache in self.controller.model.cache.items():
            if hasattr(cache, 'node'):
                cache.node = key

    def __del__(self):
        """Destructor. Cache policy used with this strategy should have method finish() that signals the end of an
        episode to the agent through method IcarusActualEnv.set_obs_and_reward().
        """
        cache = list(self.controller.model.cache.values())[0]
        cache.finish()

    @inheritdoc(Strategy)
    def process_event(self, time: float, receiver: int, content: int, log: bool):
        self._update(time, content)     # Updates time-related variables in the network.

        # Gets all required data.
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        serving_node = None

        # Routes a content request to the original source and queries the caches on the path.
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if v != source:
                if self.view.has_cache(v) and self.controller.get_content(v):
                    serving_node = v
                    break
            else:   # No cache hits, gets the content from the source.
                self.controller.get_content(v)
                serving_node = v

        # Returns the content.
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                self.controller.put_content(v)  # Inserts the content.
        self.controller.end_session()

    def _update(self, time: float, content: int):
        """Updates time-related variables of the elements in the network.

        :param time: Current time.
        :param content: Content ID.
        """
        self._prev_time = self._current_time
        self._current_time = time
        if (int(self._current_time / self._stats_interval) - int(self._prev_time / self._stats_interval)) >= 1:
            self._internal_collector.clear()
        for cache in self.controller.model.cache.values():
            cache.update(time, content)


@register_strategy('ICARUSGYM_PUE')
class PeriodicUpdateEverywhere(Strategy):
    """Periodic update everywhere (PUE) strategy. This strategy periodically updates every cache in a network in
    accordance with a binary decision array from the agent. Namely, decision[n, c] == 1 if agent decides to store
    content c in the cache of node n and decision[n, c] == 0 otherwise.
    """
    class PuePerNodeAndContentStatsCollector(PerNodeAndContentStatsCollector):
        """Collector that measures the number of content fetching overheads for each content for each node.
        """
        def __init__(self, view: NetworkView, content_max: int):
            """Constructor.

            :param view: Network view object.
            :param content_max: Maximum content ID.
            """
            super(PeriodicUpdateEverywhere.PuePerNodeAndContentStatsCollector, self).__init__(view, content_max)
            self._fetch_req_overheads = np.zeros((content_max + 1,), dtype=np.uint32)
            self._fetch_res_overheads = np.zeros((content_max + 1,), dtype=np.uint32)

        @inheritdoc(PerNodeAndContentStatsCollector)
        def request_hop(self, u, v, main_path=True):
            content = self.curr_cont
            self._fetch_req_overheads[content] += 1

        @inheritdoc(PerNodeAndContentStatsCollector)
        def content_hop(self, u, v, main_path=True):
            content = self.curr_cont
            self._fetch_res_overheads[content] += 1

        def get_fetch_req_overheads(self) -> np.ndarray:
            """Gets the array that stores the amount of fetch request overheads for every content.

            :return: A copy of _fetch_req_overheads.
            """
            return np.copy(self._fetch_req_overheads)

        def get_fetch_res_overheads(self) -> np.ndarray:
            """Gets the array that stores the amount of fetch response overheads for every content.

            :return: A copy of _fetch_res_overheads.
            """
            return np.copy(self._fetch_res_overheads)

        def clear(self):
            """Clears all the statistics.
            """
            super(PeriodicUpdateEverywhere.PuePerNodeAndContentStatsCollector, self).clear()
            self._fetch_req_overheads.fill(0)
            self._fetch_res_overheads.fill(0)

    def __init__(self, view: NetworkView, controller: NetworkController, content_max: int, decision_interval: float,
                 **kwargs):
        """Constructor.

        :param view: View object of Icarus.
        :param controller: Controller object of Icarus.
        :param content_max: Maximum content ID.
        :param decision_interval: Agent's decision interval.
        :param kwargs: Dictionary of keyword arguments.
        """
        super(PeriodicUpdateEverywhere, self).__init__(view, controller)
        self._caching_decision_array = None
        self._current_time = 0.
        self._prev_time = (-1.) * decision_interval
        self._content_max = content_max
        self._decision_interval = decision_interval

        # Prepares an internal collector for constructing an observation and information to be given to the agent.
        self._internal_collector = self.__class__.PuePerNodeAndContentStatsCollector(view, content_max)
        self.controller.collector.collectors['start_session'].append(self._internal_collector)
        self.controller.collector.collectors['cache_hit'].append(self._internal_collector)
        self.controller.collector.collectors['cache_miss'].append(self._internal_collector)
        self.controller.collector.collectors['request_hop'].append(self._internal_collector)
        self.controller.collector.collectors['content_hop'].append(self._internal_collector)
        self.__class__.internal_collector = self._internal_collector

        for key, cache in self.controller.model.cache.items():
            if hasattr(cache, 'node'):
                cache.node = key

    def __del__(self):
        """Destructor that signals the end of an episode.
        """
        obs = self._get_obs()
        reward = self._get_reward()
        done = True
        info = self._get_info()
        IcarusActualEnv.set_obs_and_reward(obs, reward, done, info)

    @inheritdoc(Strategy)
    def process_event(self, time: float, receiver: int, content: int, log: bool):
        self._current_time = time

        # If the decision interval is elapsed from the last decision, updates all caches in the network according to
        # the caching decision array from the agent.
        while ((int(self._current_time / self._decision_interval) -
                int(self._prev_time / self._decision_interval)) >= 1):
            obs = self._get_obs()
            reward = self._get_reward()
            done = False
            info = self._get_info()
            action = IcarusActualEnv.get_action(obs, reward, done, info)    # Gets the decision vector from the agent.
            self._caching_decision_array = action
            self._internal_collector.clear()
            self._update_caches()
            self._prev_time = (math.floor(self._prev_time / self._decision_interval) * self._decision_interval +
                               self._decision_interval)

        # Gets all required data.
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        serving_node = None

        # Routes a cache request to the original source and queries the caches on the path.
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if v != source:
                if self.view.has_cache(v) and self.controller.get_content(v):
                    serving_node = v
                    break
            else:   # No cache hits, get the content from the source.
                self.controller.get_content(v)
                serving_node = v

        # Returns the content.
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        self.controller.end_session()

    def _get_obs(self) -> np.ndarray:
        """Gets an observation to be given to the agent.

        :return: Numpy array. Namely, requests[n, c] is the number of requests for content c to the cache at node n for
        last decision interval.
        """
        hits = self._internal_collector.get_cache_hit_array()
        misses = self._internal_collector.get_cache_miss_array()
        requests = hits + misses
        return requests

    def _get_reward(self) -> float:
        """Gets a reward to be given to the agent.

        :return: Aggregate number of fetch responses for a decision interval.
        """
        fetch_res_overheads = self._internal_collector.get_fetch_res_overheads()
        agg_fetch_res_overheads = np.sum(fetch_res_overheads)
        return float(agg_fetch_res_overheads)

    def _get_info(self) -> Dict:
        """Gets information to be given to the agent.

        :return: A dictionary that consists of four numpy arrays: hits, misses, fetch_req_overheads, and
        fetch_res_overheads.
            hits[n, c] is the number of cache hits for content c at node n for last decision interval.
            request[n, c] is the number of cache misses for content c at node n for last decision interval.
            fetch_req_overheads[n, c] is the amount of fetch request overheads for content c at node n for last
            decision interval.
            fetch_res_overheads[n, c] is the amount of fetch response overheads for content c at node n for last
            decision interval.
        """
        hits = self._internal_collector.get_cache_hit_array()
        misses = self._internal_collector.get_cache_miss_array()
        fetch_req_overheads = self._internal_collector.get_fetch_req_overheads()
        fetch_res_overheads = self._internal_collector.get_fetch_res_overheads()
        return {'hits': hits,
                'misses': misses,
                'fetch_req_overheads': fetch_req_overheads,
                'fetch_res_overheads': fetch_res_overheads}

    def _update_caches(self):
        """Updates caches in the network according to a caching decision array from the agent.
        """
        for cache in self.controller.model.cache.values():
            for content in range(1, self._content_max + 1):
                if self._caching_decision_array[cache.node, content]:
                    self._prefetch_content(cache, content)

    def _prefetch_content(self, cache: DecisionArrayCache, content: int):
        """Pre-fetches a content from the source to a cache.

        :param cache: Cache object.
        :param content: Content ID.
        """
        if cache.has(content):
            return
        source = self.view.content_source(content)
        receiver = cache.node
        path = self.view.shortest_path(receiver, source)
        serving_node = None
        for u, v in path_links(path):
            if v != source:
                if self.view.has_cache(v) and self.controller.model.cache[v].has(content):
                    serving_node = v
                    break
            else:   # No cache hits, get the content from the source.
                serving_node = v

        # Returns content.
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            if self.view.has_cache(v):
                if self._caching_decision_array[v, content]:
                    cache.put(content)
                else:
                    cache.remove(content)


@register_strategy('PASSIVE_ICARUSGYM_LCE')
class PassiveIcarusGymLce(IcarusGymLce):
    """Leave copy everywhere (LCE) caching strategy class adapted for passive agent. The 'passive agent' means that the
    agent does not actually control the network caching. The environment controls the network caching by using the
    legacy caching algorithms (e.g. LFU, LRU, FIFO, etc.) provided by original Icarus. In other words, actions from the
    passive agent do not have influence on the network caching. This environment is useful when we need to compare the
    performances of reinforcement learning-based control and legacy control in a same scenario.
    """
    class PassiveAgentCache(Cache, ABC):
        """Abstract class that declares attributes and methods required for the passive agent cache policy.
        """
        def __init__(self, size: int):
            """Constructor.

            :param size: Size of cache.
            """
            Cache.__init__(self, size)
            self._current_time = 0.
            self._node = None

        def update(self, time: float, content: int):
            """Updates the time variable.

            :param time: Current time.
            :param content: Content ID.
            """
            self._current_time = time

        def get_obs(self, content: int, hit: bool) -> tuple:
            """Prepares an observation to be given to the agent.

            :param content: Content ID.
            :param hit: Whether a requested content is hit in the cache or not.
            :return: An observation is a tuple of two numpy arrays and a boolean variable.
            """
            return (np.array([self._current_time], dtype=np.float),
                    np.array([content], dtype=np.uint32),
                    1 if hit else 0)

        @staticmethod
        def get_reward(hit) -> float:
            """Calculates a reward to be given to the agent.

            :param hit: Whether a requested content is hit in the cache or not.
            :return: 1.0 if 'hit' is true, 0.0 otherwise.
            """
            return float(hit)

        def get_info(self, content: int = 0) -> dict:
            """Prepares an information to be given to the agent.

            :param content: Content ID.
            :return: A dictionary that consists of four numbers: cache_size, cache_used, n_cache_hits, and
            n_cache_misses.
                cache_size is the size of the cache.
                cache_used is the number of contents stored in the cache.
                n_cache_hits is the number of cache hits.
                n_cache_misses is the number of cache misses.
            """
            cache_size = self.maxlen
            cache_used = len(self)
            if content == 0:
                info = {'cache_size': cache_size, 'cache_used': cache_used}
            else:
                n_cache_hits = PassiveIcarusGymLce.internal_collector.get_num_cache_hits(self._node, content)
                n_cache_misses = PassiveIcarusGymLce.internal_collector.get_num_cache_misses(self._node, content)
                info = {'cache_size': cache_size, 'cache_used': cache_used,
                        'n_cache_hits': n_cache_hits, 'n_cache_misses': n_cache_misses}
            return info

        def finish(self):
            """Delivers an end-of-episode signal to the agent through method IcarusActualEnv.set_obs_and_reward().
            """
            obs = (np.array([np.infty], dtype=np.float), np.array([0], dtype=np.uint32), 0)
            reward = 0.
            done = True
            info = self.get_info()
            IcarusActualEnv.set_obs_and_reward(obs, reward, done, info)

    @inheritdoc(IcarusGymLce)
    def __init__(self, view: NetworkView, controller: NetworkController, content_max: int,
                 stats_interval: float = np.inf, **kwargs):
        super().__init__(view, controller, content_max, stats_interval, **kwargs)

        # Transforms all cache in the network to passive agent caches.
        passive_agent_caches = {}
        for key, cache in self.controller.model.cache.items():
            passive_agent_cache = self.transform_cache(cache, key)
            passive_agent_caches[key] = passive_agent_cache
        for key, cache in passive_agent_caches.items():
            self.controller.model.cache[key] = cache

    @staticmethod
    def transform_cache(cache: PassiveAgentCache, node: int):
        """Dynamically transforms a conventional caching policy (e.g. LFU, LRU, FIFO, etc.) implemented in the
        original Icarus to a PassiveAgentCache object. Since the transformed cache self-determines its action, the
        actions from the agent are ignored.

        :param cache: Cache object that self-determines its action (e.g. LFU, LRU, and FIFO).
        :param node: Node ID.
        :return: Transformed cache object.
        """
        if not isinstance(cache, Cache):
            raise TypeError('cache must be an instance of Cache or its subclasses')
        cache = copy.deepcopy(cache)
        cache_cls = cache.__class__
        c_get = cache.get

        # Dynamically transforms the cache object to a PassiveAgentCache object.
        setattr(cache, '_current_time', 0.)
        setattr(cache, '_node', node)
        setattr(cache_cls, 'update', PassiveIcarusGymLce.PassiveAgentCache.update)
        setattr(cache_cls, 'get_obs', PassiveIcarusGymLce.PassiveAgentCache.get_obs)
        setattr(cache_cls, 'get_reward', PassiveIcarusGymLce.PassiveAgentCache.get_reward)
        setattr(cache_cls, 'get_info', PassiveIcarusGymLce.PassiveAgentCache.get_info)

        def get(content: int, *args, **kwargs) -> bool:
            """Retrieves a content from the cache. Different from method has(), calling this method may change the
            internal state of the caching object depending on the specific cache implementation.

            :param content: Content ID.
            :param args: List of arguments.
            :param kwargs: Dictionary of keyword arguments.
            :return: True if the requested content is in the cache, false otherwise.
            """
            hit = c_get(content, *args, **kwargs)
            obs = cache.get_obs(content, hit)
            reward = cache.__class__.get_reward(hit)
            done = False
            info = cache.get_info(content)
            IcarusActualEnv.get_action(obs, reward, done, info)
            return hit

        cache.get = get
        cache.get.__doc__ = c_get.__doc__
        setattr(cache_cls, 'finish', PassiveIcarusGymLce.PassiveAgentCache.finish)
        return cache
