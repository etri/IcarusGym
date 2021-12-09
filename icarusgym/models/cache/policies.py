# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that defines additional caching policy classes used for IcarusGym. The original caching policy classes of
Icarus are defined in icarus.models.cache.policies module.
"""

__all__ = ['TtlCache', 'DecisionArrayCache']

import copy
import numpy as np

from icarus.models.cache.policies import Cache
from icarus.registry import register_cache_policy
from icarus.util import inheritdoc
from icarusgym import IcarusActualEnv
from icarusgym.from_contribs.python_priorityq.priorityq import MappedQueue
from typing import Dict, List, Tuple, Union


@register_cache_policy('ICARUSGYM_TTL')
class TtlCache(Cache):
    """Class of TTL cache used for IcarusGym. This is implemented based on the following reference:
    M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27,
    no. 3, pp. 1013-1027, May 2019 (Earlier version of the paper is presented in IEEE INFOCOM 2016).
    """
    class CacheInfo:
        """Class of cache information for constructing TTL cache.
        """
        def __init__(self, content: int, popularity: int = 0, expiration_time: float = 0.0, ttl: float = 0.0):
            """Constructor.

            :param content: Content ID.
            :param popularity: Popularity of content.
            :param expiration_time: Expiration time of content.
            :param ttl: TTL of content.
            """
            self._content = content
            self._popularity = popularity
            self._expiration_time = expiration_time
            self._ttl = ttl

        def __gt__(self, other):
            """Magic method for comparison of two cache information objects.
            For two cache objects x and y, x > y if x.expiration_time > y.expiration_time.
            """
            return self._expiration_time > other.expiration_time

        def __lt__(self, other):
            """Magic method for comparison of two cache information objects.
            For two cache objects x and y, x < y if x.expiration_time < y.expiration_time.
            """
            return self._expiration_time < other.expiration_time

        @property
        def content(self) -> int:
            """Getter for content property.

            :return: Content property.
            """
            return self._content

        @property
        def popularity(self) -> int:
            """Getter for popularity property.

            :return: Popularity property.
            """
            return self._popularity

        @property
        def expiration_time(self) -> float:
            """Getter for expiration time property.

            :return: Expiration time property.
            """
            return self._expiration_time

        @property
        def ttl(self) -> float:
            """Getter for TTL property.

            :return: TTL property.
            """
            return self._ttl

        @content.setter
        def content(self, content: int):
            self._content = content

        @popularity.setter
        def popularity(self, popularity: int):
            self._popularity = popularity

        @expiration_time.setter
        def expiration_time(self, expiration_time: float):
            self._expiration_time = expiration_time

        @ttl.setter
        def ttl(self, ttl: float):
            self._ttl = ttl

    def __init__(self, size: int, is_reset: bool = False, extra_cache_size_ratio: float = 0.001, **kwargs):
        """Constructor.

        :param size: Size of cache.
        :param is_reset: If it is true, the expiration time is extended for each cache hit. For example, suppose that
        the TTL of a cached content is 30s and its current remaining time for expiration is 10s. If the content is hit
        at that time, the remaining time of expiration is extended to 30s.
        :param extra_cache_size_ratio: Ratio between the size of extra and original caches. Extra cache is used for
        preventing from evicting a content with unexpired TTL due to the overflow of the original cache. For example,
        suppose that the original cache size is 100. If we set extra_cache_size_ratio of 0.05, the extra cache size
        becomes 5. For more details of theoretic aspects of extra cache, please refer Section IV-C in the reference
        mentioned at the beginning of this module.
        :param kwargs: Dictionary of keyword arguments.
        """
        # Dictionary of CacheInfo object indexed by content ID. _cache_infos[x] = y means that x is content ID and y is
        # the CacheInfo object that corresponds to x.
        self._cache_infos = {}

        # Queue of CacheInfo objects prioritized by remaining TTL. The top of the _min_expiration_pq is the CacheInfo
        # object that has the smallest remaining time of expiration.
        self._min_expiration_pq = MappedQueue()

        self._size = int(size)
        self._is_reset = is_reset
        self._current_time = 0
        if self._size <= 0:
            raise ValueError('maxlen must be positive')
        self._ttls = {}
        self._last_req_times = {}
        self._extra_cache_size_ratio = extra_cache_size_ratio
        self._node = None

    @inheritdoc(Cache)
    def __len__(self) -> int:
        return len(self._cache_infos)

    @property
    @inheritdoc(Cache)
    def maxlen(self) -> int:
        return self._size

    @property
    def node(self) -> int:
        """Getter for node ID property.

        :return: Node ID property.
        """
        return self._node

    @node.setter
    def node(self, node):
        self._node = node

    def dump(self) -> List[Tuple[int, float, float, int]]:
        """Returns a dump of all the elements currently in the cache possibly sorted according to content ID.

        :return List of tuples, each of which consists of four values: content, popularity, expiration_time, and TTL.
        The tuples are sorted in the ascending order of content IDs.
        """
        tuples = [(content, n.expiration_time, n.ttl, n.popularity)
                  for content, n in self._cache_infos.items()]
        return sorted(tuples, key=(lambda e: e[0]))

    def has(self, content: int, *args, **kwargs) -> bool:
        """Checks if a content is in the cache without changing the internal state of the cache.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the requested content is in the cache, false otherwise.
        """
        return True if self._cache_infos.get(content, None) else False

    def get(self, content: int, *args, **kwargs) -> bool:
        """Retrieves a content from the cache. Different from has(), calling this method may change the internal
        state of the cache depending on the specific cache implementation.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the requested content is in the cache, false otherwise.
        """
        cache_info = self._cache_infos.get(content, None)
        remaining_ttl = cache_info.expiration_time - self._current_time if cache_info else 0.
        hit = True if cache_info else False

        # Gets the TTL and cache size decision for the content from the agent.
        obs = self.get_obs(content, remaining_ttl, hit)
        reward = self.get_reward(hit)
        done = False
        info = self.get_info(content)
        action = IcarusActualEnv.get_action(obs, reward, done, info)
        ttl = action[0][0]
        cache_size = action[1][0]
        self._ttls[content] = ttl
        self._last_req_times[content] = self._current_time
        self.set_cache_size(int(cache_size * (1. + self._extra_cache_size_ratio)))

        if not hit:
            return False
        cache_info.popularity = cache_info.popularity + 1   # This means cache hit.

        # Updates _cache_infos and _min_expiration_pq if the cache uses reset mode.
        if self._is_reset:
            ttl_ = self._ttls[content]
            expiration_time_ = self._current_time + self._ttls[content]
            cache_info_ = copy.deepcopy(cache_info)
            cache_info_.expiration_time = expiration_time_
            cache_info_.ttl = ttl_
            self._min_expiration_pq.update(cache_info, cache_info_)
            self._cache_infos[content] = cache_info_

        return True

    def put(self, content: int, *args, **kwargs) -> Union[int, None]:
        """Inserts a content into the cache if it does not existed in the cache. If the content is already stored in
        the cache, it will not be inserted again, but, the internal state of the cache object may change.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: The evicted object if a content is evicted. None otherwise.
        """
        if self.has(content):   # Returns None if the cache already has content.
            return None

        # Creates a new CacheInfo object.
        popularity = 1
        ttl = self._ttls[content]
        expiration_time = self._current_time + ttl
        info = TtlCache.CacheInfo(content, popularity, expiration_time, ttl)

        evicted = None

        if len(self) == self._size:    # When the cache is full.

            # When the expiration time of new content is less than those of contents in the cache.
            if (len(self._min_expiration_pq) == 0 or
                    self._min_expiration_pq.h[0].expiration_time > info.expiration_time):
                del info   # Discards the new content.
                return None
            else:   # Evicts the content that has the smallest expiration time.
                info_ = self._min_expiration_pq.pop()
                content_ = info_.content
                self._cache_infos.pop(content_)
                del info_
                evicted = content_

        # Inserts the new CacheInfo object to the internal data structure.
        self._cache_infos[content] = info
        self._min_expiration_pq.push(info)

        return evicted

    def remove(self, content: int, *args, **kwargs) -> bool:
        """Removes a content from the cache, if it exists in the cache.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the content was in the cache, false otherwise.
        """
        info = self._cache_infos.get(content, None)
        if not info:
            return False
        self._cache_infos.pop(content)
        self._min_expiration_pq.remove(info)
        del info
        return True

    @inheritdoc(Cache)
    def clear(self):
        self._cache_infos.clear()
        while len(self._min_expiration_pq) > 0:
            self._min_expiration_pq.pop()

    def update(self, time: float, content: int):
        """Updates the time variable and removes expired contents in the internal data structure.

        :param time: Current time.
        :param content: Content ID.
        """
        self._current_time = time
        while self._min_expiration_pq.h and self._min_expiration_pq.h[0].expiration_time < time:
            info = self._min_expiration_pq.pop()
            content_ = info.content
            del self._cache_infos[content_]

    def set_cache_size(self, size: int):
        """Sets the size of cache.

        :param size: Size of cache.
        """
        self._size = int(size)
        while len(self) > self._size:
            info = self._min_expiration_pq.pop()
            del self._cache_infos[info.cid]
            del info

    def get_obs(self, content: int, remaining_ttl: float, hit: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Prepares an observation object to be given to the agent.

        :param content: Content ID.
        :param remaining_ttl: Remaining TTL.
        :param hit: True if the content is hit in the cache.
        :return: Tuple that consists of three numpy arrays and one boolean variable. The each of the first three arrays
        has the value of _current_time, content, and remaining_ttl, respectively. The boolean variable indicates
        whether the requested content is hit or not in the cache.
        """
        # An observation is a tuple that consists of four values: env_time, content, remaining_ttl, and hit. 'env_time'
        # is the current time of caching simulation. 'content' is the ID of requested content. 'remaining_ttl' is the
        # remaining time until when the requested is removed. 'hit' becomes 1 when the requested content is hit in the
        # cache, 0 for the case of cache miss.
        return (np.array([self._current_time], dtype=np.float),
                np.array([content], dtype=np.uint32),
                np.array([remaining_ttl], dtype=np.float),
                1 if hit else 0)

    @staticmethod
    def get_reward(hit) -> float:
        """Calculates a reward to be given to the agent.

        :param hit: True if the requested content is hit in the cache.
        :return: 1.0 if hit is true, 0.0 otherwise.
        """
        return float(hit)

    def get_info(self, content: int = 0) -> Dict:
        """Prepares an information dictionary to be given to the agent.

        :return: Dictionary that includes environment time, cache size, and statistics related to cache hits and misses
        for given content.
        """
        from icarusgym import IcarusGymLce
        cache_size = self.maxlen
        cache_used = len(self)
        if content == 0:
            info = {'cache_size': cache_size, 'cache_used': cache_used}
        else:
            n_cache_hits = IcarusGymLce.internal_collector.get_num_cache_hits(self._node, content)
            n_cache_misses = IcarusGymLce.internal_collector.get_num_cache_misses(self._node, content)
            info = {'cache_size': cache_size, 'cache_used': cache_used,
                    'n_cache_hits': n_cache_hits, 'n_cache_misses': n_cache_misses}
        return info

    def finish(self):
        """Signals the end of an episode.
        """
        # Meaningless observation and reward.
        obs = (np.array([0.0], dtype=np.float), np.array([0], dtype=np.uint32), np.array([0.], dtype=np.float), 0)
        reward = 0.

        done = True             # Indicates the end of episode.
        info = self.get_info()  # Valid information dictionary.
        IcarusActualEnv.set_obs_and_reward(obs, reward, done, info)


@register_cache_policy('ICARUSGYM_DECISION_ARRAY')
class DecisionArrayCache(Cache):
    """Class of decision array cache used for IcarusGym. The decision array cache environment is implemented based on
    the following reference:
    A. Sadeghi et al., "Deep reinforcement learning for adaptive caching in hierarchical content delivery networks,"
    IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 4, pp. 1024-1033, Dec. 2019.
    The action of an agent can be encoded as a 0/1-value array, that indicates, for each content for each node, whether
    the content is cached or not at the node. Namely, 'decision[n, c] = 1' means that content c is stored at the cache
    of node n. 'decision[n, c] = 0' means otherwise. If node n is not cache router, 'requests[n, c] = 1' does not
    effective since n does not have cache. We use this somewhat wasteful definition due to implementation convenience.
    """
    @inheritdoc(Cache)
    def __init__(self, size: int, *args, **kwargs):
        """Constructor.

        :param size: Size of cache.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        """
        self._cache = set()
        self._maxlen = int(size)
        self._node = None
        if self._maxlen <= 0:
            raise ValueError('maxlen must be positive')

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    def node(self) -> int:
        """Getter for node ID property.

        :return: Node ID property.
        """
        return self._node

    @node.setter
    def node(self, node: int):
        self._node = node

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return sorted(self._cache)

    def has(self, content, *args, **kwargs):
        """Checks if a content is in the cache without changing the internal state of the caching object.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the requested content is in the cache, false otherwise.
        """
        return content in self._cache

    def get(self, content: int, *args, **kwargs) -> bool:
        """Retrieves a content from the cache.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the requested content is in the cache, false otherwise.
        """
        return self.has(content)

    def put(self, content: int, *args, **kwargs) -> None:
        """Inserts a content in the cache if it does not exist in the cache. If the content is already stored in the
        cache, it will not be inserted again, but, the internal state of the cache object may change.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: The decision array cache always returns None since its eviction is conducted by remove() method.
        """
        if not self.has(content):
            self._cache.add(content)
        return None

    def remove(self, content: int, *args, **kwargs) -> bool:
        """Removes a content from the cache, if it exists in the cache.

        :param content: Content ID.
        :param args: List of arguments.
        :param kwargs: Dictionary of keyword arguments.
        :return: True if the content was in the cache, false otherwise.
        """
        if content in self._cache:
            self._cache.remove(content)
            return True
        else:
            return False

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
