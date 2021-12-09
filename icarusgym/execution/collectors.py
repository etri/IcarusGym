# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module that defines additional collector classes used for IcarusGym. The original collector classes of Icarus are
defined in icarus.execution.collectors module.
"""

__all__ = ['PerNodeAndContentStatsCollector']

import numpy as np

from icarus.execution import DataCollector, NetworkView
from icarus.registry import register_data_collector
from icarus.util import inheritdoc


@register_data_collector('PER_NODE_AND_CONTENT_STATS')
class PerNodeAndContentStatsCollector(DataCollector):
    """Collector that measures the number of cache hits and misses for each content for each node.
    """
    def __init__(self, view: NetworkView, content_max: int):
        """Constructor.

        :param view: Network view object.
        :param content_max: Maximum content ID.
        """
        super(PerNodeAndContentStatsCollector, self).__init__(view, off_path_hits=False, per_node=True,
                                                              content_hits=True)
        node_max = max(view.model.topology.nodes())
        self._cache_hits = np.zeros((node_max + 1, content_max + 1), dtype=np.uint32)
        self._cache_misses = np.zeros((node_max + 1, content_max + 1), dtype=np.uint32)
        self.curr_cont = None

    @inheritdoc(DataCollector)
    def start_session(self, timestamp: float, receiver: int, content: int):
        self.curr_cont = content

    @inheritdoc(DataCollector)
    def cache_hit(self, node: int):
        content = self.curr_cont
        self._cache_hits[node, content] += 1

    @inheritdoc(DataCollector)
    def cache_miss(self, node: int):
        content = self.curr_cont
        self._cache_misses[node, content] += 1

    def get_num_cache_hits(self, node: int, content: int) -> int:
        """Gets the number of cache hits for given content at given node.

        :param node: Node ID.
        :param content: Content ID.
        :return: Number of cache hits.
        """
        return self._cache_hits[node, content]

    def get_num_cache_misses(self, node: int, content: int) -> int:
        """Gets the number of cache misses for given content at given node.

        :param node: Node ID.
        :param content: Content ID.
        :return: Number of cache misses.
        """
        return self._cache_misses[node, content]

    def get_cache_hit_array(self) -> np.ndarray:
        """Gets the array of cache hits.

        :return: Array of cache hits.
        """
        return np.copy(self._cache_hits)

    def get_cache_miss_array(self) -> np.ndarray:
        """Gets the array of cache misses.

        :return: Array of cache misses.
        """
        return np.copy(self._cache_misses)

    def clear(self):
        """Clears all the statistics.
        """
        self._cache_hits.fill(0)
        self._cache_misses.fill(0)
