from collections import namedtuple
from typing import List

Edge = namedtuple("Edge", "weight, value")


def utility(edges: List[Edge]):
    """Calculates utility for action, based on edges pointing out from a starting position"""
    return sum(map(lambda e: e.weight * e.value, edges))


def next_utility(reward: float, leszamitolas: float, max_expected_utility: float):
    """Calculates next utility value based on first two parameters and max expected utility"""
    return reward + leszamitolas * max_expected_utility

