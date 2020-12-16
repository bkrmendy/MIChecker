from collections import namedtuple
from typing import List

Edge = namedtuple("Edge", "weight, value")


def utility(edges: List[Edge]):
    return sum(map(lambda e: e.weight * e.value, edges))


def next_utility(reward: float, leszamitolas: float, expected_utility: float):
    return  reward + leszamitolas * expected_utility

