import math
from collections import namedtuple

from typing import List

NodeData = namedtuple("NodeData", "p, n")


def I(data: NodeData) -> float:
    """Információ-igény"""
    if data.n == 0 or data.p == 0:
        return 0
    p_over = (data.p / (data.p + data.n))
    n_over = (data.n / (data.p + data.n))
    return - p_over * math.log(p_over, 2) - n_over * math.log(n_over, 2)


def Nyereseg(data: List[NodeData]) -> float:
    p_total = sum(map(lambda d: d.p, data))
    n_total = sum(map(lambda d: d.n, data))

    result = 0
    for subtree in data:
        result += (subtree.p + subtree.n) / (p_total + n_total) * I(subtree)

    return result