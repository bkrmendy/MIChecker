import math


def I(p: float, n: float) -> float:
    """Információ-igény"""
    p_over = (p / (p + n))
    n_over = (n / (p + n))
    return - p_over * math.log(p_over, 2) - n_over * math.log(n_over, 2)

