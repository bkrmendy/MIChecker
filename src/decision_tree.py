import math
from collections import namedtuple
from typing import List


NodeData = namedtuple("NodeData", "p, n")

#
# I function
#
def I(data: NodeData) -> float:
    """Információigény részfára"""
    if data.n == 0 or data.p == 0:
        return 0
    p_over = (data.p / (data.p + data.n))
    n_over = (data.n / (data.p + data.n))
    return - p_over * math.log(p_over, 2) - n_over * math.log(n_over, 2)


def Maradek(data: List[NodeData]) -> float:
    """Információigény egy csomópont összes részfájára"""
    p_total = sum(map(lambda d: d.p, data))
    n_total = sum(map(lambda d: d.n, data))

    result = 0
    for subtree in data:
        result += (subtree.p + subtree.n) / (p_total + n_total) * I(subtree)

    return result

#
# Binary classification
#
BinaryClassificationMetrics = namedtuple("BinaryClassificationMetrics",
                                         "true_positive_rate, false_positive_rate")

BinaryClassificationDescription = namedtuple("BinaryClassificationDescription",
                                             "n_positive, n_negative, ratio_actual_positive, ratio_actual_negative")


def calculate_binary_classification_metrics(desc: BinaryClassificationDescription) -> BinaryClassificationMetrics:
    """Bináris osztályzó kiértékelése"""
    n_actual_positive = desc.n_positive * desc.ratio_actual_positive
    n_actual_negative = desc.n_negative * desc.ratio_actual_negative
    true_positive_rate = n_actual_positive / (n_actual_positive + desc.n_negative - n_actual_negative)
    true_negative_rate = n_actual_negative / (n_actual_negative + desc.n_positive - n_actual_positive)
    return BinaryClassificationMetrics(true_positive_rate, true_negative_rate)


#
# Sample compexity
#
def sample_complexity(hypothesis_space: int, max_error_rate: float, margin: float) -> float:
    """Calculates sample complexity function
        Margin has to be input as 1 - delta
        Example:
            hypothesis_space    = 10^5
            epsilon             = 2%
            guaranteed rate     = 97%

            sample_complexity(10 ** 5, 0.02, 0.03)"""
    return (1 / max_error_rate) * (math.log(1 / margin) + math.log(abs(hypothesis_space)))

