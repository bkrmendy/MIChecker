import math
from typing import List, Callable


def sigmoid(z):
    return 1 / (1 + math.e ** (-z))


def perceptron_output(weights: List[float],
                      inputs: List[float],
                      activation: Callable[[float], float]) -> float:
    """Calculates the output of a perceptron"""
    result = 0
    for (w, i) in zip(weights, inputs):
        result += w*i

    return activation(result)


