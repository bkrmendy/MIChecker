import math
from typing import List, Callable


def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + math.e ** (-z))


def sigmoid_prime(x):
    """Derivative of function"""
    return x * (1 - x)


def perceptron_output(weights: List[float],
                      inputs: List[float],
                      activation: Callable[[float], float]) -> float:
    """Calculates the output of a perceptron"""
    result = 0
    for (w, i) in zip(weights, inputs):
        result += w*i

    return activation(result)


def perceptron_error(output, target):
    """Calculated perceptron error to some desired output"""
    return (target - output) ** 2 / 2


def error_derivative(output: float, target: float, input: float, activation_prime: Callable[[float], float]) -> float:
    """Derivative of error with respect to weight of input"""
    return -1 * (target - output) * activation_prime(output) * input


def perceptron_next_weight(prev_weight: float, learning_rate: float, delta_err: float):
    """Calculates next weight for given weight"""
    return prev_weight - learning_rate * delta_err


