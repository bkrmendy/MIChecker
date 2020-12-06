import unittest
from .perceptron import perceptron_output, \
                        perceptron_error, \
                        perceptron_next_weight, \
                        error_derivative, \
                        sigmoid, \
                        sigmoid_prime


class PerceptronTestCases(unittest.TestCase):
    def test_perceptron_output(self):
        """2. ZH 13. feladat"""
        output = perceptron_output([-0.92, -0.01, 0.37], [1, -0.82, -0.15], sigmoid)
        self.assertAlmostEqual(output, 0.2754, 4)

    def test_perceptron_error(self):
        """2. ZH 14. feladat"""
        output = perceptron_output([-0.92, -0.01, 0.37], [1, -0.82, -0.15], sigmoid)
        error = perceptron_error(output, 0.19)
        self.assertAlmostEqual(error, 0.0036, 4)

    def test_error_derivative(self):
        """2. ZH 15. feladat"""
        w1 = 0.37
        i1 = -0.15
        target = 0.19
        inputs = [1, -0.82, i1]
        output = perceptron_output([-0.92, -0.01, w1], inputs, sigmoid)
        derivative = error_derivative(output, target, i1, sigmoid_prime)
        self.assertAlmostEqual(derivative, -0.0026, 4)

    def test_next_weight(self):
        """2. ZH 16. feladat"""
        w1 = 0.37
        i1 = -0.15
        target = 0.19
        inputs = [1, -0.82, i1]
        output = perceptron_output([-0.92, -0.01, w1], inputs, sigmoid)
        derivative = error_derivative(output, target, i1, sigmoid_prime)
        next_weight_for_w1 = perceptron_next_weight(w1, derivative, 0.33)
        self.assertAlmostEqual(next_weight_for_w1, 0.3708, 4)
