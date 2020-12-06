import unittest
from .perceptron import perceptron_output, sigmoid

class MyTestCase(unittest.TestCase):
    def test_something(self):
        output = perceptron_output([-0.92, -0.01, 0.37], [1, -0.82, -0.15], sigmoid)
        self.assertAlmostEqual(output, 0.2754, 4)
