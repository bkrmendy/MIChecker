import unittest
from .decision_tree import *


class MyTest(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(I(1070, 990), 0.9989, 4)