import unittest
from .active_reinforcement import Edge, utility, next_utility


class PerceptronTestCases(unittest.TestCase):
    def test_utility_a1(self):
        """2. ZH 20.feladat"""
        edges = [
            Edge(0.5, -1),      # nyíl s1-ből s1-be
            Edge(0.1, -5),      # nyíl s1-ből s2-be
            Edge(0.1, -2.9),    # nyíl s1-ből s3-ba
            Edge(0.3, 8),       # nyíl s1-ből s3-ba
        ]
        expected_utility = utility(edges)
        self.assertAlmostEqual(expected_utility, 1.11, 2)

    def test_utility_a2(self):
        """2. ZH 21.feladat"""
        edges = [
            Edge(0.3, -5),  # nyíl s1-ből s1-be
            Edge(0.7, -1)   # nyíl s1-ből s2-be
        ]
        expected_utility = utility(edges)
        self.assertAlmostEqual(expected_utility, -2.2, 1)

    def test_next_utility(self):
        """2. ZH 23.feladat"""
        edges_a1 = [
            Edge(0.5, -1),      # nyíl s1-ből s1-be
            Edge(0.1, -5),      # nyíl s1-ből s2-be
            Edge(0.1, -2.9),    # nyíl s1-ből s3-ba
            Edge(0.3, 8),       # nyíl s1-ből s3-ba
        ]

        edges_a2 = [
            Edge(0.3, -5),  # nyíl s1-ből s1-be
            Edge(0.7, -1)   # nyíl s1-ből s2-be
        ]

        reward = -1.8
        leszam = 0.5

        nu = next_utility(
                reward,
                leszam,
                max(utility(edges_a1), utility(edges_a2))
        )

        self.assertAlmostEqual(nu, -1.245, 3)
