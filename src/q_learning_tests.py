import unittest
from .q_learning import q_update


class MyTestCase(unittest.TestCase):
    def test_q_update_step_1(self):
        """2. ZH 24. feladat"""
        table = [
        #   a1      a2
            [9.5,   0],     # s1
            [1.5,   3.2],   # s2
            [0,     5.6]    # s3
        ]
        reward = 5
        learning_rate = 0.9
        y = 0.9
        update = q_update(table, 0, 2, 0, reward, learning_rate, y)
        self.assertAlmostEqual(update, 9.986, 3)

    def test_q_update_step_2(self):
        """2. ZH 25. feladat"""
        table = [
        #   a1      a2
            [9.986,   0],       # s1, frissített értékkel
            [1.5,   3.2],       # s2
            [0,     5.6]        # s3
        ]
        reward = -10
        learning_rate = 0.9
        y = 0.9
        update = q_update(table, 2, 1, 0, reward, learning_rate, y)
        self.assertAlmostEqual(update, -6.408, 3)

    def test_q_update_step_3(self):
        """2. ZH 26. feladat"""
        table = [
            #   a1      a2
            [9.986, 0],  # s1, frissített értékkel
            [1.5, 3.2],  # s2
            [-6.408, 5.6]  # s3, frissített értékkel
        ]
        reward = 5
        learning_rate = 0.9
        y = 0.9
        update = q_update(table, 1, 2, 1, reward, learning_rate, y)
        self.assertAlmostEqual(update, 9.356, 3)

    def test_q_update_step_4(self):
        """2. ZH 27. feladat"""
        table = [
            #   a1      a2
            [9.986, 0],  # s1, frissített értékkel
            [1.5, 9.356],  # s2, frissített értékkel
            [-6.408, 5.6]  # s3, frissített értékkel
        ]
        reward = 10
        learning_rate = 0.9
        y = 0.9
        update = q_update(table, 2, 2, 1, reward, learning_rate, y)
        self.assertAlmostEqual(update, 14.096, 3)
