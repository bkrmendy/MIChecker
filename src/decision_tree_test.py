import unittest
from .decision_tree import I,\
    NodeData,\
    I_all_subtrees,\
    BinaryClassificationDescription,\
    calculate_binary_classification_metrics, \
    sample_complexity


class MyTest(unittest.TestCase):
    def test_I(self):
        """Tesztesetek a 2. zh-ból: 1., 2., 3., 4. feladat"""
        self.assertAlmostEqual(I(NodeData(1070, 990)), 0.9989, 4)
        self.assertAlmostEqual(I(NodeData(490, 174)), 0.8298, 4)
        self.assertAlmostEqual(I(NodeData(230, 0)), 0, 4)
        self.assertAlmostEqual(I(NodeData(350, 816)), 0.8815, 4)

    def test_I_all_subtrees(self):
        """Tesztesetek a 2. zh-ból: 5. feladat"""
        self.assertAlmostEqual(I_all_subtrees([NodeData(230, 0), NodeData(490, 174), NodeData(350, 816)]), 0.7664, 4)

    def test_binary_classification(self):
        """Tesztesetek a 2. zh-ból: 11. feladat"""
        metrics = calculate_binary_classification_metrics(BinaryClassificationDescription(600, 500, 0.78, 0.92))
        self.assertAlmostEqual(metrics.true_positive_rate, 0.9213, 4)

    def test_sample_complexity(self):
        """Tesztesetek a 2. zh-ból: 12. feladat"""
        self.assertAlmostEqual(sample_complexity(10 ** 5, 0.02, 0.03), 750.9742, 4)
