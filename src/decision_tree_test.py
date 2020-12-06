import unittest
from .decision_tree import I, NodeData, I_all_subtrees


class MyTest(unittest.TestCase):
    def test_I(self):
        """Tesztesetek a 2. zh-ból"""
        self.assertAlmostEqual(I(NodeData(1070, 990)), 0.9989, 4)
        self.assertAlmostEqual(I(NodeData(490, 174)), 0.8298, 4)
        self.assertAlmostEqual(I(NodeData(230, 0)), 0, 4)
        self.assertAlmostEqual(I(NodeData(350, 816)), 0.8815, 4)

    def test_I_all_subtrees(self):
        """Tesztesetek a 2. zh-ból"""
        self.assertAlmostEqual(I_all_subtrees([NodeData(230, 0), NodeData(490, 174), NodeData(350, 816)]), 0.7664, 4)
