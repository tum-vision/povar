import unittest
from collections import defaultdict
from random import randrange
from fp_tree import FPTree, load_bal


class TestFPTree(unittest.TestCase):

    def test_support(self):
        support = defaultdict(lambda: 0)
        lms_obs = defaultdict(list)

        for lm_idx in range(100):
            obs = []
            for _ in range(randrange(1, 100)):
                idx = randrange(0, 100)
                obs.append(idx)
                support[idx] += 1
            lms_obs[lm_idx] = obs

        tree = FPTree(lms_obs, verbose=False)
        self.assertEqual(support, tree.support)

    def test_insert_order(self):
        lms_obs = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4], 5: [5]}
        tree = FPTree(lms_obs, verbose=False)

        lm_obs = [0, 2, 4, 1, 3, 5]
        lm_obs.sort(key=lambda x: (tree.support[x], x))

        # ascending wrt to cam support, break tie by cam idx
        self.assertEqual(lm_obs, [4, 5, 3, 2, 1, 0])

    def test_grow_tree(self):
        # datapoints from the paper
        lms_obs = {
            1: [1, 2],
            2: [1, 2, 3],
            3: [1, 2, 3],
            4: [1, 2, 3],
            5: [1, 2, 3],
            6: [2, 3],
            7: [3, 4, 5],
            8: [5, 6],
            9: [5, 6],
            10: [5, 6]
        }

        tree = FPTree(lms_obs, verbose=False)
        edges, nodes = tree.get_edges()

        labeled_edges = []
        for fr, to in edges:
            labeled_edges.append((nodes[fr].cam_idx, nodes[to].cam_idx))

        self.assertEqual(labeled_edges.count((-1, 2)), 1)
        self.assertEqual(labeled_edges.count((2, 1)), 2)
        self.assertEqual(labeled_edges.count((-1, 3)), 1)
        self.assertEqual(labeled_edges.count((3, 2)), 1)
        self.assertEqual(labeled_edges.count((3, 5)), 1)
        self.assertEqual(labeled_edges.count((5, 4)), 1)
        self.assertEqual(labeled_edges.count((-1, 5)), 1)
        self.assertEqual(labeled_edges.count((5, 6)), 1)
        self.assertEqual(len(labeled_edges), 9)

        non_empty_nodes_path = [([2, 1], [1]), ([3, 2], [6]), ([3, 2, 1], [2, 3, 4, 5]), ([3, 5, 4], [7]),
                                ([5, 6], [8, 9, 10])]
        empty_nodes_path = [[2], [3], [3, 5], [5]]

        for path, lms in non_empty_nodes_path:
            it = tree.root
            while path:
                it = it.childs[path.pop(0)]
            self.assertEqual(set(it.lm_indices), set(lms))

        for path in empty_nodes_path:
            it = tree.root
            while path:
                it = it.childs[path.pop(0)]
            self.assertEqual(len(it.lm_indices), 0)

    def test_factors_paper(self):
        # datapoints from the paper
        lms_obs = {
            1: [1, 2],
            2: [1, 2, 3],
            3: [1, 2, 3],
            4: [1, 2, 3],
            5: [1, 2, 3],
            6: [2, 3],
            7: [3, 4, 5],
            8: [5, 6],
            9: [5, 6],
            10: [5, 6]
        }

        tree = FPTree(lms_obs, verbose=False)
        factors, non_factors, _ = tree.get_factors()

        factors = {tuple(sorted(k)): set(v) for k, v in factors}

        ref_factors = {(1, 2, 3): {1, 2, 3, 4, 5, 6}, (5, 6): {8, 9, 10}}
        ref_non_factors = [7]

        self.assertEqual(factors, ref_factors)
        self.assertEqual(non_factors, ref_non_factors)

    def test_factors_bal(self):
        bal_path = "../../../data/rootba/bal/final/problem-93-61203-pre.txt"
        lms_obs_dict, nums = load_bal(bal_path)
        num_lms = nums[1]

        tree = FPTree(lms_obs_dict, verbose=False)
        factors, non_factors, _ = tree.get_factors()

        # check total number of returned landmarks
        total_lms = len(non_factors)
        for _, lm_indices in factors:
            total_lms += len(lm_indices)
        self.assertEqual(total_lms, num_lms)

        # check subset
        for cam_indices, lm_indices in factors:
            for lm_idx in lm_indices:
                lm_obs = lms_obs_dict[lm_idx]
                self.assertTrue(set(cam_indices).issuperset(lm_obs))

        # check non factor lms isn't a subset of any factor
        for lm_idx in non_factors:
            lm_obs = lms_obs_dict[lm_idx]
            for cam_indices, _ in factors:
                self.assertFalse(set(cam_indices).issuperset(lm_obs))

        # check path and grouping condition
        for cam_indices, _ in factors:
            cam_indices = sorted(cam_indices, key=lambda x: (tree.support[x], x))

            it = tree.root
            num_cams = 0
            num_lms = 0
            while cam_indices:
                cam_idx = cam_indices.pop()
                self.assertTrue(cam_idx in it.childs)
                num_lms += len(it.lm_indices)
                it = it.childs[cam_idx]
                num_cams += 1
            num_lms += len(it.lm_indices)

            self.assertTrue(it.level == num_cams)
            self.assertGreater(num_lms, num_cams)


if __name__ == "__main__":
    unittest.main()
