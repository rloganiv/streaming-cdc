from unittest import TestCase

from meercat import eval_clusters


class TestMUC(TestCase):
    # NOTE: We use inconsistent names for clusters since this is almost certain to be the case
    # for most of our clustering outputs.

    def test_case_1(self):
        # Vilain Table 1 Row 1
        true_clusters = {0: {'A', 'B', 'C', 'D'}}
        pred_clusters = {1: {'A', 'B'}, 2: {'C', 'D'}}
        p, r, f = eval_clusters.muc(true_clusters, pred_clusters)
        assert p == 2/2
        assert r == 2/3

    def test_case_2(self):
        # Villain Table 1 Row 2
        true_clusters = {0: {'A', 'B'}, 1: {'C', 'D'}}
        pred_clusters = {2: {'A', 'B', 'C', 'D'}}
        p, r, f = eval_clusters.muc(true_clusters, pred_clusters)
        assert p == 2/3
        assert r == 2/2

    def test_case_3(self):
        # Villain Table 1 Row 3
        true_clusters = {0: {'A', 'B', 'C', 'D'}}
        pred_clusters = {1: {'A', 'B', 'C', 'D'}}
        p, r, f = eval_clusters.muc(true_clusters, pred_clusters)
        assert p == 3/3
        assert r == 3/3

    def test_case_4(self):
        # Villain Table 1 Row 5 (Skipped 4 since equvalent to 1)
        true_clusters = {0: {'A', 'B', 'C'}}
        pred_clusters = {1: {'A', 'C'}, 2: {'B'}}
        p, r, f = eval_clusters.muc(true_clusters, pred_clusters)
        assert p == 1/1
        assert r == 1/2


class TestB3(TestCase):
    def test_case_1(self):
        # Luo Table 1.a
        true_clusters = {
            0: {'1', '2', '3', '4', '5'},
            1: {'6', '7'},
            2: {'8', '9', 'A', 'B', 'C'},
        }
        pred_clusters = {
            3: {'1', '2', '3', '4', '5'},
            4: {'6', '7', '8', '9', 'A', 'B', 'C'},
        }
        total = 12
        *_, f = eval_clusters.b3(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.865) < 1e-3

    def test_case_2(self):
        # Luo Table 1.b
        true_clusters = {
            0: {'1', '2', '3', '4', '5'},
            1: {'6', '7'},
            2: {'8', '9', 'A', 'B', 'C'},
        }
        pred_clusters = {
            0: {'1', '2', '3', '4', '5', '8', '9', 'A', 'B', 'C'},
            1: {'6', '7'},
        }
        total = 12
        *_, f = eval_clusters.b3(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.737) < 1e-3

    def test_case_3(self):
        # Luo Table 1.c
        true_clusters = {
            0: {'1', '2', '3', '4', '5'},
            1: {'6', '7'},
            2: {'8', '9', 'A', 'B', 'C'},
        }
        pred_clusters = {
            0: {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C'},
        }
        total = 12
        *_, f = eval_clusters.b3(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.545) < 1e-3


class TestCeafE(TestCase):
    def test_case_1(self):
        # Luo Table 1.a
        true_clusters = {
            0: {0, 1, 2, 3, 4},
            1: {5, 6},
            2: {7, 8, 9, 10, 11},
        }
        pred_clusters = {
            3: {0, 1, 2, 3, 4},
            4: {5, 6, 7, 8, 9, 10, 11},
        }
        total = 12
        *_, f = eval_clusters.ceaf_e(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.733) < 1e-3

    def test_case_2(self):
        # Luo Table 1.b
        true_clusters = {
            0: {0, 1, 2, 3, 4},
            1: {5, 6},
            2: {7, 8, 9, 10, 11},
        }
        pred_clusters = {
            3: {0, 1, 2, 3, 4, 7, 8, 9, 10, 11},
            4: {5, 6},
        }
        total = 12
        *_, f = eval_clusters.ceaf_e(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.667) < 1e-3

    def test_case_3(self):
        # Luo Table 1.c
        true_clusters = {
            0: {0, 1, 2, 3, 4},
            1: {5, 6},
            2: {7, 8, 9, 10, 11},
        }
        pred_clusters = {
            3: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        }
        total = 12
        *_, f = eval_clusters.ceaf_e(true_clusters, pred_clusters, total)
        # Result in table is only approximate so we only measure that it is close
        assert abs(f - 0.294) < 1e-3

