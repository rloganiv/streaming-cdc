import io
import unittest

import dendrogram_purity

tree_tsv = '''0	None	None
a	0	None
1	0	None
b	1	None
2	1	None
c	2	None
d	2	None
'''

metadata = {
    'a': {'color': 'white'},
    'b': {'color': 'white'},
    'c': {'color': 'black'},
    'd': {'color': 'black'},
}


def load_test_tree():
    f = io.StringIO(tree_tsv)
    root = dendrogram_purity.load_dendrogram(f)
    return root
    

class DendrogramTest(unittest.TestCase):
    def setUp(self):
        self.root = load_test_tree()

    def test_tree_structure(self):
        self.assertEqual(self.root.uid, '0')
        for child in self.root.children:
            self.assertEqual(child.parent, self.root)
            self.assertIn(child.uid, {'a', '1'})

    def test_traverse(self):
        expected = ['0', '1', '2', 'd', 'c', 'b', 'a']
        observed = list(x.uid for x in dendrogram_purity.traverse(self.root))
        self.assertListEqual(expected, observed)


class DendrogramPurityTest(unittest.TestCase):
    def test_num_pairs(self):
        expected = 2
        observed = dendrogram_purity.num_pairs(metadata, 'color')
        self.assertEqual(expected, observed)

    def test_accumulate_purity(self):
        root = load_test_tree()
        p_star = dendrogram_purity.num_pairs(metadata, 'color')
        summand = dendrogram_purity.accumulate_purity(root, metadata, 'color')
        expected = 0.75
        observed = summand / p_star
        self.assertEqual(expected, observed)

