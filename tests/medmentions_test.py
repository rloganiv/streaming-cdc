import io
import unittest

from meercat import medmentions


example = '''1234|t|Title 1
1234|a|Abstract 1
1234	0	1	T	T1,T2	C00
1234	2	3	t	T3	C01

2345|t|Title 2
2345|a|Abstract 2
2345	0	2	Ti	T1,T3	C02

'''


class ParseTest(unittest.TestCase):

    def test_parse(self):
        f = io.StringIO(example)
        documents = list(medmentions.parse_pubtator(f))
        self.assertEqual(len(documents), 2)

        document = documents[0]
        self.assertEqual(document.title, 'Title 1')
        self.assertEqual(document.abstract, 'Abstract 1')
        self.assertEqual(len(document.mentions), 2)

        mention = document.mentions[0]
        self.assertEqual(mention.start, 0)
        self.assertEqual(mention.end, 1)
        self.assertEqual(mention.text, 'T')
        self.assertListEqual(mention.semantic_types, ['T1', 'T2'])

