import io
import unittest

import torch
import transformers

from meercat import medmentions
from meercat import embed


# TODO: Update to using new data format (e.g., with left / right context)
# class MentionTokenizerTest(unittest.TestCase):
    # def test_call(self):
        # tokenizer = transformers.BertTokenizer('tests/fixtures/vocab.txt')
        # mention_tokenizer = embed.MentionTokenizer(tokenizer)

        # text = 'I am a banana!'
        # mention = medmentions.Mention(
            # start=7,
            # end=13,
            # text='banana',
            # semantic_types=['T0', 'T1'],
            # entity_id='C00'
        # )
        # inputs, mention_mask = mention_tokenizer(mention, text)
        # expected_input_ids = torch.tensor([[0, 2, 3, 4, 5, 6, 6, 7, 1]])
        # self.assertTrue(torch.equal(inputs.input_ids, expected_input_ids))
        # expected_mention_mask = torch.tensor(
            # [[False, False, False, False, True, True, True, False, False]]
        # )
        # self.assertTrue(torch.equal(mention_mask, expected_mention_mask))

