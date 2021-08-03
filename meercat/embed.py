"""Transformer embeddings of MedMentions"""
import argparse
import json
import logging

import torch
import transformers

from meercat import medmentions
from meercat.nn_thresh import cluster
from meercat.utils import EntityTokenizer


logger = logging.getLogger(__name__)


class MentionTokenizer:
    """Simple wrapper around HuggingFace tokenizer to assist with tracking
    mention boundaries"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, left_context, mention, right_context):
        # Break text into relevant chunks
        # prefix = text[:mention.start]
        # center = text[mention.start:mention.end]
        # suffix = text[mention.end:]

        # Tokenize text
        # TODO(rloganiv): The suffix tokenizer will not create correct
        # wordpieces if the suffix is part of the same word as the mention.
        # I don't know if there is a generic, principled way to deal with this
        # but it is worth being aware of.
        left_context = self.tokenizer.tokenize(left_context)
        mention = self.tokenizer.tokenize(mention)
        right_context = self.tokenizer.tokenize(right_context)

        # If max length exceeded then create as big a window as possible around
        # the mention.
        length = len(left_context) + len(mention) + len(right_context) + 2
        max_length = 512 # self.tokenizer.max_len_single_sentence
        excess = length - max_length
        if excess > 0:
            # Start with a symmetric window
            left_size = right_size = (max_length - len(mention)) // 2 - 1
            
            # Distribute any remaining space
            if left_size > len(left_context):
                right_size += left_size - len(left_context)
                left_size = len(left_context)
            elif right_size > len(right_context):
                left_size += right_size - len(right_context)
                right_size = len(right_context)

            # Truncate
            left_context = left_context[-left_size:]
            right_context = right_context[:right_size]

        # Add special tokens.
        if self.tokenizer.bos_token is not None:
            left_context.insert(0, self.tokenizer.bos_token)
        elif self.tokenizer.cls_token is not None:
            left_context.insert(0, self.tokenizer.cls_token)

        if self.tokenizer.eos_token is not None:
            right_context.append(self.tokenizer.eos_token)
        elif self.tokenizer.sep_token is not None:
            right_context.append(self.tokenizer.sep_token)

        # Encode tokens and get mention mask
        tokens = left_context + mention + right_context
        inputs = self.tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=False,
            return_tensors='pt',
        )
        mention_mask = torch.zeros(len(tokens), dtype=torch.bool)
        mention_start = len(left_context)
        mention_end = mention_start + len(mention)
        mention_mask[mention_start:mention_end] = 1
        mention_mask.unsqueeze_(0)

        return inputs, mention_mask


def main(args):
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cpu_device = torch.device('cpu')

    model = transformers.AutoModel.from_pretrained(args.model_name)
    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    mention_tokenizer = MentionTokenizer(tokenizer)
    entity_tokenizer = EntityTokenizer.from_pretrained(args.entity_vocab)

    j = 0
    logger.info('Embedding')
    with open(args.input, 'r') as f, \
         open(args.output, 'w') as g:
        for i, line in enumerate(f):
            data = json.loads(line)
            # Compute embedding by taking average of mention
            # representations.
            inputs, mention_mask = mention_tokenizer(
                left_context=data['left_context'],
                mention=data['mention'],
                right_context=data['right_context'],
            )
            inputs.to(device)
            mention_mask.to(device)
            with torch.no_grad():
                hidden, *_ = model(**inputs)
                embedding = hidden[mention_mask].mean(0)
            # embeddings[j] = embedding.to(cpu_device)

            # Serialize.
            # mention_id = f'{pmid}_{i}'
            entity_id = entity_tokenizer(data['entity_id'])
            embedding_string = '\t'.join(str(x) for x in embedding.tolist())
            serialized = f'{i}\t{entity_id}\t{embedding_string}\n'
            g.write(serialized)
    # logger.info('Clustering')
    # pred_clusters = cluster(embeddings, args.threshold)
    # for t, p in zip(true_clusters, pred_clusters):
        # print('%i, %i' % (t.item(), p.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--entity_vocab', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

