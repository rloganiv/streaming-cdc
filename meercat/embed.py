"""Transformer embeddings of MedMentions"""
import argparse
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

    def __call__(self, mention, text):
        # Break text into relevant chunks
        prefix = text[:mention.start]
        center = text[mention.start:mention.end]
        suffix = text[mention.end:]

        # Tokenize text
        # TODO(rloganiv): The suffix tokenizer will not create correct
        # wordpieces if the suffix is part of the same word as the mention.
        # I don't know if there is a generic, principled way to deal with this
        # but it is worth being aware of.
        prefix_tokens = self.tokenizer.tokenize(prefix)
        center_tokens = self.tokenizer.tokenize(center)
        suffix_tokens = self.tokenizer.tokenize(suffix)

        # If max length exceeded then create as big a window as possible around
        # the mention.
        length = len(prefix_tokens) + len(center_tokens) + len(suffix_tokens)
        max_length = self.tokenizer.max_len_single_sentence
        excess = length - max_length
        if excess > 0:
            # Start with a symmetric window
            left_size = right_size = excess // 2
            
            # Distribute any remaining space
            if left_size > len(prefix_tokens):
                right_size += left_size - len(prefix_tokens)
            if right_size > len(suffix_tokens):
                left_size += right_size - len(suffix_tokens)

            # Truncate
            prefix_tokens = prefix_tokens[-left_size:]
            suffix_tokens = suffix_tokens[:right_size]

        # Add special tokens.
        if self.tokenizer.bos_token is not None:
            prefix_tokens.insert(0, self.tokenizer.bos_token)
        elif self.tokenizer.cls_token is not None:
            prefix_tokens.insert(0, self.tokenizer.cls_token)

        if self.tokenizer.eos_token is not None:
            suffix_tokens.append(self.tokenizer.eos_token)
        elif self.tokenizer.sep_token is not None:
            suffix_tokens.append(self.tokenizer.sep_token)

        # Encode tokens and get mention mask
        tokens = prefix_tokens + center_tokens + suffix_tokens
        inputs = self.tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=False,
            return_tensors='pt',
        )
        mention_mask = torch.zeros(len(tokens), dtype=torch.bool)
        mention_start = len(prefix_tokens)
        mention_end = mention_start + len(center_tokens)
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

    embeddings = torch.empty(args.num_mentions, model.config.hidden_size)
    true_labels = []

    j = 0
    logger.info('Embedding')
    with open(args.dataset_path, 'r') as f:
        for document in medmentions.parse_pubtator(f):
            text = ' '.join((document.title, document.abstract))
            pmid = document.pmid
            for i, mention in enumerate(document.mentions):
                # Compute embedding by taking average of mention
                # representations.
                inputs, mention_mask = mention_tokenizer(mention, text)
                inputs.to(device)
                mention_mask.to(device)
                with torch.no_grad():
                    hidden, *_ = model(**inputs)
                    embedding = hidden[mention_mask].mean(0)
                embeddings[j] = embedding.to(cpu_device)

                true_clusters.append(entity_tokenizer(mention.entity_id))

                # Serialize.
                # mention_id = f'{pmid}_{i}'
                # embedding_string = '\t'.join(str(x) for x in embedding.tolist())
                # serialized = '\t'.join((mention_id, 'NA', embedding_string))
                # print(serialized)
    logger.info('Clustering')
    pred_clusters = cluster(embeddings, args.threshold)
    for t, p in zip(true_clusters, pred_clusters):
        print('%i, %i' % (t.item(), p.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--entity_vocab', type=str, required=True)
    parser.add_argument('--num_mentions', type=int, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.24)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

