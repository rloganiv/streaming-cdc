import argparse
import json
import logging

import torch
from blink.biencoder.biencoder import BiEncoderRanker
from blink.biencoder.data_process import (
    ENT_START_TAG, ENT_END_TAG
)


logger = logging.getLogger(__name__)


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
):
    mention_tokens = [ENT_START_TAG] + tokenizer.tokenize(sample['mention']) + [ENT_END_TAG]
    left_context = tokenizer.tokenize(sample['left_context'])
    right_context = tokenizer.tokenize(sample['right_context'])

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(left_context)
    right_add = len(right_context)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = left_context[-left_quota:] + mention_tokens + right_context[:right_quota]
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def load_dataset(fname, tokenizer, max_seq_length):
    processed_samples = []
    entity_ids = []
    with open(fname, 'r') as f:
        for line in f:
            sample = json.loads(line)
            context = get_context_representation(
                sample,
                tokenizer,
                max_seq_length,
            )
            processed_samples.append(context)
            entity_ids.append(sample['entity_id'])
    tokens = torch.tensor([x['ids'] for x in processed_samples], dtype=torch.long)

    return torch.utils.data.TensorDataset(tokens), entity_ids


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading model.')
    with open(args.config, 'r') as f:
        params = json.load(f)
    model = BiEncoderRanker(params)
    model.load_model(args.ckpt)
    model.to(device)

    logger.info('Loading data.')
    dataset, entity_ids = load_dataset(args.input, model.tokenizer, args.max_seq_length)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

    with open(args.output, 'w') as g:
        for i, (tokens, entity_id) in enumerate(zip(dataloader, entity_ids)):
            tokens = tokens[0].to(device)
            with torch.no_grad():
                encodings = model.encode_context(tokens)
            serialized = '\t'.join(str(x) for x in encodings[0].tolist())
            line = f'{i}\t{entity_id}\t{serialized}\n'
            g.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

