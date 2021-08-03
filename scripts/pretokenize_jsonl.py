import argparse

import transformers

import meercat.utils as utils


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.model_max_length = 128
    utils.add_mention_seps(tokenizer)
    entity_tokenizer = utils.EntityTokenizer.from_pretrained(args.entity_vocab)
    data = utils.ELDataset.from_jsonl(args.input, tokenizer, entity_tokenizer)
    data.save(args.output)


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--entity_vocab', type=str)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()

    main(args)
