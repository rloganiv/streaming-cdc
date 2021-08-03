"""
Oracle within doc clusters.
"""
import argparse
import json
import logging

import numpy as np


logger = logging.getLogger(__name__)


def main(args):
    entity_vocab = {}
    wd_entity_vocab = {} 
    with open(args.input, 'r') as f, \
         open(args.output, 'w') as g:
        for line in f:
            try:
                data = json.loads(line) 
            except:
                print(line)
                raise Exception()

            entity_id = data['entity_id']
            if entity_id not in entity_vocab:
                entity_vocab[entity_id] = len(entity_vocab)
            true = entity_vocab[entity_id]

            wd_entity_id = f'{data["entity_id"]}::{data["document_id"]}'
            if wd_entity_id not in entity_vocab:
                wd_entity_vocab[wd_entity_id] = len(wd_entity_vocab)
            pred = wd_entity_vocab[wd_entity_id]

            print(f'{true}, {pred}', file=g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

