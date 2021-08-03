"""
String matching baseline
"""
import argparse
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def main(args):
    with open(args.input, 'r') as f:
        instances = [json.loads(line) for line in f]
    entity_ids = []
    clusters = []
    entity_vocab = {}
    unique_mentions = {}
    for instance in instances:
        entity_id = instance['entity_id']
        if entity_id not in entity_vocab:
            entity_vocab[entity_id] = len(entity_vocab)
        entity_ids.append(entity_vocab[entity_id])
        mention = instance['mention']
        if mention not in unique_mentions:
            unique_mentions[mention] = len(unique_mentions)
        clusters.append(unique_mentions[mention])

    with open(args.output, 'w') as g:
        for t, p in zip(entity_ids, clusters):
            g.write('%i, %i\n' % (t, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

