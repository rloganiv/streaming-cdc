"""
Preprocesses the MedMentions corpus.

Because the corpus does not come pre-split, and pubtator is a kind of annoying
format, this script serializes the data as a JSONL file where each line is a
mention.
"""
import argparse
import collections
import csv
import json
import logging

from meercat import medmentions


logger = logging.getLogger(__name__)


def main(args):
    logger.info('Reading split pmids')
    with open(args.train, 'r') as f:
        train_pmids = set(line.strip() for line in f)
    with open(args.dev, 'r') as f:
        dev_pmids = set(line.strip() for line in f)
    with open(args.test, 'r') as f:
        test_pmids = set(line.strip() for line in f)

    logger.info('Creating jsonl files')
    entity_ids_train = collections.Counter()
    entity_ids_dev = collections.Counter()
    entity_ids_test = collections.Counter()
    with open(args.pubtator_file, 'r') as f, \
         open(args.prefix + 'train.jsonl', 'w') as g_train, \
         open(args.prefix + 'dev.jsonl', 'w') as g_dev, \
         open(args.prefix + 'test.jsol', 'w') as g_test:
        for document in medmentions.parse_pubtator(f):
            if document.pmid in train_pmids:
                g = g_train
                entity_ids = entity_ids_train
            elif document.pmid in dev_pmids:
                g = g_dev
                entity_ids = entity_ids_dev
            elif document.pmid in test_pmids:
                g = g_test
                entity_ids = entity_ids_test
            else:
                raise RuntimeError(f'Unexpected pmid: {document.pmid}.')

            for mention in document.mentions:
                text = ' '.join((document.title, document.abstract))
                json_obj = {
                    'left_context': text[:mention.start],
                    'mention': text[mention.start:mention.end],
                    'right_context': text[mention.end:],
                    'entity_id': mention.entity_id,
                    'type': mention.semantic_types,
                }
                g.write(json.dumps(json_obj) + '\n')
                entity_ids[mention.entity_id] += 1

    logger.info('Serializing entity vocab')
    counts = sorted(entity_ids_train.items(), key=lambda x: x[1], reverse=True)
    with open(args.prefix + 'entity_vocab.txt', 'w') as g:
        writer = csv.writer(g)
        writer.writerow(('[PAD]', 0))
        for entity_id, count in counts:
            writer.writerow((entity_id, count))

    # logger.info(f'Unique Entities: {len(entity_ids_all)}')
    # logger.info(f'Unseen Dev Entities: {len(entity_ids_dev - entity_ids_train)}')
    # logger.info(f'Unseen Test Entities: {len(entity_ids_test - entity_ids_train)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--pubtator-file', type=str,
                        help='Input pubtator file')
    parser.add_argument('-o', '--prefix', type=str,
                        help='Output prefix')
    parser.add_argument('--train', type=str, help='Training PMIDs')
    parser.add_argument('--dev', type=str, help='Dev PMIDs')
    parser.add_argument('--test', type=str, help='Test PMIDs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

