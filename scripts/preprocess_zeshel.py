"""
Preprocesses the zeshel corpus.
"""
import argparse
import collections
import csv
import json
import logging
import pathlib


logger = logging.getLogger(__name__)


def main(args):
    logger.info('Loading documents')
    documents = {}
    titles = {}
    document_path = args.input / 'documents'
    for document_file in document_path.iterdir():
        with open(document_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                document_id = data['document_id']
                title = data['title']
                text = data['text']
                documents[document_id] = (title, text, document_file.stem)

    if not args.prefix.exists():
        logger.info('%s does not exist. Creating.', args.prefix)
        args.prefix.mkdir(parents=True)
    
    logger.info('Processing mentions')

    mentions_path = args.input / 'mentions'
    for mention_file in mentions_path.iterdir():
        out_file = args.prefix / mention_file.name
        label_document_ids = collections.Counter()
        with open(mention_file, 'r') as f, \
             open(out_file, 'w') as g:
            instances = []
            for line in f:
                data = json.loads(line)
                document_id = data['context_document_id']
                label_document_ids[data['label_document_id']] += 1
                tokens = documents[document_id][1].split()
                category = documents[document_id][2]
                start = data['start_index']
                end = data['end_index'] + 1
                entity_id = documents[data['label_document_id']][0]
                out = {
                    'left_context': ' '.join(tokens[:start]),
                    'mention': ' '.join(tokens[start:end]),
                    'right_context': ' '.join(tokens[end:]),
                    'entity_id': entity_id,
                    'category': category,
                    'document_id': document_id,
                }
                instances.append(out)
            for label_document_id in label_document_ids:
                title, text, category = documents[label_document_id]
                text = text.replace(title, '', 1)
                out = {
                    'left_context': '',
                    'mention': title,
                    'right_context': text.strip(),
                    'entity_id': title,
                    'category': category,
                    'document_id': label_document_id,
                }
                instances.append(out)
            instances.sort(key=lambda x: x['category'])
            for mention_index, instance in enumerate(instances):
                instance['mention_index'] = mention_index
                mention_index += 1
                print(json.dumps(instance), file=g)
        if mention_file.name == 'train.json':
            with open(args.prefix / 'entity_vocab.txt', 'w') as g:
                writer = csv.writer(g)
                writer.writerow(('[PAD]', 0))
                for entity_id, count in sorted(label_document_ids.items(), key=lambda x: x[1], reverse=True):
                    writer.writerow((entity_id, count+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=pathlib.Path, required=True)
    parser.add_argument('-o', '--prefix', type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

