import argparse
import collections
import csv
from dataclasses import dataclass
import json
import os
import re

from typing import List


@dataclass
class Mention:
    entity_id: str
    start: int = None
    end: int = None


@dataclass
class Document:
    docid: str
    text: str
    mentions: List[Mention]

    @classmethod
    def from_lines(cls, lines):
        header = lines.popleft()
        match = re.search(f'(?<=\().*(?=\))', header)
        if match:
            docid = match.group(0)
            print(docid)
        else:
            raise RuntimeError('No DocID identified')
        text = ''
        mentions = []
        while lines:
            line = lines.popleft().strip()
            split = line.split('\t')
            if len(split) == 1:
                token = split[0]
                bio_tag = None
                yago2_id = None
            else:
                token, bio_tag, _, yago2_id, *_ = split
            if bio_tag == 'B':
                mentions.append(Mention(yago2_id, start=len(text)))
            text += token
            if bio_tag in ('B', 'I'):
                mentions[-1].end = len(text)
            if len(lines) > 1:
                text += ' '
        return Document(docid, text, mentions)


def parse(f):
    instances = []
    lines = collections.deque()
    lines.append(next(f))
    for line in f:
        if '-DOCSTART-' in line:
            yield Document.from_lines(lines)
        lines.append(line)
    # One last document
    yield Document.from_lines(lines)


def main(args):
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)

    g_train = open(os.path.join(args.prefix, 'train.jsonl'), 'w')
    g_dev = open(os.path.join(args.prefix, 'dev.jsonl'), 'w')
    g_test = open(os.path.join(args.prefix, 'test.jsonl'), 'w')
    g_entity = open(os.path.join(args.prefix, 'entity_vocab.txt'), 'w')

    splits = {
        '1 EU': ('train', g_train),
        '947testa CRICKET': ('dev', g_dev),
        '1163testb SOCCER': ('test', g_test),
    }

    entity_ids = collections.Counter()
    with open(args.input, 'r') as f:
        for document in parse(f):
            if document.docid in splits:
                mention_index = 0  # Reset at start of each section
                active_split, g = splits[document.docid]
            for mention in document.mentions:
                if mention.entity_id == '--NME--':
                    continue
                obj = {
                    'left_context': document.text[:mention.start],
                    'mention': document.text[mention.start:mention.end],
                    'right_context': document.text[mention.end:],
                    'entity_id': mention.entity_id,
                    'document_id': document.docid,
                    'mention_index': mention_index
                }
                if active_split == 'train':
                    entity_ids[mention.entity_id] += 1
                print(json.dumps(obj), file=g)
                mention_index += 1
    writer = csv.writer(g_entity)
    writer.writerow(('[PAD]', 0))
    for entity_id, count in sorted(entity_ids.items(), key=lambda x: x[1], reverse=True):
        writer.writerow((entity_id, count))

    g_train.close()
    g_dev.close()
    g_test.close()
    g_entity.close()

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()

    main(args)

