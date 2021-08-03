import argparse
import collections
import json


def main(args):
    entities = collections.Counter()
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            entities[data['entity_id']] += 1
    print('Unique entities: %i', len(entities))
    singletons = filter(lambda x: x[1] == 1, entities.items())
    print('Num singletons: %i', len(list(singletons)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    main(args)

