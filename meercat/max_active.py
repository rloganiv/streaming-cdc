import argparse
import json


def main(args):
    times = dict()
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            entity_id = data['entity_id']
            if entity_id not in times:
                times[entity_id] = [i, i]
            else:
                times[entity_id][1] = i
    time_list = []
    for entity_id, (start, end) in times.items():
        time_list.append((entity_id, start, 'start'))
        time_list.append((entity_id, end, 'end'))
    time_list.sort(key=lambda x: x[1])
    active_entities = set()
    max_active_entities = 0
    for entity_id, _, event_type in time_list:
        if event_type == 'start':
            active_entities.add(entity_id)
        elif event_type == 'end':
            active_entities.remove(entity_id)
        max_active_entities = max(len(active_entities), max_active_entities)
    print(f'Max active entities: {max_active_entities}')
    print(f'Total entities: {len(times)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    main(args)

