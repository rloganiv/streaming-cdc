import argparse
import collections
import json
import logging

import numpy as np

from meercat.eval_clusters import _create_muc_lookup


logger = logging.getLogger(__name__)


def clean(s):
    """
    Adds escape characters for a string.
    """
    s = s.replace(r'{', r'\{')
    s = s.replace(r'}', r'\}')
    s = s.replace(r'_', r'\_')
    s = s.replace(r'%', r'\%')
    return s

def load_clusters(f):
    true_clusters = collections.defaultdict(set)
    pred_clusters = collections.defaultdict(set)
    for i, line in enumerate(f):
        t, p = [x.strip() for x in line.split(',')]
        true_clusters[t].add(i)
        pred_clusters[p].add(i)
    return true_clusters, pred_clusters


def pretty(mention):
    clean_mention = {k: clean(s) for k, s in mention.items() if isinstance(s, str)}
    left = '\ldots ' + ' '.join(clean_mention['left_context'].split(' ')[-10:])
    center = f'\emph{{{clean_mention["mention"]}}}'
    right = ' '.join(clean_mention['right_context'].split(' ')[:10]) + ' \ldots'
    return ' '.join((left, center, right))

def main(args):
    logger.info('Loading data')
    mid_to_mention = {}
    with open(args.dataset, 'r') as f:
        for mid, line in enumerate(f):
            data = json.loads(line)
            mid_to_mention[mid] = data

    with open(args.clusters, 'r') as f:
        true_clusters, pred_clusters = load_clusters(f)

    # MUC Lookups map mids to cluster ids
    true_lookup = _create_muc_lookup(true_clusters)
    pred_lookup = _create_muc_lookup(pred_clusters)

    # The following data structure maps cluster ids to sequential integers.
    true_idx_lookup = {}
    for v in true_lookup.values():
        if v not in true_idx_lookup:
            true_idx_lookup[v] = len(true_idx_lookup)
    reverse_true_idx_lookup = {i: x for x, i  in true_idx_lookup.items()}

    # Number of distinct entities
    num_distinct_entities = len(true_clusters)
    logger.info(f'Num Entities: {num_distinct_entities}')

    conflated_entities = 0
    most_conflated = 0
    most_conflated_cluster = None
    for cluster in pred_clusters.values():
        
        # Map each mention in predicted cluster to its true cluster id.
        true_set = set(true_lookup[mid] for mid in cluster)
        
        # The number of conflated entities is the number of additional clusters
        # that mentions would need to be split into.
        conflated_entities_ = len(true_set) - 1
        conflated_entities += conflated_entities_

        if conflated_entities_ > most_conflated:
            most_conflated = conflated_entities_
            most_conflated_cluster = cluster

    logger.info(f'Conflated Entities: {conflated_entities}')

    split_entities = 0
    most_split = 0
    most_split_cluster = None
    for cluster in true_clusters.values():
    
        # Map each mention in the true cluster to its predicted cluster id.
        pred_set = set(pred_lookup[mid] for mid in cluster)

        # TODO: Double check this actually computes split entities.
        split_entities_ = len(pred_set) - 1
        split_entities += split_entities_
        
        if split_entities_ > most_split:
            most_split = split_entities_
            most_split_cluster = cluster
        
    logger.info(f'Split Entities: {split_entities}')

    if split_entities > 0:
        most_split_cluster_mentions = [mid_to_mention[mid] for mid in most_split_cluster]
        print('Most Split')
        for mention in most_split_cluster_mentions[:5]:
            print(pretty(mention))

    print()
    if conflated_entities > 0:
        seen =  set()
        most_conflated_cluster_mentions = [mid_to_mention[mid] for mid in most_conflated_cluster]
        print('Most Conflated')
        i = 0
        for mention in most_conflated_cluster_mentions:
            entity_id = mention['entity_id'].replace('_', ' ')
            if entity_id in seen:
                continue
            line = ' & '.join((entity_id, pretty(mention))) + r' \\'
            print(line)
            seen.add(entity_id)
            if i == 4:
                break
            i+=1
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-c', '--clusters', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

