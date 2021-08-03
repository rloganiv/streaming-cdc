"""
Feature-based greedy nearest neighbor clustering baseline described in
Shrimpton et al. 2015
"""
import argparse
import json
import logging
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


logger = logging.getLogger(__name__)


def score(
    instances,
    vectorizers,
    weights=(0.8, 0.2)
):
    mentions = [x['mention'] for x in instances]
    contexts = [' '.join((x['left_context'], x['right_context'])) for x in instances]

    logger.info('Encoding mentions.')
    mention_vectors = vectorizers['bigram'].transform(mentions)

    logger.info('Encoding contexts.')
    context_vectors = vectorizers['context'].transform(contexts)

    logger.info('Scoring mentions.')
    mention_scores = linear_kernel(mention_vectors, mention_vectors)

    logger.info('Scoring contexts.')
    context_scores = linear_kernel(context_vectors, context_vectors)

    scores = weights[0] * mention_scores + weights[1] * context_scores

    return scores


def find_threshold(scores, target, max_iters=100):
    logger.info(f'Finding threshold. Target # of clusts: {target}.')
    bounds = [0.0, 1.0]
    n_clusters = -1
    epsilon = scores.shape[0] / 1000.0
    logger.info(f'Epsilon: {epsilon}')
    i = 0
    while abs(n_clusters - target) > epsilon:
        threshold = (bounds[0] + bounds[1]) / 2
        clusters = cluster(scores, threshold)
        n_clusters = len(np.unique(clusters))
        logger.info(f'Threshold: {threshold}, # of clusts: {n_clusters}')
        if n_clusters < target:
            bounds[0] = threshold
        else:
            bounds[1] = threshold
    return clusters


def cluster(scores, threshold):
    logger.info('Clustering.')
    clusters = np.arange(scores.shape[0])
    for i, row in enumerate(scores):
        clusters[row > threshold] = clusters[i]
    return clusters


def main(args):
    instances = []
    entity_vocab = {}
    entity_ids = []
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            instances.append(data)
            if data['entity_id'] not in entity_vocab:
                entity_vocab[data['entity_id']] = len(entity_vocab)
            entity_ids.append(entity_vocab[data['entity_id']])
    entity_ids = np.array(entity_ids)

    with open(args.vectorizer, 'rb') as f:
        vectorizers = pickle.load(f)

    scores = score(instances, vectorizers)
    if args.threshold is not None:
        clusters = cluster(scores, args.threshold)
    else:
        target = len(entity_vocab)
        clusters = find_threshold(scores, target)

    with open(args.output, 'w') as g:
        for t, p in zip(entity_ids, clusters):
            print(f'{t}, {p}', file=g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--vectorizer', type=str, required=True)
    args = parser.parse_args()
    

    logging.basicConfig(level=logging.INFO)

    main(args)

