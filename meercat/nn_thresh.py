"""
O(n^2) nearest neighbor threshold clustering algorithm.
"""
import argparse
import csv
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def find_threshold(embeddings, entity_ids):
    """Find a heuristic threshold based on empirical Bayes classifier."""
    same_scores = embeddings.new_zeros(101)
    different_scores = embeddings.new_zeros(101)
    for i, (embedding, entity_id) in enumerate(zip(embeddings, entity_ids)):
        with torch.no_grad():
            scores = torch.mv(embeddings, embedding)
            same = entity_ids.eq(entity_id)
            same[i] = False  # Ignore self-similarity
            same_scores += torch.histc(scores[same], bins=101, min=0.0, max=1.0)
            different = entity_ids.ne(entity_id)
            different_scores += torch.histc(scores[different], bins=101, min=0.0, max=1.0)
    # same_scores /= same_scores.sum()
    # different_scores /= different_scores.sum()
    a = same_scores.sum() - same_scores.cumsum(dim=0)
    b = different_scores.cumsum(dim=0)
    thresholds = torch.linspace(0.0, 1.0, 101)
    optimal_threshold = thresholds[torch.argmax(a*b)]
    logger.info('Optimal threshold: %0.3f', optimal_threshold.item())
    return optimal_threshold
    

def cluster(embeddings, threshold=None):
    clusters = torch.arange(embeddings.size(0))
    for i, row in enumerate(embeddings):
        with torch.no_grad():
            scores = torch.mv(embeddings, row)
            clusters[scores > threshold] = clusters[i].clone()
    return clusters


def main(args):
    logger.info('Loading embeddings')
    entity_vocab = {}
    entity_ids = []
    embeddings = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            uid, entity, *embedding = line
            embedding = [float(x) for x in embedding]
            embeddings.append(embedding)
            if entity not in entity_vocab:
                entity_vocab[entity] = len(entity_vocab)
            entity_id = entity_vocab[entity]
            entity_ids.append(entity_id)

    logger.info('Clustering')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    if not args.dot_prod:
        embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)
    entity_ids = torch.tensor(entity_ids, dtype=torch.int64, device=device)
    if args.threshold:
        threshold = args.threshold
    else:
        logger.info('No threshold specified, searching for heuristic...')
        threshold = find_threshold(embeddings, entity_ids)
    clusters = cluster(embeddings, threshold=threshold)
    clusters = clusters.tolist()
    
    with open(args.output, 'w') as g:
        for t, p in zip(entity_ids, clusters):
            g.write('%i, %i\n' % (t, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('-d', '--dot_prod', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

