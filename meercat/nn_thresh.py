"""
O(n^2) nearest neighbor threshold clustering algorithm.
"""
import argparse

import torch


def cluster(embeddings, threshold):
    clusters = torch.arange(embeddings.size(0))
    for i, row in enumerate(embeddings):
        with torch.no_grad():
            scores = torch.mv(embeddings, row)
            clusters[scores > threshold] = clusters[i].clone()
    return clusters

