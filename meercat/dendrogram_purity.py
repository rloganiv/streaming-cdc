r"""Fast dendrogram purity computation.

TODO(rloganiv): Write usage
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from typing import List, Optional

import medmentions


@dataclass
class Node:
    uid: str
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list, repr=False)
    histogram: Counter = field(default_factory=Counter)

    def __repr__(self):
        return f'Node(uid={self.uid}, parent={self.parent.uid}, ' \
               f'children=[{", ".join(c.uid for c in self.children)}])'


def traverse(node):
    queue = [node]
    while queue:
        node = queue.pop()
        queue.extend(node.children)
        yield node


def load_metadata(f):
    metadata = {}
    for document in medmentions.parse_pubtator(f):
        for i, mention in enumerate(document.mentions):
            uid = f'{document.pmid}_{i}'
            # Semantic types are guaranteed unique for the subset of data
            # we're working with
            metadata[uid] = {
                'semantic_type': mention.semantic_types[0],
                'entity_id': mention.entity_id,
            }
    return metadata 


def load_dendrogram(f):
    lookup = {}
    reader = csv.reader(f, delimiter='\t')
    for uid, parent_uid, _ in reader:
        node = Node(uid=uid)
        lookup[uid] = node
        if parent_uid == 'None':
            root = node
        else:
            node.parent = lookup[parent_uid]
            node.parent.children.append(node)
    return root 


def num_pairs(metadata, cluster_by):
    """Computes the number of pairs, e.g., |P*|"""
    clusters = defaultdict(set)
    for key, value in metadata.items():
        clusters[value[cluster_by]].add(key)
    p_star = 0
    for cluster in clusters.values():
        size = len(cluster)
        p_star += size * (size - 1) / 2
    return p_star


def accumulate_purity(root, metadata, cluster_by):
    summand = 0
    for node in reversed(list(traverse(root))):
        if node.children:
            # Get node's histogram
            for child in node.children:
                node.histogram.update(child.histogram)
            # Add purity contributions
            n_leaves = sum(node.histogram.values())
            # Note: Assuming tree is binary
            for key in node.histogram:
                pairs = node.children[0].histogram[key] * node.children[1].histogram[key]
                summand += pairs * node.histogram[key] / n_leaves
        else:
            # Get metadata for leaf node
            cluster = metadata[node.uid][cluster_by]
            node.histogram[cluster] = 1
    return summand


def main(args):
    with open(args.medmentions_path, 'r') as f:
        metadata = load_metadata(f)
    with open(args.dendrogram_path, 'r') as f:
        root = load_dendrogram(f)
    p_star = num_pairs(metadata, args.cluster_by)
    summand = accumulate_purity(root, metadata, args.cluster_by)
    print(f'Dendrogram Purity: {p_star / summand: 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--medmentions-path', type=str)
    parser.add_argument('-d', '--dendrogram-path', type=str)
    parser.add_argument('-c', '--cluster-by', type=str,
                        choices=('semantic_type', 'entity_id'))
    args = parser.parse_args()

    main(args)

