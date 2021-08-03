import argparse
import csv
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Node:
    uid: str
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)
    entity_id: Optional[str] = None
    score: float = 1.0
    embedding: Optional[torch.FloatTensor] = None
    n_leaves: float = 1.0

    def __repr__(self):
        return f'Node(uid={self.uid}, parent={self.parent.uid}, ' \
               f'children=[{", ".join(c.uid for c in self.children)}])'


def traverse(node):
    queue = [node]
    while queue:
        node = queue.pop()
        queue.extend(node.children)
        yield node


def load_embeddings(f):
    embeddings = []
    reader = csv.reader(f, delimiter='\t')
    for _, _, *embedding in reader:
        embeddings.append([float(x) for x in embedding])
    embeddings = torch.tensor(embeddings)
    return embeddings


def load_dendrogram(f):
    lookup = {}
    reader = csv.reader(f, delimiter='\t')
    for uid, parent_uid, label in reader:
        node = Node(uid=uid)
        lookup[uid] = node
        if parent_uid == 'None':
            root = node
        else:
            node.parent = lookup[parent_uid]
            node.parent.children.append(node)
        if label != 'None':
            node.entity_id = label
    return root 


def leaves(root):
    leaves = []
    for node in traverse(root):
        if not node.children:
            leaves.append(node)
    return leaves


def compute_score(node, dot_prod=False):
    with torch.no_grad():
        left_embedding = node.children[0].embedding / node.children[0].n_leaves
        right_embedding = node.children[1].embedding / node.children[1].n_leaves
        if not dot_prod:
            left_embedding /= torch.norm(left_embedding)
            right_embedding /= torch.norm(right_embedding)
        return torch.dot(left_embedding, right_embedding)



def main(args):
    with open(args.dendrogram, 'r') as f:
        root = load_dendrogram(f)
    with open(args.embeddings, 'r') as f:
        embeddings = load_embeddings(f)

    # Assign/Propagate embeddings
    for node in reversed(list(traverse(root))):
        if not node.children:
            index = int(node.uid)
            node.embedding = embeddings[index]
        else:
            node.embedding = sum(x.embedding for x in node.children)
            node.n_leaves = sum(x.n_leaves for x in node.children)
            node.score = compute_score(node, args.dot_prod)

    queue = [root]
    i = 0
    with open(args.output, 'w') as g:
        while queue:
            node = queue.pop()
            if node.score < args.threshold:
                queue.extend(node.children)
            else:
                for leaf in leaves(node):
                    line = f'{leaf.entity_id}, {i}\n'
                    g.write(line)
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dendrogram', type=str)
    parser.add_argument('--embeddings', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('-d', '--dot_prod', action='store_true')
    args = parser.parse_args()

    main(args)

