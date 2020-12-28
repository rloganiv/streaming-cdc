import argparse
import collections

import scipy.sparse as sparse
from scipy.optimize import linear_sum_assignment


def sparse_from_set(clusters, total):
    row_ind = []
    col_ind = []
    data = []
    for i, cluster in enumerate(clusters.values()):
        for j in cluster:
            row_ind.append(i)
            col_ind.append(j)
            data.append(1)
    M = len(clusters)
    N = total
    return sparse.csr_matrix((data, (row_ind, col_ind)), (M,N))
        

def phi_4(k, r):
    """
    k : (keys, ents)
    r : (responses, ents)
    """
    intersections = k * r.transpose() # (keys, responses)
    k_counts = k.sum(axis=-1).reshape(-1, 1)
    r_counts = r.sum(axis=-1).reshape(1, -1)
    score = 2 * intersections / (k_counts + r_counts)
    return score


def main(args):
    # Indexed by cluster id
    true_clusters = collections.defaultdict(set)
    true_lookup = {}
    true_muc_lookup = {}
    pred_clusters = collections.defaultdict(set)
    pred_lookup = {}
    pred_muc_lookup = {}
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            t, p = [x.strip() for x in line.split(',')]
            true_clusters[t].add(i)
            true_lookup[i] = true_clusters[t]
            true_muc_lookup[i] = t
            pred_clusters[p].add(i)
            pred_lookup[i] = pred_clusters[p]
            pred_muc_lookup[i] = p
    total = i + 1  # You suck at coding!

    # Do MUC!
    precision_numerator = 0
    precision_denominator = 0
    for pred_cluster in pred_clusters.values():
        size = len(pred_cluster)
        partitions = len(set(true_muc_lookup[i] for i in pred_cluster))
        precision_numerator += size - partitions
        precision_denominator += size - 1
    muc_precision = precision_numerator / precision_denominator
    print(f'MUC Precision: {muc_precision}')

    recall_numerator = 0
    recall_denominator = 0
    for true_cluster in true_clusters.values():
        size = len(true_cluster)
        partitions = len(set(pred_muc_lookup[i] for i in true_cluster))
        recall_numerator += size - partitions
        recall_denominator += size - 1
    muc_recall = recall_numerator / recall_denominator
    print(f'MUC Recall: {muc_recall}')

    muc_f1 = 2 * muc_precision * muc_recall / (muc_precision + muc_recall)
    print(f'MUC F1: {muc_f1}')

    # Now do B-Cubed!
    b3_precision = 0
    b3_recall = 0
    for i in true_lookup.keys():
        numerator = len(true_lookup[i] & pred_lookup[i])
        b3_precision += numerator / len(pred_lookup[i])
        b3_recall += numerator / len(true_lookup[i])
    b3_precision /= total
    b3_recall /= total
    b3_f1 = 2 * b3_precision * b3_recall / (b3_precision + b3_recall)
    print(f'B3 Precision: {b3_precision}')
    print(f'B3 Recall: {b3_recall}')
    print(f'B3 F1: {b3_f1}')

    # Now do CEAF!
    k = sparse_from_set(true_clusters, total)
    r = sparse_from_set(pred_clusters, total)
    scores = phi_4(k, r)
    row_opt, col_opt = linear_sum_assignment(scores, maximize=True)
    numerator = scores[row_opt, col_opt].sum()
    ceaf_precision = numerator / len(pred_clusters)
    ceaf_recall = numerator / len(true_clusters)
    ceaf_f1 = 2 * ceaf_precision * ceaf_recall / (ceaf_precision + ceaf_recall)
    print(f'CEAF-e Precision: {ceaf_precision}')
    print(f'CEAF-e Recall: {ceaf_recall}')
    print(f'CEAF-e F1: {ceaf_f1}')

    line = '\t'.join('%0.3f' % x for x in [
        muc_precision,
        muc_recall,
        muc_f1,
        b3_precision,
        b3_recall,
        b3_f1,
        ceaf_precision,
        ceaf_recall,
        ceaf_f1
    ])
    print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    main(args)

