import argparse
import collections


def main(args):
    # Indexed by cluster id
    true_clusters = collections.defaultdict(set)
    true_lookup = {}
    pred_clusters = collections.defaultdict(set)
    pred_lookup = {}
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            t, p = [x.strip() for x in line.split(',')]
            true_clusters[t].add(i)
            true_lookup[i] = true_clusters[t]
            pred_clusters[p].add(i)
            pred_lookup[i] = pred_clusters[p]
    total = i + 1  # You suck at coding!
    precision = 0
    recall = 0
    for i in true_lookup.keys():
        numerator = len(true_lookup[i] & pred_lookup[i])
        precision += numerator / len(true_lookup[i])
        recall += numerator / len(pred_lookup[i])
    precision /= total
    recall /= total
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    main(args)

