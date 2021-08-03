import argparse
import json
import logging
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


def main(args):
    logger.info('Loading.')
    with open(args.input, 'r') as f:
        instances = [json.loads(line) for line in f]
    mentions = [x['mention'] for x in instances]
    contexts = [' '.join((x['left_context'], x['right_context'])) for x in instances]

    logger.info('Fitting.')
    bigram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), use_idf=False)
    bigram_vectorizer.fit(mentions)
    context_vectorizer = TfidfVectorizer(max_features=10000)
    context_vectorizer.fit(contexts)

    logger.info('Serializing.')
    with open(args.output, 'wb') as g:
        output_dict = {
            'bigram': bigram_vectorizer,
            'context': context_vectorizer,
        }
        pickle.dump(output_dict, g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

