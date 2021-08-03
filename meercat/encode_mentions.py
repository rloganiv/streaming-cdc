"""
Use mention encoder to create embeddings.

NOTE: Right now mainly useful for doing inference with a pretrained RELIC w/out needing to load
entity embeddings.
"""
import argparse
import logging
import os
import random
import sys
import tempfile

import numpy as np
import torch
import transformers
from tqdm import tqdm

from meercat.nn_thresh import cluster
from meercat.models import MentionEncoderConfig, MentionEncoderModel
import meercat.utils as utils


logger = logging.getLogger(__name__)


def main(args):
    # Handle multi-GPU setup
    world_size = None
    if args.local_rank == -1:
        if args.model_parallel:
            embedding_device = torch.device('cuda:0')
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.model_parallel:
            embedding_device = torch.device('cuda', 2 * args.local_rank)
            device = torch.device('cuda', 2 * args.local_rank + 1)
        else:
            device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        world_size = torch.distributed.get_world_size()
    is_main_process = args.local_rank in [-1, 0]
    if args.debug:
        main_level = logging.DEBUG
        level = logging.DEBUG
    else:
        main_level = logging.INFO
        level = logging.WARN

    logging.basicConfig(level=main_level if is_main_process else level)
    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    # Still want an entity vocab for bookkeeping, even though we won't load embeddings.
    if args.entity_vocab is not None:
        entity_vocab = args.entity_vocab
    else:
        entity_vocab = args.model_name
    entity_tokenizer = utils.EntityTokenizer.from_pretrained(entity_vocab)
    counts = torch.tensor(entity_tokenizer.counts, device=device, dtype=torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
        use_fast=False,
    )
    logger.info('Adding separators to tokenizer')
    utils.add_mention_seps(tokenizer)

    logger.info('Loading model')
    config = MentionEncoderConfig.from_pretrained(
        args.model_name,
        entity_embedding_dim=args.entity_embedding_dim,
    )
    model = MentionEncoderModel.from_pretrained(
        args.model_name,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))  # Opt. resize for entity separators
    model.to(device)
    if args.model_parallel:
        model.entity_embeddings.to(embedding_device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    logger.info('Loading data')
    dataset = utils.ELIterableDataset(
        fname=args.input,
        tokenizer=tokenizer,
        entity_tokenizer=entity_tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    logger.info('Running inference')
    model.eval()
    embeddings = []
    true_clusters = []
    for model_inputs in data_loader:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = model_inputs.pop('labels')
        with torch.no_grad():
            embedding = model(**model_inputs, counts=counts)
        embeddings.append(embedding.to('cpu'))
        true_clusters.append(labels)
    embeddings = torch.cat(embeddings, dim=0)
    true_clusters = torch.cat(true_clusters, dim=0)
    for i, (true_cluster, embedding) in enumerate(zip(true_clusters, embeddings)):
        embedding_str = '\t'.join(str(x) for x in embedding.tolist())
        print(f'{i}\t{true_cluster.item()}\t{embedding_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Specific
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)

    # Model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--entity_embedding_dim', type=int, required=True)
    parser.add_argument('--entity_vocab', type=str, default=None)

    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--model_parallel', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(args)

