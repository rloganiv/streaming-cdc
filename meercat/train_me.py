"""
Mention Encoder Training.
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

from meercat.models import MentionEncoderConfig, MentionEncoderModel
import meercat.utils as utils


logger = logging.getLogger(__name__)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


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

    # Basic initialization stuff
    set_random_seeds(args.seed)

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

    logger.info('Loading train')
    train_data = utils.APIterableDataset.load(
        fname=args.train,
        tokenizer=tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )
    logger.info('Loading dev')
    dev_data = utils.APIterableDataset.load(
        fname=args.dev,
        tokenizer=tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
    )
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    if is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=args.output_dir,
        )

    logger.info('Starting training')
    best_dev_loss = float('Inf')
    n_iter = 0
    for epoch in range(args.epochs):
        logger.info('Epoch: %s', epoch)

        # Train loop
        model.train()
        optimizer.zero_grad()
        total = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        if args.local_rank in [-1, 0]:
            train_loader = tqdm(train_loader, file=sys.stdout)
        for i, (pos_1, pos_2, neg) in enumerate(train_loader):
            n_iter += 1
            pos_1 = {k: v.to(device) for k, v in pos_1.items()}
            pos_2 = {k: v.to(device) for k, v in pos_2.items()}
            neg = {k: v.to(device) for k, v in neg.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16):
                pos_1_enc = model(**pos_1)
                pos_2_enc = model(**pos_2)
                neg_enc = model(**neg)
                pos_sim = torch.einsum('bi,bi->b', pos_1_enc, pos_2_enc)
                neg_sim = torch.einsum('bi,bi->b', pos_1_enc, neg_enc)
                loss = args.margin - pos_sim + neg_sim
                loss[loss<0] = 0.0
                total_loss += (loss.sum()).detach()
                loss = loss.mean()
                loss /= args.accumulation_steps  # fp16 adjustment
            total += pos_1_enc.size(0)
            scaler.scale(loss).backward()

            if i % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if not i % args.log_every and i != 0:
                # If distrubuted, then need to accumulate results across
                # processes.
                if args.local_rank != -1:
                    torch.distributed.reduce(total_loss, 0)
                    torch.distributed.reduce(total, 0)
                if is_main_process:
                    writer.add_scalar('Loss/train', (total_loss / (total + 1e-13)).item(), n_iter)
                # Reset accumulators
                total = torch.tensor(0.0, device=device)
                total_loss = torch.tensor(0.0, device=device)

        # Eval loop
        model.eval()
        total = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        if args.local_rank in [-1, 0]:
            dev_loader = tqdm(dev_loader, file=sys.stdout)
        for pos_1, pos_2, neg in dev_loader:
            n_iter += 1
            pos_1 = {k: v.to(device) for k, v in pos_1.items()}
            pos_2 = {k: v.to(device) for k, v in pos_2.items()}
            neg = {k: v.to(device) for k, v in neg.items()}
            with torch.no_grad():
                pos_1_enc = model(**pos_1)
                pos_2_enc = model(**pos_2)
                neg_enc = model(**neg)
                pos_sim = torch.einsum('bi,bi->b', pos_1_enc, pos_2_enc)
                neg_sim = torch.einsum('bi,bi->b', pos_1_enc, neg_enc)
                loss = args.margin - pos_sim + neg_sim
                loss[loss<0] = 0.0
                total_loss += (loss.sum()).detach()
                loss = loss.mean()
            total += pos_1_enc.size(0)

        # Gather accuracy accross processes
        if args.local_rank != -1:
            torch.distributed.reduce(total_loss, 0)
            torch.distributed.reduce(total, 0)
        total_loss = (total_loss / total).item()
        if is_main_process:
            writer.add_scalar('Loss/dev', total_loss, epoch)

        # Serialize if best
        if total_loss < best_dev_loss:
            logger.info('Best dev loss so far. Saving.')
            if args.local_rank == -1:
                model.save_pretrained(args.output_dir)
            elif args.local_rank == 0:
                model.module.save_pretrained(args.output_dir)
            if is_main_process:
                config.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        best_dev_loss = min(best_dev_loss, total_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Specific
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--dev', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed',  type=int, default=1234)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--log_every', type=int, default=100)

    # Model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--entity_embedding_dim', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--margin', type=float, default=1.0)

    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--model_parallel', action='store_true')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(args)

