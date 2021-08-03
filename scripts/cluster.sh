#!/usr/bin/env bash
python meercat/train_el.py \
--train ext_data/MedMentions/train.jsonl  \
--dev ext_data/MedMentions/dev.jsonl.seen \
--test ext_data/MedMentions/dev.jsonl \
--skip_train \
--entity_vocab ext_data/MedMentions/entity_vocab.txt \
--model_name ext_data/mm-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_mm-bert-base-uncased_data_medmentions-dev.clusters
python meercat/train_el.py \
--train ext_data/MedMentions/train.jsonl  \
--dev ext_data/MedMentions/dev.jsonl.seen \
--test ext_data/MedMentions/dev.jsonl \
--skip_train \
--entity_vocab ext_data/KILT/entity_vocab.txt \
--model_name ext_data/wiki-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_wiki-bert-base-uncased_data_medmentions-dev.clusters
python meercat/train_el.py \
--train ext_data/MedMentions/train.jsonl  \
--dev ext_data/MedMentions/dev.jsonl.seen \
--test ext_data/MedMentions/dev.jsonl \
--skip_train \
--entity_vocab ext_data/ecbplus_processed/entity_vocab.txt \
--model_name ext_data/ecbplus-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_ecbplus-bert-base-uncased_data_medmentions-dev.clusters
python meercat/train_el.py \
--train ext_data/ecbplus_processed/train.jsonl  \
--dev ext_data/ecbplus_processed/dev.jsonl.seen \
--test ext_data/ecbplus_processed/dev.jsonl \
--skip_train \
--entity_vocab ext_data/MedMentions/entity_vocab.txt \
--model_name ext_data/mm-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_mm-bert-base-uncased_data_ecbplus-dev.clusters
python meercat/train_el.py \
--train ext_data/ecbplus_processed/train.jsonl  \
--dev ext_data/ecbplus_processed/dev.jsonl.seen \
--test ext_data/ecbplus_processed/dev.jsonl \
--skip_train \
--entity_vocab ext_data/ecbplus_processed/entity_vocab.txt \
--model_name ext_data/ecbplus-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_ecbplus-bert-base-uncased_data_ecbplus-dev.clusters
python meercat/train_el.py \
--train ext_data/ecbplus_processed/train.jsonl  \
--dev ext_data/ecbplus_processed/dev.jsonl.seen \
--test ext_data/ecbplus_processed/dev.jsonl \
--skip_train \
--entity_vocab ext_data/KILT/entity_vocab.txt \
--model_name ext_data/wiki-bert-base-uncased \
--entity_embedding_dim 256 > ext_data/model_wiki-bert-base-uncased_data_ecbplus-dev.clusters
