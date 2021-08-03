#! /usr/bin/env bash


GRINCH_DIR=/home/rlogan/projects/grinch/
EMBEDDING_PATH=${1}
EMBEDDING_FILE="${EMBEDDING_PATH##*/}"
EMBEDDING_NAME="${EMBEDDING_FILE%%.*}"
OUTPUT_DIR=data/grinch_out/$EMBEDDING_NAME/

echo "Input: $EMBEDDING_PATH"
echo "Writing clusters to: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
source $GRINCH_DIR/bin/setup.sh


java -Xmx20G -classpath $GRINCH_JARPATH grinch.eval.RunGrinch \
  --input $EMBEDDING_PATH \
  --outdir $OUTPUT_DIR \
  --algorithm "GrinchFull" \
  --dataset $EMBEDDING_NAME \
  --max-leaves None \
  --clusters None \
  --threads 1 \
  --max-frontier-par 50 \
  --k 5 \
  --max_num_leaves=-1 \
  --graft_beam 100000000 \
  --rotation_size_cap 100000000 \
  --graft_size_cap 100000000 \
  --restruct_size_cap 100000000 \
  --perform_rotate true \
  --perform_graft true \
  --perform_restruct true \
  --single_graft_search false \
  --single_elimination false \
  --nsw_r 1 \
  --exact_nn true \
  --max_degree 25 \
  --linkage coslink \
  --max_search_time 1000 > $OUTPUT_DIR/grinch.log

