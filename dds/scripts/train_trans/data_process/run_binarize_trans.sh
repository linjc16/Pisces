#!/bin/bash

# binarize dataset
DATAFOLD=$1
DATADIR=./data/processed/$DATAFOLD/

python3 fairseq_cli/preprocess.py \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --testpref $DATADIR/test \
    --destdir  $DATADIR/data-bin/ \
    --workers 4 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule

python3 fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.cell \
    --validpref $DATADIR/valid.cell \
    --testpref $DATADIR/test.cell \
    --destdir $DATADIR/data-bin/cell/ \
    --workers 4 --srcdict $DATADIR/cell.dict

python3 fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --testpref $DATADIR/test.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 4 --srcdict $DATADIR/label.dict
