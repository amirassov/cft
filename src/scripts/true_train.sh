#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=1 python ../true_train.py \
    --config=../configs/true_seq2seq.yml \
    --paths=../configs/paths.yml
