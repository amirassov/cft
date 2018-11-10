#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=2 python ../train_ulmfit.py \
    --config=../configs/ulmfit_seq2seq.yml \
    --paths=../configs/paths.yml
