#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=0 python ../train.py \
    --config=../configs/teacher_forcing_seq2seq.yml \
    --paths=../configs/paths.yml
