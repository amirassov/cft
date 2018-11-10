#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=1 python ../predict.py \
    --config=../configs/seq2seq.yml \
    --paths=../configs/paths.yml \
    --weights=/mnt/hdd1/amirassov/cft/weights/seq2seq_hard/best.pt

