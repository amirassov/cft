#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=2 python ../ensemble_inference.py \
    --config=../configs/true_seq2seq.yml \
    --test=../../my_val.csv \
    --out=../../data/prediction_fullname_val.csv \
    --checkpoint=/mnt/hdd1/amirassov/cft/weights/seq2seq_transformq/best_*.pt
