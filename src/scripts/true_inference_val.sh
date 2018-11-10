#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=2 python ../true_inference.py \
    --config=../configs/true_seq2seq.yml \
    --test=../../data/test.csv \
    --out=../../data/amirassov_prediction_fullname_test.csv \
    --checkpoint=/mnt/hdd1/amirassov/cft/weights/seq2seq_transform512/best_109_0.45219.pt
