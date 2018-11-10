#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=2 python ../inference.py \
    --config=../configs/ulmfit_seq2seq.yml \
    --test=../../data/test.csv \
    --out=../../data/ulmfit_prediction_fullname_test.csv \
    --weights=/mnt/hdd1/amirassov/cft/weights/ulmfit_seq2seq/best.pt
