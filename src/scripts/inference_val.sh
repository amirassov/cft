#!/bin/bash

PYTHONPATH=/home/amirassov/cft CUDA_VISIBLE_DEVICES=2 python ../inference.py \
    --config=../configs/teacher_forcing_seq2seq.yml \
    --test=../../my_val.csv \
    --out=../../data/prediction_fullname_val.csv \
    --weights=/mnt/hdd1/amirassov/cft/weights/teacher_forcing_seq2seq/best.pt
