#!/bin/bash

PYTHONPATH=././../../../ml-ocr-passport/ CUDA_VISIBLE_DEVICES=1 python back_predict.py \
    --config=configs/back_train.yml \
    --paths=configs/back_paths.yml \
    --weights=../../data/dumps/weights/seq2seq_test/best.pt
