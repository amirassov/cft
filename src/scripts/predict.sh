#!/bin/bash

PYTHONPATH=././../../../ml-ocr-passport/ CUDA_VISIBLE_DEVICES=0 python predict.py \
    --config=configs/train_ia.yml \
    --paths=configs/paths_ia.yml \
    --weights=../../data/dumps/weights/seq2seq_ia/best.pt
