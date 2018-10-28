#!/bin/bash

PYTHONPATH=././../../../ml-ocr-passport/ CUDA_VISIBLE_DEVICES=0 python train.py \
    --config=configs/train_ia.yml \
    --paths=configs/paths_ia.yml
