#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python train.py train debug_centerloss --model_name VggStride16_centerloss --gamma 0.03 --batch_size 1 --debug True --loss CenterLoss
CUDA_VISIBLE_DEVICES=2 python train.py train debug_centerloss --model_name VggStride16_centerloss --gamma 0.03 --batch_size 16 --loss CenterLoss --alpha 0.1