#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=3 python train.py train centerloss_hem_onlypos --model_name VggStride16_centerloss --loss CenterLoss --only_pos_centerloss True
CUDA_VISIBLE_DEVICES=3 python train.py train debug_centerloss --model_name VggStride16_centerloss --only_pos_centerloss True --batch_size 1 --debug True --loss CenterLoss