#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python3 train.py train onlycenterloss_pos --model_name VggStride16_centerloss --loss CenterLoss --train_by_iter True --only_centerloss True --batch_size 4