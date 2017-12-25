#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py train centerloss_hem_iter --model_name VggStride16_centerloss --loss CenterLoss --train_by_iter True