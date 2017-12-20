#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py train centerloss_hem --center_loss True --model_name VggStride16_centerloss