#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py train debug_centerloss --center_loss True --model_name VggStride16_centerloss