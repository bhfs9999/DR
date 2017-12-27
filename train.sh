#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py train baseline_v1.4 --model_name VggStride16 --loss MultiBoxLoss --resume 15_single_feature_.pth