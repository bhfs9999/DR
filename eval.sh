#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python eval.py test centerloss_hem_w1 --trained_model 15_single_feature_.pth --conf_th 0.2 --model_name VggStride16_centerloss