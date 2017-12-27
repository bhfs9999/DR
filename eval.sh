#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python eval.py test centerloss_hem_iter --trained_model 25000_single_feature_.pth --conf_th 0.2 --model_name VggStride16_centerloss