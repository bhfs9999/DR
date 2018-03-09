#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6
python eval.py \
test \
aug_centerloss \
--trained_model 35000_single_feature_.pth \
--conf_th 0.2 \
--model_name VggStride16_centerloss
