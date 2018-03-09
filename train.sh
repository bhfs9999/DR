#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
python3 train.py \
train \
aug_centerloss \
--model_name VggStride16_centerloss \
--loss CenterLoss \
--train_by_iter True \
--only_centerloss False \
--batch_size 16
