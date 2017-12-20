#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python eval.py test centerloss_v0_ --trained_model 15_single_feature_.pth --conf_th 0.2