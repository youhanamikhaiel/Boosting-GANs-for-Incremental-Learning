#!/bin/bash
export PATH="/home/ymikhaiel/anaconda3/bin:$PATH"
CUDA_VISIBLE_DEVICES=0,1 python train_classifier.py
