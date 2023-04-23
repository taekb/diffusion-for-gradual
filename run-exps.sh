#!/bin/bash

export LD_LIBRARY_PATH=/usr0/home/danielje/anaconda3/envs/10707/lib
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=0 python3 run_exps.py --exp no-adapt