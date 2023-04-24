#!/bin/bash

#export LD_LIBRARY_PATH=/usr0/home/danielje/anaconda3/envs/10707/lib
CUDA_VISIBLE_DEVICES=7 python3 run_exps.py --seeds 1 2 3 4 5
