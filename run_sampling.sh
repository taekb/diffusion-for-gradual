#!/bin/bash
echo "Start Time: $(date)"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True"
MODEL_PATH="./model/ema_0_750000.pt"
LOG_PATH="./sampled/gpu0/class_0"
SAMPLING_FLAG="--batch_size 512 --num_sample 2560"

CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=$LOG_PATH python improved-diffusion/scripts/image_sample.py --model_path $MODEL_PATH $SAMPLING_FLAG $MODEL_FLAGS $DIFFUSION_FLAGS


echo "End Time: $(date)"