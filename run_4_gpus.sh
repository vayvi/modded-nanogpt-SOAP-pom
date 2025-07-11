#!/bin/bash

# Run training with Hydra configuration
export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_multihead \
    model.n_head=192 \
    training.batch_size=48 \
    training.accumulation=2