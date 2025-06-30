#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_baseline \
    training.batch_size=48 \
    training.accumulation=2