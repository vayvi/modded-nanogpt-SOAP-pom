#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomgpt_multihead \
    model.n_head=32 \
    training.batch_size=24 \
    training.accumulation=16