#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomgpt_baseline \
    experiment_name=pomgpt_baseline_mup_coord_check_2.0_1600 \
    training.batch_size=12 \
    training.accumulation=32 \
    mup.enabled=true \
    mup.cfg.width_multiplier=2.0 \
    model.n_embd=1600