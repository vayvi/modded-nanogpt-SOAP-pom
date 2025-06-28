#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train_hydra.py \
    model=d12 \
    optimizer=soap \
    training.batch_size=32 \
    training.accumulation=16 \
    training.sequence_length=1024 \
    training.num_iterations=7000 \
    training.learning_rate=0.0018 \
    training.warmup_iters=250 \
    training.warmdown_iters=2000 \
    training.weight_decay=0.5 \
    evaluation.val_loss_every=128 \
    evaluation.val_max_steps=20