#!/bin/bash

export WANDB_MODE=offline

# Width multiplier to embedding dimension mapping
declare -A WIDTH_MAP=([1.0]=768 [1.33]=1024 [1.66]=1280 [2.0]=1600)

for width_mult in "${!WIDTH_MAP[@]}"; do
    echo "Training: width_mult=$width_mult, n_embd=${WIDTH_MAP[$width_mult]}"
    
    torchrun --standalone --nproc_per_node=1 train.py \
        experiment=pomgpt_baseline \
        model.n_embd="${WIDTH_MAP[$width_mult]}" \
        experiment_name=pomgpt_baseline_mup_coord_check_${width_mult}_${n_embd} \
        training.batch_size=24 \
        training.accumulation=16 \
        mup.enabled=true \
        mup.cfg.width_multiplier=$width_mult
done