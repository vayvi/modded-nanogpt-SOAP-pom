#!/usr/bin/env python3
"""
Standalone parameter counting script for the modded nanoGPT model.
This script initializes the model configuration and counts parameters without distributed training.
"""

import sys
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize

def count_parameters(model, print_breakdown=True):
    """
    Count the number of parameters in the model and optionally print a breakdown.
    
    Args:
        model: PyTorch model
        print_breakdown: Whether to print detailed parameter breakdown
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if print_breakdown:
        print("\n" + "="*60)
        print("MODEL PARAMETER BREAKDOWN")
        print("="*60)
        
        # Group parameters by component
        component_params = {}
        
        for name, param in model.named_parameters():
            # Extract component name (first part before first dot)
            component = name.split('.')[0] if '.' in name else name
            
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()
        
        # Print breakdown
        for component, params in sorted(component_params.items()):
            percentage = (params / total_params) * 100
            print(f"{component:<20}: {params:>12,} parameters ({percentage:>5.1f}%)")
        
        print("-"*60)
        print(f"{'Total':<20}: {total_params:>12,} parameters (100.0%)")
        print(f"{'Trainable':<20}: {trainable_params:>12,} parameters")
        print(f"{'Memory (fp32)':<20}: {total_params * 4 / 1024**2:>8.1f} MB")
        print(f"{'Memory (bf16)':<20}: {total_params * 2 / 1024**2:>8.1f} MB")
        print("="*60)
    
    return total_params, trainable_params

def analyze_model_architecture(model, cfg):
    """Analyze and print model architecture details."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Model Type: GPT with Polynomial Mixer (PoM)")
    print(f"Vocabulary Size: {cfg.model.vocab_size:,}")
    print(f"Number of Layers: {cfg.model.n_layer}")
    print(f"Number of Heads: {cfg.model.n_head}")
    print(f"Embedding Dimension: {cfg.model.n_embd}")
    print(f"PoM Degree: {cfg.model.degree}")
    print(f"PoM Expand Factor: {cfg.model.expand}")
    print(f"Attention Mechanism: Polynomial Mixer")
    print("="*60)

def main():
    """Main function to count parameters for different configurations."""
    
    # Initialize Hydra
    with initialize(version_base=None, config_path="config"):
        # Test different configurations
        configs_to_test = [
            ("pomgpt_baseline", "POM-GPT Baseline (12 layers, 1 head)"),
            ("pomgpt_multihead", "POM-GPT Multihead (12 layers, 32 heads)"),
            ("transformers_baseline", "Transformer Baseline (standard attention)")
        ]
        
        for experiment_name, description in configs_to_test:
            try:
                print(f"\n{'='*80}")
                print(f"ANALYZING: {description}")
                print(f"Configuration: {experiment_name}")
                print(f"{'='*80}")
                
                # Compose configuration
                cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])
                
                # Print configuration summary
                analyze_model_architecture(cfg, cfg)
                
                # Initialize model
                print(f"\nInitializing model...")
                model = instantiate(cfg.model.gpt)
                
                # Count parameters
                total_params, trainable_params = count_parameters(model, print_breakdown=True)
                
                # Calculate model size in different precisions
                print(f"\nModel Size Estimates:")
                print(f"FP32: {total_params * 4 / 1024**3:.2f} GB")
                print(f"FP16/BF16: {total_params * 2 / 1024**3:.2f} GB")
                print(f"INT8: {total_params / 1024**3:.2f} GB")
                
            except Exception as e:
                print(f"Error analyzing {experiment_name}: {e}")
                continue

if __name__ == "__main__":
    main() 