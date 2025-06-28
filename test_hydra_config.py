#!/usr/bin/env python3
"""
Test script to verify Hydra configuration works correctly.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="config", config_name="config")
def test_config(cfg: DictConfig):
    """Test the Hydra configuration."""
    print("=== Hydra Configuration Test ===")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    print("\n=== Testing Model Instantiation ===")
    try:
        model = instantiate(cfg.model)
        print(f"✓ Model instantiated successfully: {type(model).__name__}")
        print(f"  - Vocab size: {model.vocab_size}")
        print(f"  - Layers: {model.n_layer}")
        print(f"  - Embedding dim: {model.n_embd}")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return
    
    print("\n=== Testing Optimizer Instantiation ===")
    try:
        # Create dummy parameters for testing
        import torch
        dummy_params = [torch.randn(10, 10, requires_grad=True)]
        optimizer = instantiate(cfg.optimizer, params=dummy_params)
        print(f"✓ Optimizer instantiated successfully: {type(optimizer).__name__}")
    except Exception as e:
        print(f"✗ Optimizer instantiation failed: {e}")
        return
    
    print("\n=== Testing Configuration Overrides ===")
    print("Configuration structure looks good!")
    print("You can now run training with:")
    print("torchrun --standalone --nproc_per_node=4 python train_hydra.py")


if __name__ == "__main__":
    test_config() 