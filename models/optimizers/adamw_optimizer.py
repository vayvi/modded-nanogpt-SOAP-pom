import torch.optim as optim


class AdamWOptimizer:
    """Wrapper for AdamW optimizer with Hydra instantiation support."""
    
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam betas
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
        """
        self.optimizer = optim.AdamW(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    def step(self):
        """Perform optimization step."""
        return self.optimizer.step()
    
    def zero_grad(self, **kwargs):
        """Zero gradients."""
        return self.optimizer.zero_grad(**kwargs)
    
    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        return self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Get parameter groups."""
        return self.optimizer.param_groups 