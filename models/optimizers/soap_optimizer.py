import torch
from models.optimizers.soap import SOAP


class SOAPOptimizer:
    """Wrapper for SOAP optimizer with Hydra instantiation support."""
    
    def __init__(self, params, lr: float = 3e-3, betas=(0.95, 0.95), shampoo_beta: float = -1,
                 eps: float = 1e-8, weight_decay: float = 0.01, precondition_frequency: int = 10,
                 max_precond_dim: int = 10000, merge_dims: bool = False, precondition_1d: bool = False,
                 normalize_grads: bool = False, data_format: str = "channels_first", correct_bias: bool = True):
        """
        Initialize SOAP optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam betas
            shampoo_beta: SOAP shampoo beta
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            precondition_frequency: How often to update preconditioner
            max_precond_dim: Maximum dimension of preconditioner
            merge_dims: Whether to merge dimensions
            precondition_1d: Whether to precondition 1D gradients
            normalize_grads: Whether to normalize gradients
            data_format: Data format for convolutions
            correct_bias: Whether to use bias correction
        """
        self.optimizer = SOAP(
            params=params,
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            data_format=data_format,
            correct_bias=correct_bias
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