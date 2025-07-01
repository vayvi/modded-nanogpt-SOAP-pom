import torch.nn as nn
from models.gpt import GPT, Block, CausalSelfAttention

class AlternateGPT(GPT):
    """Hybrid GPT model with alternating PoM and Self-Attention layers."""
    
    def __init__(self, mixing_layer, vocab_size: int = 50257, n_layer: int = 12, n_head: int = 12, n_embd: int = 768, 
                 degree: int = 2, expand: int = 2, mixing_layer_alternate: nn.Module = None, pom_to_sa_ratio: float = 5):
        """
        Args:
            pom_to_sa_ratio: Ratio of PoM to self-attention layers (default 5 = 1 sa layer to 5 pom layers)
        """
        # if n_layer % (pom_to_sa_ratio + 1) != 0:
        #     raise ValueError(f"n_layer must be a multiple of (pom_to_sa_ratio + 1) for proper hybrid ratio, got {n_layer}")
        
        if mixing_layer_alternate is None:
            mixing_layer_alternate = CausalSelfAttention(n_embd, degree, expand, n_head)
        
        mixing_layers = []
        for i in range(n_layer):
            if i % (pom_to_sa_ratio + 1) == pom_to_sa_ratio:  # Every pom_to_sa_ratio + 1 layers
                mixing_layers.append(mixing_layer_alternate)
            else:
                mixing_layers.append(mixing_layer)
        
        super().__init__(mixing_layer=None, vocab_size=vocab_size, n_layer=n_layer, 
                        n_head=n_head, n_embd=n_embd)
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            h=nn.ModuleList([Block(mixing_layer, self.n_embd, self.n_layer) for mixing_layer in mixing_layers]),
        ))
        
        self.transformer.wte.weight = self.lm_head.weight