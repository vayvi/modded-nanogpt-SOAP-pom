import torch
import torch.nn as nn
import torch.nn.functional as F
import models.pom as pom
import math
from copy import deepcopy

def rmsnorm(x0, eps=1e-6):
    """RMS normalization function (matching reference implementation)."""
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class RMSNorm(nn.Module):
    """RMS normalization module."""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.type_as(x0)


class Rotary(torch.nn.Module):
    """Rotary position embeddings."""
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfPoM(nn.Module):
    """Causal self-attention using Polynomial Mixer."""
    
    def __init__(self, n_embd, degree, expand, n_head):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        self.pom = pom.PoM(self.n_embd, self.degree, self.expand, self.n_head, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        mask = torch.tril(torch.ones((T, T))).unsqueeze(0)
        return self.pom(x, x, mask)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, degree, expand, n_head, mup_enabled=False):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.mup_enabled = mup_enabled

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.mup_enabled:
            attention_scale = 1.0 / k.size(-1)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=attention_scale)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Multi-layer perceptron block."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with PoM attention and MLP."""
    
    def __init__(self, mixing_layer, n_embd, n_layer):
        super().__init__()
        self.attn = deepcopy(mixing_layer) #CausalSelfPoM(n_embd, degree, expand, n_head)

        # Reinitialize with pytorch defaults
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=1)
        self.mlp = MLP(n_embd)
        self.attn_scale = (1 / (2 * n_layer)**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class GPT(nn.Module):
    """GPT model with Polynomial Mixer attention."""
    
    def __init__(self, mixing_layer, vocab_size: int = 50257, n_layer: int = 12, n_head: int = 12, n_embd: int = 768, mup_enabled: bool = False, mup_config=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            h=nn.ModuleList([Block(mixing_layer, self.n_embd, self.n_layer) for _ in range(self.n_layer)]),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.rotary = Rotary(self.head_dim)
        self.mup_enabled = mup_enabled
        self.mup_config = mup_config
        if self.mup_enabled: # change initialization for some params
            for pn, p in self.named_parameters():
                # Adjust hidden weight initialization variance by 1 / mup_width_multiplier
                if pn.endswith('c_attn.weight') or pn.endswith('c_fc.weight') or pn.endswith('po_proj.weight') or pn.endswith('se_proj.weight') or pn.endswith('ag_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.mup_config.init_std / math.sqrt(self.mup_config.width_multiplier))
                elif pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.mup_config.init_std / math.sqrt(2 * self.n_layer * self.mup_config.width_multiplier))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_logits: bool = True):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices for loss computation
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (logits, loss) if targets provided, else just logits
        """
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(x)
        x = apply_rotary_emb(x, cos, sin)
        x = x.view(B, T, C)
        if self.mup_enabled:
            x *= self.mup_config.input_alpha
        
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            if self.mup_enabled:
                # Scaling `x` instead of `logits` allows coord check to log change
                x *= self.mup_config.output_alpha / self.mup_config.width_multiplier
                
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple):
        """
        Configure optimizers for the model.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas
            
            Returns:
                Combined optimizer
        """
        from models.optimizers.combined_optimizer import CombinedOptimizer
        from models.optimizers.adamw_optimizer import AdamWOptimizer
        from models.optimizers.soap import SOAP

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter out those that do not require grad
        
        optimizers = []
        
        # AdamW for lm_head parameters (wte weights are tied to lm_head, so we only include lm_head)
        lm_head_optimizer = AdamWOptimizer(
            self.lm_head.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=0  # No weight decay for lm_head
        )
        optimizers.append(lm_head_optimizer)
        
        if self.mup_enabled and not self.mup_config.disable_hidden_lr_scaling:
            # MUP case: separate parameters by type for SOAP
            mup_decay_params = []
            decay_params = []
            nodecay_params = []
            
            # Exclude lm_head parameters (already handled by AdamW above)
            transformer_param_dict = {pn: p for pn, p in param_dict.items() if not pn.startswith('lm_head')}
            
            for n, p in transformer_param_dict.items():
                if p.dim() >= 2:
                    if n.endswith('c_attn.weight') or n.endswith('c_fc.weight') or n.endswith('c_proj.weight') or n.endswith('po_proj.weight') or n.endswith('se_proj.weight') or n.endswith('ag_proj.weight'):
                        mup_decay_params.append(p)
                    else:
                        decay_params.append(p)
                else:
                    nodecay_params.append(p)
            
            num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num mup decayed parameter tensors: {len(mup_decay_params)}, with {num_mup_decay_params:,} parameters")
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            
            
            # Create separate SOAP optimizers for each parameter group
            mup_decay_optimizer = SOAP(
                [{'params': mup_decay_params, 'weight_decay': weight_decay, 'lr_scale': 1/self.mup_config.width_multiplier}], # FIXME: may need to scale weight_decay as well
                lr=learning_rate,
                betas=(0.95, 0.95),
                weight_decay=0, # FIXME: is it 0 or weight_decay?
                precondition_frequency=10
            )
            optimizers.append(mup_decay_optimizer)
            
            decay_optimizer = SOAP(
                [{'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1}],
                lr=learning_rate, 
                betas=(0.95, 0.95),
                weight_decay=0,
                precondition_frequency=10
            )
            optimizers.append(decay_optimizer)
            
            nodecay_optimizer = SOAP(
                [{'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1}],
                lr=learning_rate,
                betas=(0.95, 0.95), 
                weight_decay=0,
                precondition_frequency=10
            )
            optimizers.append(nodecay_optimizer)
            
        else:
            # Non-MUP case: SOAP for transformer layers
            transformer_optimizer = SOAP(
                self.transformer.h.parameters(),
                lr=learning_rate,
                betas=(0.95, 0.95),  # Fixed betas for SOAP
                weight_decay=0,  # No weight decay for transformer layers
                precondition_frequency=10  # Fixed precondition frequency
            )
            optimizers.append(transformer_optimizer)
        
        return CombinedOptimizer(optimizers)