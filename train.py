import os
import sys
import uuid
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from models import GPT
from data import DistributedDataLoader

# Set float32 matmul precision to match reference implementation
torch.set_float32_matmul_precision('high')

def print0(*args, **kwargs):
    """Modified print that only prints from the master process."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def get_lr(step: int, num_iterations: int, warmup_iters: int, warmdown_iters: int) -> float:
    """
    Calculate learning rate scale for current step.
    
    Args:
        step: Current training step
        num_iterations: Total number of iterations
        warmup_iters: Number of warmup iterations
        warmdown_iters: Number of warmdown iterations
        
    Returns:
        Learning rate scale
    """
    assert step <= num_iterations
    
    # 1) linear warmup for warmup_iters steps
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    # 2) constant lr for a while
    elif step < num_iterations - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - step) / warmdown_iters
        return decay_ratio


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print0(f"Running pytorch {torch.version.__version__}")
    print0(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set up distributed training
    assert torch.cuda.is_available()
    dist.init_process_group(backend=cfg.distributed.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0)
    
    # Load data
    train_loader = DistributedDataLoader(
        cfg.data.train.input_bin,
        cfg.training.batch_size,
        cfg.training.sequence_length,
        ddp_rank,
        ddp_world_size
    )
    print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    
    val_loader = DistributedDataLoader(
        cfg.data.val.input_bin,
        cfg.training.batch_size,
        cfg.training.sequence_length,
        ddp_rank,
        ddp_world_size
    )
    print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    
    x, y = train_loader.next_batch()
    
    # Initialize model using Hydra instantiate
    model = instantiate(cfg.model.gpt)
    model = model.cuda()
    
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    
    print0("compiling the model...")
    if cfg.hardware.compile:
        model = torch.compile(model)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=cfg.distributed.find_unused_parameters)
    raw_model = model.module
    
    # Set up context manager for mixed precision
    ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, cfg.hardware.dtype))
    
    # Initialize optimizer using model's configure_optimizers method
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,  # Use config value (0.5) to match reference
        learning_rate=cfg.training.learning_rate,
        betas=(0.9, 0.95)  # Fixed betas to match reference implementation
    )
    
    # Set up logging
    run_id = str(uuid.uuid4())
    if master_process:
        log_dir = Path(cfg.logging.log_dir) / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / "log.txt"
        logfile.touch()  # create empty log file
    
    # Training loop
    for step in range(cfg.training.num_iterations + 1):
        last_step = (step == cfg.training.num_iterations)
        
        # Validation
        if (last_step or (cfg.evaluation.val_loss_every > 0 and step % cfg.evaluation.val_loss_every == 0)):
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(cfg.evaluation.val_max_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= cfg.evaluation.val_max_steps
            
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")
        
        # Save checkpoint
        if master_process and (last_step or (cfg.evaluation.save_every > 0 and step % cfg.evaluation.save_every == 0)):
            checkpoint = {
                'step': step,
                'config': cfg,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, log_dir / f'state_step{step:06d}.pt')
        
        if last_step:
            break
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Training
        model.train()
        for _ in range(cfg.training.accumulation):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            x, y = train_loader.next_batch()
            loss.backward()
        
        # Gradient accumulation
        for p in model.parameters():
            p.grad /= cfg.training.accumulation
        
        # Learning rate scheduling
        lr_scale = get_lr(
            step,
            cfg.training.num_iterations,
            cfg.training.warmup_iters,
            cfg.training.warmdown_iters
        )
        optimizer.scale_lrs(lr_scale)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        # Logging
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        tokens_per_second = ddp_world_size * cfg.training.batch_size * cfg.training.sequence_length / (t1 - t0)
        
        if step % cfg.logging.log_every == 0:
            print0(f"step {step+1:4d}/{cfg.training.num_iterations} | "
                   f"train loss {train_loss.item():.4f} | "
                   f"lr_scale {lr_scale:.2e} | "
                   f"({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trl:{train_loss.item()}\n")
    
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main() 