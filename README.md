# Modded-NanoGPT (SOAP optimizer experiments with Hydra Configuration)

This is a fork of [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan, which is a variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. The original description is pasted at the end of this file.

## Architecture & Configuration

This repository has been refactored to use [Hydra](https://hydra.cc/) for configuration management, providing a modular and flexible way to configure experiments. The configuration system is organized as follows:

### Configuration Structure

```
config/
├── config.yaml              # Main configuration with defaults
├── experiment/               # Pre-defined experiment configurations
│   ├── pomgpt_baseline.yaml
│   └── transformers_baseline.yaml
├── model/                    # Model architecture configurations
│   ├── pomgpt.yaml          # POM-based GPT configuration
│   ├── transformer.yaml     # Standard transformer configuration
│   ├── gpt/                 # GPT size configurations
│   │   ├── d12.yaml         # 12-layer (124M parameters)
│   │   ├── d24.yaml         # 24-layer 
│   │   ├── d36.yaml         # 36-layer
│   │   └── d48.yaml         # 48-layer
│   └── mixing_layer/        # Mixing layer options
│       ├── pom.yaml         # Polynomial mixing (POM)
│       └── self-attention.yaml # Standard self-attention
├── optimizer/               # Optimizer configurations
│   ├── soap.yaml           # SOAP optimizer settings
│   └── adamw.yaml          # AdamW optimizer settings
├── data/                    # Dataset configurations
│   └── fineweb.yaml        # FineWeb dataset settings
└── training/                # Training configurations
```

### Usage

#### Running Experiments

To execute training with the default configuration:
```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py
```

**Single GPU:**
```bash
./run_single_gpu.sh
```

**Multi-GPU (4 GPUs):**
```bash
./run_4_gpus.sh
```

#### Configuration Override Examples

You can override any configuration parameter via command line:

```bash
# Run with different optimizer
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_baseline \
    optimizer=adamw

# Run with different model size
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_baseline \
    model.gpt=d24

# Override training parameters
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_baseline \
    training.learning_rate=0.003 \
    training.batch_size=32

# Run standard transformer instead of POM
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=transformers_baseline
```

#### Creating Custom Experiments

Create new experiment files in `config/experiment/` to define reusable configurations:

```yaml
# config/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /model: transformer
  - override /optimizer: soap

training:
  learning_rate: 0.003
  batch_size: 64
```

Then run with: `train.py experiment=my_experiment`

### Benefits of Hydra Configuration

- **Modularity**: Mix and match different components (models, optimizers, datasets) easily
- **Reproducibility**: Complete configuration is logged and can be easily reproduced
- **Experimentation**: Quick parameter sweeps and ablation studies via command-line overrides
- **Organization**: Clean separation of concerns with logical config grouping
- **Extensibility**: Easy to add new models, optimizers, or experiments without code changes

## SOAP Optimizer Experiments

We wanted to test how the new optimizer in Modded-NanoGPT compares to [SOAP](https://github.com/nikhilvyas/SOAP). So we run the code as it is with following changes:

0. **Baseline**: The baseline loss from new optimizer was 3.279, with updated hyperparameters (https://x.com/kellerjordan0/status/1842616414894719310) it was reduced to 3.271. We have launched the below experiments with the hyperparams from the first runs, perhaps a slight improvement can be made with switching to the new hyperparams.

1. **SOAP Optimizer**: We replace OrthogonalNesterov optimizer with SOAP optimizer. We keep the usage of AdamW optimizer for first/last layer to make the comparison fairer. Hyperparams are: LR=.0018 (same as AdamW LR in Modded-NanoGPT in the first baseline), betas=(.95, .95) (SOAP default), weight_decay=0 (same as Modded-NanoGPT), precondition_frequency=10 (SOAP default). We get loss 3.2564, a slightly higher SOAP LR of .003 gives 3.2561.

2. **Performance**: For a run with 10% lesser iterations we get a loss of 3.2702. But we note that this is only iterations and not wall clock time. While we estimate the overhead* (over non-fused AdamW) to be ~ 5 to 10% in 1gpu experiments from prior runs this needs to be confirmed. Re 1gpu vs multigpu the overhead of the optimizers can also be distributed as done in [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo), so we should expect similar overhead in 1 gpu vs multigpu experiments. This argument also applies to the new optimizer in Modded-NanoGPT, implying that the overhead for the new optimizer in Modded-NanoGPT should be < 1%.

3. **Memory Optimization**: SOAP has a larger memory overhead, to reduce the overhead we have one-sided/factorized versions of SOAP in the [paper](https://arxiv.org/abs/2409.11321). We are currently running these experiments.

4. **Future Work**: We are also planning to compare the two optimizers in the small batch size setting.

*We are hoping to improve this with a lower precision implementation of SOAP.

---------------


# Modded-NanoGPT

This is a variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. It:
* Trains 2.7x more efficiently (taking only 3.67B tokens instead of 10B to reach the same validation loss).
* Has shorter code (497 lines instead of 860).
* Implements architectural modernizations (rotary embeddings and RMSNorm).
* Implements a new optimizer.

To execute the training, run the following commands on an 8xA100 or 8xH100 node.
They complete in <45min on an 8xH100 with decent internet connection.
```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py
./run_4_gpus.sh  # For 4+ GPUs
# or
./run_single_gpu.sh  # For single GPU
```

This will train a 124M-parameter transformer for 7000 steps on 3.67B tokens of Fineweb [1], achieving ~3.280 validation
loss with the modded-nanogpt optimizer or ~3.256 validation loss with SOAP optimizer.
For comparison, the default llm.c PyTorch trainer yields [~3.285 validation loss after training for 10B tokens](https://github.com/karpathy/llm.c/discussions/481).

---

## Figures

Figure 1. Proposed optimizer vs. a well-tuned AdamW.
![](img/fig_optimizer.png)

---

## Proposed optimizer

For this training scenario, the proposed optimizer has the following properties:
* Half the memory usage of Adam
* 1.36x faster training
* <3% wallclock overhead

It is defined as follows:

![](img/algo_optimizer.png)

Where NewtonSchulz5 is the following Newton-Schulz iteration [2, 3]:
```python
@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T 
    for _ in range(steps):
        A = X @ X.T 
        B = A @ X 
        X = a * X + b * B + c * A @ B 
    if G.size(0) > G.size(1):
        X = X.T 
    return X.to(G.dtype)
```

### Provenance

Many of the choices made to generate this optimizer were obtained experimentally by our pursuit of [CIFAR-10 speedrunning](https://github.com/KellerJordan/cifar10-airbench).
In particular, we experimentally obtained the following practices:
* Using Nesterov momentum inside the update, with orthogonalization applied after momentum.
* Using a specifically quintic Newton-Schulz iteration as the method of orthogonalization.
* Using non-convergent coefficients for the quintic polynomial in order to maximize slope at zero, and thereby minimize the number of necessary Newton-Schulz iterations.
* Running the Newton-Schulz iteration in bfloat16 (whereas Shampoo implementations often compute the preconditioners in fp32 or fp64).

Our use of a Newton-Schulz iteration for orthogonalization traces to [Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325),
who suggested it as a way to compute Shampoo [5, 6] preconditioners, and theoretically explored Shampoo without preconditioner accumulation.
In particular, Jeremy Bernstein @jxbz sent us the draft, which caused us to experiment with various Newton-Schulz iterations as the
orthogonalization method for this optimizer.
If we had used SVD instead of a Newton-Schulz iteration, this optimizer would have been too slow to be useful.
Bernstein & Newhouse also pointed out that Shampoo without preconditioner accumulation is equivalent to steepest descent in the spectral norm,
and therefore Shampoo can be thought of as a way to smooth out spectral steepest descent.
The proposed optimizer can be thought of as a second way of smoothing spectral steepest descent, with a different set of memory and runtime tradeoffs
compared to Shampoo.

---

## Other general differences between this codebase and NanoGPT

To simplify the code, some features have been removed, including text generation.
And to obtain a training speed improvement, we have diverged from being a strict reproduction of the GPT-2 paper.

The speedup is due to the following changes:
- Increased learning rate by 3x
- Switched to trapezoidal learning rate schedule following [7]
- Switched to rotary embeddings
- Removed the special initialization for linear layers before residuals. Instead, just scale down the output of the attention block by a fixed scalar.
- Removed all affine scale and bias parameters from the architecture, and switched to RMSNorm (actually this causes a slight slowdown, and I just did it to reduce code complexity)
- Switched from AdamW to new optimizer

---

## References

1. [Penedo, Guilherme, et al. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)
2. Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics, 2008. Equation 5.22.
3. Günther Schulz. Iterative Berechnung der reziproken Matrix. Z. Angew. Math. Mech., 13:57–59, 1933.
4. [Jeremy Bernstein and Laker Newhouse. "Old Optimizer, New Norm: An Anthology." arxiv preprint arXiv:2409.20325 (2024).](https://arxiv.org/abs/2409.20325)
5. [Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned stochastic tensor optimization." International Conference on Machine Learning. PMLR, 2018.](https://arxiv.org/abs/1802.09568)
6. [Anil, Rohan, et al. "Scalable second order optimization for deep learning." arXiv preprint arXiv:2002.09018 (2020).](https://arxiv.org/abs/2002.09018)
7. [Hägele, Alexander, et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." arXiv preprint arXiv:2405.18392 (2024).](https://arxiv.org/abs/2405.18392)

![dofa](img/dofa.jpg)

### Citation

```
@software{moddednanogpt2024,
    author={Jordan, Keller},
    title={Modded-NanoGPT},
    url={https://github.com/KellerJordan/Modded-NanoGPT},
    version={0.1.0},
    year = {2024}
}
```

