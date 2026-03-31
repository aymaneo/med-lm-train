# med-lm-train

## Setup + Installation

1. Clone the repository

```bash
git clone --recurse-submodules --shallow-submodules --depth 50 https://github.com/MedARC-AI/med-lm-train.git
cd med-lm-train
```

Or if you already have the repo cloned without submodules:

```bash
git submodule update --init --recursive --depth 50
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install dependencies

```bash
uv sync
```

To install the bundled PRIME-RL environment packages used by some examples:

```bash
uv sync --extra envs
```

For flash attention support:

```bash
uv sync --extra fa2   # flash-attn 2
uv sync --extra fa3   # flash-attn 2 + 3 (use for H100s)
uv sync --extra fa4   # flash-attn 2, 3, & 4 (use for B200s)
```

Legacy extra names `flash-attn-2`, `flash-attn-3`, and `flash-attn-4` remain supported for backward compatibility.

And to install both env and flash attention extras:

```bash
uv sync --extra envs --extra fa3
```

## medarc_slurm

`medarc_slurm` is a CLI tool that generates and submits single-node SLURM jobs for [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) SFT and RL training. It is based on PRIME-RL's built-in `rl_slurm` and `sft_slurm` commands but adapted for shared-node environments where jobs don't neccesarily have exclusive access to the machine.

```bash
# SFT: single torchrun job
medarc_slurm sft --config config.toml --output-dir runs/my-sft --gpus 2

# RL: splits GPUs between vLLM inference and training
medarc_slurm rl --config config.toml --output-dir runs/my-rl --train-gpus 1 --infer-gpus 2

# RL: share a single GPU between inference and training
medarc_slurm rl --config config.toml --output-dir runs/my-rl --single-gpu

# SFT: low-priority queue + email notifications + resume from latest checkpoint
medarc_slurm sft --config config.toml \
  --output-dir runs/my-sft \
  --gpus 2 \
  --priority low \
  --mail all \
  --mail-user email@domain.com \
  --slurm-resume

# Validate an RL submission (including dependency syntax) without creating a job
medarc_slurm rl --config config.toml \
  --output-dir runs/my-rl \
  --train-gpus 1 \
  --infer-gpus 2 \
  --dependency afterok:123456 \
  --test-only
```

Generated artifacts are written to `--output-dir`:
- `sft.sh` or `rl.sh` — the SLURM batch script
- `configs/` — resolved TOML subconfigs passed to each component

You can pass PRIME-RL config overrides directly as extra flags (for example `--wandb.project my-proj --wandb.name my-run`). You may also insert `--` before passthrough overrides for readability, but it is optional. To layer multiple PRIME-RL configs, repeat `--config` with later files overriding earlier ones.

`medarc_slurm` now defaults `--account` to `training`. You can override it with `--account <name>`.
Email mode is `--mail all` or `--mail begin_end` (with `--mail-user`).
Use `--dependency "<expr>"` to pass SLURM dependencies and `--test-only` to run `sbatch` validation without submitting.

Run `medarc_slurm sft --help` or `medarc_slurm rl --help` for more details on available options.

## Examples

Each example has its own README with setup instructions, SFT/RL commands, and eval steps:

| Example | GPUs | Description |
|---------|------|-------------|
| [reverse_text](examples/reverse_text/) | 1 (shared) | Single-GPU SFT + RL on a toy text reversal task |
| [hendrycks_sanity](examples/hendrycks_sanity/) | 4 | Multi-GPU RL on Hendrycks MATH (sanity subset) |
| [alphabet_sort](examples/alphabet_sort/) | 8 | Full-node RL on alphabet sorting |

All examples use `medarc_slurm` to generate and submit single-node SLURM jobs. Start with [reverse_text](examples/reverse_text/) to verify your setup.
Examples that rely on PRIME-RL environment packages require installing the `envs` extra first: `uv sync --extra envs`.
