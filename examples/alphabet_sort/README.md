# Alphabet Sort

> Adapted from the [PRIME-RL alphabet_sort example](../../prime-rl/examples/alphabet_sort/) to use `medarc_slurm` for single-node SLURM submission.

This trains `Qwen3-4B-Instruct-2507` to sort names alphabetically using LoRA. The base model already understands the conversation format, so no SFT warmup is needed — we proceed directly to multi-turn RL against the [`primeintellect/alphabet-sort`](https://app.primeintellect.ai/dashboard/environments/primeintellect/alphabet-sort) environment.

> Note: This example uses 8 GPUs on a single node: 2 for training (FSDP) and 6 for inference (data parallel).

> Note: `medarc_train` and `medarc_slurm` accept arbitrary PRIME-RL config overrides as CLI flags. In these examples, we use that passthrough to set `wandb.project` and `wandb.name`.

## Setup

Install the environment:

```bash
prime env install primeintellect/alphabet-sort
```

Verify it's installed:

```bash
uv run python -c "import alphabet_sort"
```

## RL (8 GPUs: 2 train + 6 inference)

Submit an 8-GPU RL job:

```bash
medarc_slurm rl --config examples/alphabet_sort/rl.toml \
    --output-dir output/examples/alphabet-sort \
    --train-gpus 2 \
    --infer-gpus 6 \
    --auto-auth \
    --wandb.project alphabet-sort --wandb.name alphabet-sort-4b-example
```

Or preview without submitting:

```bash
medarc_slurm rl --config examples/alphabet_sort/rl.toml \
    --output-dir output/examples/alphabet-sort \
    --train-gpus 2 \
    --infer-gpus 6 \
    --auto-auth \
    --dry-run \
    --wandb.project alphabet-sort --wandb.name alphabet-sort-4b-example
```

The base model gets ~0.26 average reward. After LoRA RL, expect ~0.8.
