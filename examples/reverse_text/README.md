# Reverse Text

> Adapted from the [PRIME-RL reverse text example](../../prime-rl/examples/reverse_text/).

We demonstrate how to train `Qwen3-0.6B` to reverse a small chunk of text. We use a SFT warmup to learn the skill of text reversal on longer documents and then a quick RL run to reverse smaller chunks of text in the [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment.

> Info: The configs in this example are tuned for H100 GPUs. If you're on consumer GPUs, you may need to lower `micro_batch_size` in `sft.toml` and/or `seq_len` in the RL config you use (`rl_multi.toml`, `rl_single.toml`, or `rl_slurm.toml`). See **Batching Options** at the end for token-based alternatives and microbatching notes.

> Note: `medarc_train` and `medarc_slurm` accept arbitrary PRIME-RL config overrides as CLI flags. In these examples, we use that passthrough to set `wandb.project` and `wandb.name`.

## Setup

Install the bundled PRIME-RL environment packages (assuming you want flash attention 3 for Ampere, Hopper, and Lovelace GPUs):

```bash
uv sync --extra envs --extra fa3
```
Verify it's installed:

```bash
python -c "import reverse_text"
```

Let's check how well `Qwen3-0.6B` does out-of-the-box on the `reverse-text` environment:

```bash
inference --model.name Qwen/Qwen3-0.6B
```

```bash
vf-eval reverse-text -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

The model struggles with this task — expect an **average reward of ~0.05** across the 20x3 rollouts. Let's do some training!

## SFT

We fine-tune [`PrimeIntellect/Qwen3-0.6B`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B) (a clone of `Qwen/Qwen3-0.6B` with a chat template suitable for multi-turn RL) on [`willcb/R1-reverse-wikipedia-paragraphs-v1-1000`](https://huggingface.co/datasets/willcb/R1-reverse-wikipedia-paragraphs-v1-1000) which contains 1K examples of reversals of small paragraphs.

### Local

To train on a single GPU with `medarc_train`:

```bash
medarc_train sft --config examples/reverse_text/sft.toml \
  --output-dir outputs/examples/reverse-sft \
  --wandb.project reverse-text --wandb.name reverse-text-sft
```

Or directly with PrimeRL's `sft` entrypoint:

```bash
sft @ examples/reverse_text/sft.toml \
  --output-dir outputs/examples/reverse-sft
```

To train on multiple GPUs with `medarc_train`:

```bash
medarc_train sft --config examples/reverse_text/sft.toml \
  --output-dir outputs/examples/reverse-sft \
  --gpus 2 \
  --wandb.project reverse-text --wandb.name reverse-text-sft
```

Or with PrimeRL's `sft` entrypoint:

```bash
sft @ examples/reverse_text/sft.toml \
  --output-dir outputs/examples/reverse-sft \
  --deployment.num_gpus 2
```

This writes a checkpoint to `outputs/examples/reverse-sft/weights/step_100`. Optionally upload it to HF to use as the base model for RL:

```bash
hf upload <user>/Qwen3-0.6B-Reverse-Text-SFT outputs/weights/step_100
```

The RL config uses the published [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT) by default — to use your own SFT checkpoint instead, override `--model.name`.

### SLURM

Submit a 1-GPU SFT job via `medarc_slurm`:

```bash
medarc_slurm sft --config examples/reverse_text/sft.toml \
    --output-dir outputs/examples/reverse-sft \
    --gpus 1 \
    --auto-auth \
    --wandb.project reverse-text --wandb.name reverse-text-sft
```

Or preview without submitting:

```bash
medarc_slurm sft --config examples/reverse_text/sft.toml \
    --output-dir outputs/examples/reverse-sft \
    --gpus 1 \
    --auto-auth \
    --dry-run \
    --wandb.project reverse-text --wandb.name reverse-text-sft
```

## RL

For RL we do 20 steps with sequence length 128. All three RL configs in this example use the same orchestrator batching as PRIME-RL's reverse-text example (`batch_size = 128`, `rollouts_per_example = 16`, i.e. 8 examples x 16 rollouts). The single-GPU configs differ in deployment/runtime settings (for example vLLM memory utilization), not orchestrator batching.

### Local (single GPU)

Run RL locally on a single shared GPU (assumes a 24GB GPU like a 3090 or 4090):

```bash
medarc_train rl --config examples/reverse_text/rl_single.toml \
  --output-dir outputs/examples/reverse-rl \
  --single-gpu \
  --wandb.project reverse-text --wandb.name reverse-text-rl
```

This writes a checkpoint to `outputs/examples/reverse-rl/weights/step_20`.

### Local (two GPUs)

If you have 2 GPUs, you can dedicate one to inference and one to training. This avoids the shared-GPU memory pressure of `--single-gpu` and uses PrimeRL's default `gpu_memory_utilization` (0.9). The `rl_multi.toml` config omits the lowered `gpu_memory_utilization` from `rl_single.toml`.

With `medarc_train`:

```bash
medarc_train rl --config examples/reverse_text/rl_multi.toml \
  --output-dir outputs/examples/reverse-rl \
  --wandb.project reverse-text --wandb.name reverse-text-rl
```

Or directly with PrimeRL's `rl` entrypoint (assumes both GPUs are visible):

```bash
rl @ examples/reverse_text/rl_multi.toml
```

### SLURM (single GPU)

This example shares a single GPU between the trainer and vLLM inference server. The config lowers vLLM `gpu_memory_utilization` so the trainer has headroom — if you still see OOMs, reduce it further.

```bash
medarc_slurm rl --config examples/reverse_text/rl_slurm.toml \
    --output-dir outputs/examples/reverse-rl \
    --single-gpu \
    --auto-auth \
    --wandb.project reverse-text --wandb.name reverse-text-rl
```

Or preview without submitting:

```bash
medarc_slurm rl --config examples/reverse_text/rl_slurm.toml \
    --output-dir outputs/examples/reverse-rl \
    --single-gpu \
    --auto-auth \
    --dry-run \
    --wandb.project reverse-text --wandb.name reverse-text-rl
```

This writes a checkpoint to `outputs/examples/reverse-rl/weights/step_20`.

For larger multi-GPU RL examples, see [hendrycks_sanity](../hendrycks_sanity/) (4 GPUs) and [alphabet_sort](../alphabet_sort/) (8 GPUs).

## Evals

To evaluate the final RL checkpoint, start an inference server and run `vf-eval`:

```bash
uv run inference --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
```

```bash
uv run vf-eval reverse-text \
    -m PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
    -b http://localhost:8000/v1 \
    -n 20 --max-tokens 1024
```

The base model gets ~0.05 average reward. After SFT + RL, expect ~0.8.

## Batching Options

The example configs above intentionally match PRIME-RL's reverse-text orchestrator batching:

```toml
[orchestrator]
batch_size = 128
rollouts_per_example = 16
```

If you want to switch to the newer token-based batching (`token_batch_size`) for more stable token volume per training step when rollout lengths vary, PRIME-RL supports:

- `batch_size`: rollout-count batching (fixed accepted rollouts per step)
- `token_batch_size`: token-count batching (fixed accepted tokens per step; rollout count varies)

When using `token_batch_size`, you must also set `max_inflight_rollouts`. A practical conversion from a rollout-based config is:

```toml
# rollout mode
batch_size = 128
rollouts_per_example = 16

# approximate token-mode equivalent
token_batch_size = batch_size * avg_accepted_rollout_tokens
max_inflight_rollouts = batch_size                    # if no oversampling_factor
# max_inflight_rollouts = int(batch_size * oversampling_factor)  # if converting an older rollout config
```

Example: if accepted rollouts average ~256 tokens, `batch_size = 128` is roughly comparable to:

```toml
token_batch_size = 32768
max_inflight_rollouts = 128
```

This is only approximate: token-based mode stabilizes token volume per step, while rollout-based mode stabilizes rollout count per step.

### Microbatching and Memory Tuning

SFT and RL expose memory/throughput tradeoffs differently:

- SFT has an explicit `micro_batch_size` (`sft.toml`), and PRIME-RL accumulates gradients across micro-steps to reach the configured global `batch_size`.
- If SFT still OOMs after lowering `micro_batch_size`, lower `seq_len` next (this reduces context length and can truncate more tokens).

RL does not use the same explicit `micro_batch_size` knob:

- RL trainer steps are built from internally packed micro-batches (up to the configured training `seq_len`).
- In practice, the main RL peak-memory knob is `seq_len` (top-level `seq_len` or `trainer.model.seq_len` in these configs).
- Lowering RL `batch_size` (or `token_batch_size` if you switch to token-based batching) changes global rollout/tokens per optimizer step and throughput/update size..
