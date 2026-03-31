# AGENTS.md

This file provides guidance to Codex, Claude Code, and other coding agents when working with code in this repository.

## Overview

med-lm-train provides CLI tools (`medarc_slurm`, `medarc_train`) for single-node SLURM submission and local SFT/RL training workflows built on PRIME-RL.

`prime-rl/` is a pinned external git submodule — do not modify.

## Commands

```bash
uv sync                                        # Install deps
uv sync --extra fa2                            # With Flash Attention v2
uv sync --extra fa3                            # With FA2 + FA3 (H100s)
uv sync --extra fa4                            # With FA2 + FA3 + FA4 (B200s)

pytest tests/                                   # Run tests
pytest tests/test_medarc_slurm.py::test_name    # Single test
ruff check medarc_rl tests                      # Lint
ruff format medarc_rl tests                     # Format
```

Legacy extras `flash-attn-2`, `flash-attn-3`, and `flash-attn-4` remain supported for backward compatibility.

Testing scope:
- `pyproject.toml` sets `pytest` `testpaths = ["tests"]`, so default collection is scoped correctly.
- Do not run `prime-rl/tests/` by default.
- Avoid `pytest .` (or other explicit repo-root paths), which can widen collection and include `prime-rl/tests/`.
- Only run this repo's tests under `tests/` unless the user explicitly asks to run PRIME-RL tests.

## Architecture

### CLI (`medarc_rl/medarc_slurm.py`)

Typer-based CLI with two commands (`sft` and `rl`). Each command:
1. Loads and resolves TOML configs using PRIME-RL's Pydantic config classes
2. Renders a Jinja2 SLURM template (`medarc_rl/slurm_templates/`)
3. Writes the script + resolved configs to the output directory
4. Submits via `sbatch` (or prints in `--dry-run` mode)

### Local Training CLI (`medarc_rl/medarc_train.py`)

Typer-based local runner for PRIME-RL SFT/RL. It resolves configs the same way as `medarc_slurm`, writes resolved configs, and launches local training (`sft` / `torchrun` / `rl_local`).

### RL Launcher (`medarc_rl/launchers/rl_local.py`)

Modified version of PRIME-RL's `rl_local()` for shared-node environments. Handles GPU isolation via `CUDA_VISIBLE_DEVICES`, per-process cache separation, and coordinated multi-process lifecycle with thread-based monitoring.

### Config System

TOML-based configs with inheritance via PRIME-RL's `toml_files` mechanism. Example configs in `examples/`. Resolved configs are written to the output directory for reproducibility.

Both `medarc_slurm` and `medarc_train` support PRIME-RL-style nested CLI overrides (for example `-- --wandb.name run1`). Wrapper-owned fields (especially GPU split / deployment and `output_dir`) take precedence over passthrough overrides.

Shared config/TOML helper functions live in `medarc_rl/utils.py`; do not import underscore helpers from `medarc_rl.medarc_slurm` into other modules.

## Constraints

- RL jobs: total GPUs (train + infer) must be 2-8, or use `--single-gpu` for 1
- NCCL broadcast is only compatible with `async_level=1`
- Ruff line length: 120
