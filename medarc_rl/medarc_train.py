from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer import Option

from medarc_rl.utils import TYPER_PASSTHROUGH_CONTEXT, _load_settings_from_toml, _write_toml, extra_config_args


app = typer.Typer(
    add_completion=False,
    help=(
        "Run PRIME-RL SFT/RL training on local GPU(s). "
        "Pass PRIME-RL config overrides as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)


def _gpu_ids(n: int) -> str:
    return ",".join(str(i) for i in range(n))


def _enable_sft_resume(config, *, enabled: bool) -> None:
    if not enabled:
        return
    if config.ckpt is None:
        from prime_rl.configs.trainer import CheckpointConfig as TrainerCheckpointConfig

        config.ckpt = TrainerCheckpointConfig()
    config.ckpt.resume_step = -1


def _enable_rl_resume(config, *, enabled: bool) -> None:
    if not enabled:
        return
    if config.ckpt is None:
        from prime_rl.configs.rl import SharedCheckpointConfig

        config.ckpt = SharedCheckpointConfig()
    config.ckpt.resume_step = -1


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Run PRIME-RL SFT locally. PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def sft(
    ctx: typer.Context,
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write resolved configs and checkpoints.")],
    config: Annotated[list[Path] | None, Option("--config", "--config-toml", help="One or more PRIME-RL SFT trainer TOMLs. Repeat `--config` to layer files with later files overriding earlier ones.")] = None,
    gpus: Annotated[int, Option("--gpus", min=1, max=8, help="Number of GPUs for SFT.")] = 1,
    resume: Annotated[bool, Option("--resume/--no-resume", help="Resume from the latest checkpoint (sets ckpt.resume_step=-1).")] = False,
) -> None:  # fmt: skip
    from prime_rl.configs.sft import SFTConfig

    config_tomls = list(config or [])
    if not config_tomls:
        raise typer.BadParameter("Missing config path. Pass one or more --config values.", param_hint="--config")
    output_dir = output_dir.expanduser().resolve()
    config = _load_settings_from_toml(
        SFTConfig,
        [config_toml.expanduser().resolve() for config_toml in config_tomls],
        output_dir=output_dir,
        extra_cli_args=extra_config_args(ctx, positional_count=0),
    )
    _enable_sft_resume(config, enabled=resume)

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config_dir / "trainer.toml"
    _write_toml(resolved_path, config.model_dump(exclude_none=True, mode="json"))

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": _gpu_ids(gpus)}

    if gpus == 1:
        cmd = ["sft", "@", str(resolved_path)]
    else:
        cmd = [
            "torchrun",
            "--local-ranks-filter=0",
            f"--nproc-per-node={gpus}",
            "-m",
            "prime_rl.trainer.sft.train",
            "@",
            str(resolved_path),
        ]

    typer.echo(f"Starting SFT on {gpus} GPU(s): {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(code=result.returncode)


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Run PRIME-RL RL locally. "
        "Use medarc GPU flags for placement/splitting. "
        "PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def rl(
    ctx: typer.Context,
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write resolved configs and checkpoints.")],
    config: Annotated[list[Path] | None, Option("--config", "--config-toml", help="One or more PRIME-RL RL TOMLs. Repeat `--config` to layer files with later files overriding earlier ones.")] = None,
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs for training.")] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs for inference.")] = 1,
    single_gpu: Annotated[bool, Option("--single-gpu", help="Share a single GPU between trainer and inference.")] = False,
    resume: Annotated[bool, Option("--resume/--no-resume", help="Resume from the latest checkpoint (sets ckpt.resume_step=-1).")] = False,
) -> None:  # fmt: skip
    from prime_rl.configs.rl import RLConfig

    from medarc_rl.launchers.rl_local import rl_local

    config_tomls = list(config or [])
    if not config_tomls:
        raise typer.BadParameter("Missing config path. Pass one or more --config values.", param_hint="--config")
    output_dir = output_dir.expanduser().resolve()
    train_gpus = 1 if single_gpu else train_gpus
    infer_gpus = 1 if single_gpu else infer_gpus
    gpus = 1 if single_gpu else (train_gpus + infer_gpus)

    if not single_gpu and gpus < 2:
        raise typer.BadParameter(
            f"Total GPUs must be at least 2, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )
    if gpus > 8:
        raise typer.BadParameter(
            f"Total GPUs must be at most 8, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )

    try:
        config = _load_settings_from_toml(
            RLConfig,
            [config_toml.expanduser().resolve() for config_toml in config_tomls],
            extra_cli_args=extra_config_args(ctx, positional_count=0),
            output_dir=output_dir,
            deployment={"type": "single_node", "num_train_gpus": train_gpus, "num_infer_gpus": infer_gpus},
        )
    except ValidationError as e:
        raise typer.BadParameter(
            f"RL config validation failed:\n{e}",
            param_hint="CONFIG_TOML/--train-gpus/--infer-gpus",
        ) from e
    _enable_rl_resume(config, enabled=resume)

    if single_gpu and getattr(config.trainer.weight_broadcast, "type", None) == "nccl":
        raise typer.BadParameter(
            "--single-gpu does not support NCCL weight broadcast. Use filesystem broadcast or 2+ GPUs.",
            param_hint="CONFIG_TOML/--single-gpu",
        )
    if single_gpu and config.inference is not None and config.inference.gpu_memory_utilization >= 0.9:
        typer.echo(
            "Warning: --single-gpu with inference.gpu_memory_utilization >= 0.9 may OOM. "
            "Try 0.7-0.8 for shared trainer+vLLM.",
            err=True,
        )

    # Set env vars for rl_local
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids(gpus)
    os.environ["MEDARC_SINGLE_GPU"] = "1" if single_gpu else "0"

    typer.echo(f"Starting RL on {gpus} GPU(s) (single_gpu={single_gpu})")
    rl_local(config)


if __name__ == "__main__":
    app()
