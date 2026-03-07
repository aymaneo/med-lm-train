from __future__ import annotations

import os
import shlex
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from typer import Option

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

from medarc_rl.utils import (
    TYPER_PASSTHROUGH_CONTEXT,
    _load_settings_from_toml,
    _write_toml,
    extra_config_args,
    maybe_autoset_auth_env,
)


app = typer.Typer(
    add_completion=False,
    help=(
        "Generate single-node SLURM jobs for PRIME-RL SFT/RL. "
        "Pass PRIME-RL config overrides as extra flags  e.g. "
        "`--wandb.project my-proj --wandb.name my-run`."
    ),
)

TEMPLATE_DIR = Path(__file__).parent / "slurm_templates"
PANEL_INPUTS = "Inputs"
PANEL_COMPUTE = "Compute"
PANEL_SUBMISSION = "Submission"
PANEL_NOTIFICATIONS = "Notifications"
PANEL_RUNTIME = "Runtime Environment"


class QoS(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    TOP = "top"


class MailSetting(StrEnum):
    ALL = "all"
    BEGIN_END = "begin_end"


class Account(StrEnum):
    TRAINING = "training"
    SOPHONT = "sophont"


def _resolve_path(path: Path | None, fallback: Path) -> Path:
    return (path or fallback).expanduser().resolve()
def _default_hf_cache_dir(project_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        return Path(env_hf_home).expanduser().resolve()
    return (project_dir / ".hf_cache").resolve()


def _ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(parents=True, exist_ok=True)


def _enable_sft_resume(config: SFTConfig, *, enabled: bool) -> None:
    if not enabled:
        return
    if config.ckpt is None:
        from prime_rl.configs.trainer import CheckpointConfig as TrainerCheckpointConfig

        config.ckpt = TrainerCheckpointConfig()
    config.ckpt.resume_step = -1


def _enable_rl_resume(config: RLConfig, *, enabled: bool) -> None:
    if not enabled:
        return
    if config.ckpt is None:
        from prime_rl.configs.rl import SharedCheckpointConfig

        config.ckpt = SharedCheckpointConfig()
    config.ckpt.resume_step = -1


def _render_template(template_name: str, **context: Any) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template(template_name).render(**context)


def _write_script(output_dir: Path, name: str, text: str) -> Path:
    path = output_dir / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _submit_or_print(
    script_path: Path,
    *,
    dry_run: bool,
    account: str | Account | None = None,
    dependency: str | None = None,
    test_only: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    if account is None:
        account = Account.TRAINING
    if isinstance(account, Account):
        account = account.value
    if dependency is not None:
        dependency = dependency.strip()
        if not dependency:
            raise typer.BadParameter("--dependency must not be empty.", param_hint="--dependency")

    sbatch_cmd = ["sbatch"]
    if account:
        sbatch_cmd.extend(["--account", account])
    if dependency:
        sbatch_cmd.extend(["--dependency", dependency])
    if test_only:
        sbatch_cmd.append("--test-only")
    sbatch_cmd.append(str(script_path))

    cmd = shlex.join(sbatch_cmd)
    if dry_run:
        typer.echo(cmd)
        return

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        typer.echo(result.stderr.strip() or "sbatch failed", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.stdout.strip())


def _load_sft_config(
    config_tomls: list[Path],
    output_dir: Path | None,
    *,
    extra_cli_args: list[str] | None = None,
) -> SFTConfig:
    kwargs: dict[str, Any] = {"extra_cli_args": extra_cli_args}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return _load_settings_from_toml(SFTConfig, config_tomls, **kwargs)


def _load_rl_config(
    config_tomls: list[Path],
    output_dir: Path | None,
    *,
    train_gpus: int,
    infer_gpus: int,
    extra_cli_args: list[str] | None = None,
) -> RLConfig:
    kwargs: dict[str, Any] = {
        "extra_cli_args": extra_cli_args,
        "deployment": {"type": "single_node", "num_train_gpus": train_gpus, "num_infer_gpus": infer_gpus},
    }
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return _load_settings_from_toml(RLConfig, config_tomls, **kwargs)


def _write_sft_outputs(
    config: SFTConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    gpus: int,
    cpus_per_gpu: int,
    priority: QoS | None,
    mail_type: str | None,
    mail_user: str | None,
    slurm_resume: bool,
) -> Path:
    config_dir = output_dir / "configs"
    _write_toml(config_dir / "trainer.toml", config.model_dump(exclude_none=True, mode="json"))
    script = _render_template(
        "one_node_sft.j2",
        job_name=job_name,
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        project_dir=str(project_dir),
        hf_cache_dir=str(hf_cache_dir),
        hf_hub_offline=hf_hub_offline,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        qos=priority.value if priority is not None else None,
        mail_type=mail_type,
        mail_user=mail_user,
        slurm_resume=slurm_resume,
    )
    return _write_script(output_dir, "sft.sh", script)


def _write_rl_outputs(
    config: RLConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    gpus: int,
    single_gpu: bool,
    cpus_per_gpu: int,
    priority: QoS | None,
    mail_type: str | None,
    mail_user: str | None,
    slurm_resume: bool,
) -> Path:
    if config.inference is None:
        raise typer.BadParameter("RL requires an [inference] config.", param_hint="CONFIG_TOML")

    config_dir = output_dir / "configs"
    _write_toml(config_dir / "rl.toml", config.model_dump(exclude_none=True, mode="json"))

    script = _render_template(
        "one_node_rl.j2",
        job_name=job_name,
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        project_dir=str(project_dir),
        hf_cache_dir=str(hf_cache_dir),
        hf_hub_offline=hf_hub_offline,
        gpus=gpus,
        single_gpu=single_gpu,
        cpus_per_gpu=cpus_per_gpu,
        qos=priority.value if priority is not None else None,
        mail_type=mail_type,
        mail_user=mail_user,
        slurm_resume=slurm_resume,
    )
    return _write_script(output_dir, "rl.sh", script)


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Generate/submit an SFT SLURM job. PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def sft(
    ctx: typer.Context,
    gpus: Annotated[int, Option("--gpus", min=1, max=8, help="Number of GPUs for SFT on this single node (sets SLURM gres and torchrun nproc-per-node).", rich_help_panel=PANEL_COMPUTE)],
    output_dir: Annotated[Path | None, Option("--output-dir", file_okay=False, dir_okay=True, help="Optional output directory for generated artifacts (configs/ and sft.sh). Overrides output_dir from TOML when set.", rich_help_panel=PANEL_INPUTS)] = None,
    config: Annotated[list[Path] | None, Option("--config", "--config-toml", help="One or more PRIME-RL SFT trainer TOMLs. Repeat `--config` to layer files with later files overriding earlier ones.", rich_help_panel=PANEL_INPUTS)] = None,
    cpus_per_gpu: Annotated[int, Option("--cpus-per-gpu", min=1, max=32, help="Number of CPUs to allocate per GPU (sets SLURM --cpus-per-gpu).", rich_help_panel=PANEL_COMPUTE)] = 16,
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-sft'.", rich_help_panel=PANEL_SUBMISSION)] = None,
    account: Annotated[Account, Option("--account", help="SLURM account to pass to sbatch.", rich_help_panel=PANEL_SUBMISSION)] = Account.TRAINING,
    priority: Annotated[QoS | None, Option("--priority", help="SLURM job priority (sets the SLURM QoS value). Only project leads can set high.", rich_help_panel=PANEL_SUBMISSION)] = None,
    dependency: Annotated[str | None, Option("--dependency", help="SLURM dependency expression for sbatch (e.g. 'afterok:12345' or 'singleton').", rich_help_panel=PANEL_SUBMISSION)] = None,
    test_only: Annotated[bool, Option("--test-only", help="Pass --test-only to sbatch to validate without submitting a job.", rich_help_panel=PANEL_SUBMISSION)] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.", rich_help_panel=PANEL_SUBMISSION)] = False,
    mail: Annotated[MailSetting | None, Option("--mail", help="SLURM email setting: 'all' or 'begin_end'.", rich_help_panel=PANEL_NOTIFICATIONS)] = None,
    mail_user: Annotated[str | None, Option("--mail-user", help="Email address for SLURM notifications.", rich_help_panel=PANEL_NOTIFICATIONS)] = None,
    slurm_resume: Annotated[bool, Option("--slurm-resume/--no-slurm-resume", help="Enable SLURM requeue and resume from the latest checkpoint (sets ckpt.resume_step=-1).", rich_help_panel=PANEL_SUBMISSION)] = False,
    source_dir: Annotated[Path | None, Option("--source-dir", file_okay=False, dir_okay=True, help="Source directory used by the script to source .env and activate .venv (defaults to current working directory).", rich_help_panel=PANEL_RUNTIME)] = None,
    hf_cache_dir: Annotated[Path, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job).", rich_help_panel=PANEL_RUNTIME)] = "/data/medlm_cache/.hf_cache",
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.", rich_help_panel=PANEL_RUNTIME)] = False,
    auto_auth: Annotated[bool, Option("--auto-auth/--no-auto-auth", help="Try to load HF_TOKEN from local CLI credentials and inject it into the sbatch submission environment.", rich_help_panel=PANEL_RUNTIME)] = False,
) -> None:  # fmt: skip
    config_tomls = list(config or [])
    if not config_tomls:
        raise typer.BadParameter("Missing config path. Pass one or more --config values.", param_hint="--config")
    output_dir_override = output_dir.expanduser().resolve() if output_dir is not None else None
    project_dir = _resolve_path(source_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    if mail is None and mail_user is not None:
        mail = MailSetting.ALL
    if mail is not None and not mail_user:
        raise typer.BadParameter("--mail-user is required when --mail is set.", param_hint="--mail-user")
    mail_type = "begin,end" if mail == MailSetting.BEGIN_END else (mail.value if mail is not None else None)
    resolved_config_paths = [path.expanduser().resolve() for path in config_tomls]
    sft_config = _load_sft_config(
        resolved_config_paths,
        output_dir_override,
        extra_cli_args=extra_config_args(ctx, positional_count=0),
    )
    output_dir = output_dir_override or sft_config.output_dir.expanduser().resolve()
    sft_config.output_dir = output_dir
    job_name = job_name or f"{resolved_config_paths[-1].stem}-sft"
    _ensure_output_dirs(output_dir)
    _enable_sft_resume(sft_config, enabled=slurm_resume)
    script_path = _write_sft_outputs(
        sft_config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        priority=priority,
        mail_type=mail_type,
        mail_user=mail_user,
        slurm_resume=slurm_resume,
    )
    submit_env = os.environ.copy()
    for msg in maybe_autoset_auth_env(submit_env, enabled=auto_auth):
        typer.echo(msg, err=True)
    _submit_or_print(
        script_path,
        dry_run=dry_run,
        account=account,
        dependency=dependency,
        test_only=test_only,
        env=submit_env,
    )


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Generate/submit an RL SLURM job. "
        "Use medarc GPU flags for placement/splitting. "
        "PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def rl(
    ctx: typer.Context,
    output_dir: Annotated[Path | None, Option("--output-dir", file_okay=False, dir_okay=True, help="Optional output directory for generated artifacts (configs/ and rl.sh). Overrides output_dir from TOML when set.", rich_help_panel=PANEL_INPUTS)] = None,
    config: Annotated[list[Path] | None, Option("--config", "--config-toml", help="One or more PRIME-RL RL TOMLs. Repeat `--config` to layer files with later files overriding earlier ones.", rich_help_panel=PANEL_INPUTS)] = None,
    single_gpu: Annotated[bool, Option("--single-gpu", help="Run trainer and inference on the same single GPU (shared). Overrides --train-gpus/--infer-gpus to 1/1.", rich_help_panel=PANEL_COMPUTE)] = False,
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs reserved for trainer processes (1..4). Total GPUs is train + infer.", rich_help_panel=PANEL_COMPUTE)] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs reserved for local inference server (1..7). Total GPUs is train + infer.", rich_help_panel=PANEL_COMPUTE)] = 1,
    cpus_per_gpu: Annotated[int, Option("--cpus-per-gpu", min=1, max=32, help="Number of CPUs to allocate per GPU (sets SLURM --cpus-per-gpu).", rich_help_panel=PANEL_COMPUTE)] = 16,
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-rl'.", rich_help_panel=PANEL_SUBMISSION)] = None,
    account: Annotated[Account, Option("--account", help="SLURM account to pass to sbatch.", rich_help_panel=PANEL_SUBMISSION)] = Account.TRAINING,
    priority: Annotated[QoS | None, Option("--priority", help="SLURM job priority (sets the SLURM QoS value). Only project leads can set high.", rich_help_panel=PANEL_SUBMISSION)] = None,
    dependency: Annotated[str | None, Option("--dependency", help="SLURM dependency expression for sbatch (e.g. 'afterok:12345' or 'singleton').", rich_help_panel=PANEL_SUBMISSION)] = None,
    test_only: Annotated[bool, Option("--test-only", help="Pass --test-only to sbatch to validate without submitting a job.", rich_help_panel=PANEL_SUBMISSION)] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.", rich_help_panel=PANEL_SUBMISSION)] = False,
    mail: Annotated[MailSetting | None, Option("--mail", help="SLURM email setting: 'all' or 'begin_end'.", rich_help_panel=PANEL_NOTIFICATIONS)] = None,
    mail_user: Annotated[str | None, Option("--mail-user", help="Email address for SLURM notifications.", rich_help_panel=PANEL_NOTIFICATIONS)] = None,
    slurm_resume: Annotated[bool, Option("--slurm-resume/--no-slurm-resume", help="Enable SLURM requeue and resume from the latest checkpoint (sets ckpt.resume_step=-1).", rich_help_panel=PANEL_SUBMISSION)] = False,
    source_dir: Annotated[Path | None, Option("--source-dir", file_okay=False, dir_okay=True, help="Source directory used by the script to source .env and activate .venv (defaults to current working directory).", rich_help_panel=PANEL_RUNTIME)] = None,
    hf_cache_dir: Annotated[Path, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job).", rich_help_panel=PANEL_RUNTIME)] = "/data/medlm_cache/.hf_cache",
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.", rich_help_panel=PANEL_RUNTIME)] = False,
    auto_auth: Annotated[bool, Option("--auto-auth/--no-auto-auth", help="Try to load HF_TOKEN from local CLI credentials and inject it into the sbatch submission environment.", rich_help_panel=PANEL_RUNTIME)] = False,
) -> None:  # fmt: skip
    config_tomls = list(config or [])
    if not config_tomls:
        raise typer.BadParameter("Missing config path. Pass one or more --config values.", param_hint="--config")
    output_dir_override = output_dir.expanduser().resolve() if output_dir is not None else None
    project_dir = _resolve_path(source_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    resolved_config_paths = [path.expanduser().resolve() for path in config_tomls]
    job_name = job_name or f"{resolved_config_paths[-1].stem}-rl"
    train_gpus = 1 if single_gpu else train_gpus
    infer_gpus = 1 if single_gpu else infer_gpus
    gpus = 1 if single_gpu else (train_gpus + infer_gpus)

    if (not single_gpu and gpus < 2) or gpus > 8:
        raise typer.BadParameter(
            (
                f"Total GPUs must be between 2 and 8, got train_gpus ({train_gpus}) + "
                f"infer_gpus ({infer_gpus}) = {train_gpus + infer_gpus}."
            ),
            param_hint="--train-gpus/--infer-gpus",
        )

    if mail is None and mail_user is not None:
        mail = MailSetting.ALL
    if mail is not None and not mail_user:
        raise typer.BadParameter("--mail-user is required when --mail is set.", param_hint="--mail-user")
    mail_type = "begin,end" if mail == MailSetting.BEGIN_END else (mail.value if mail is not None else None)
    try:
        config = _load_rl_config(
            resolved_config_paths,
            output_dir_override,
            train_gpus=train_gpus,
            infer_gpus=infer_gpus,
            extra_cli_args=extra_config_args(ctx, positional_count=0),
        )
    except ValidationError as e:
        raise typer.BadParameter(
            f"RL config validation failed:\n{e}",
            param_hint="CONFIG_TOML/--train-gpus/--infer-gpus",
        ) from e
    _enable_rl_resume(config, enabled=slurm_resume)
    if single_gpu and getattr(config.trainer.weight_broadcast, "type", None) == "nccl":
        raise typer.BadParameter(
            "--single-gpu does not support NCCL weight broadcast. Use filesystem broadcast or 2+ GPUs.",
            param_hint="CONFIG_TOML/--single-gpu",
        )
    if single_gpu and config.inference is not None and config.inference.gpu_memory_utilization >= 0.9:
        typer.echo(
            (
                "Warning: --single-gpu with inference.gpu_memory_utilization >= 0.9 may OOM. "
                "PrimeRL's default is 0.9; try 0.7-0.8 for shared trainer+vLLM."
            ),
            err=True,
        )
    output_dir = output_dir_override or config.output_dir.expanduser().resolve()
    config.output_dir = output_dir
    _ensure_output_dirs(output_dir)
    script_path = _write_rl_outputs(
        config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        gpus=gpus,
        single_gpu=single_gpu,
        cpus_per_gpu=cpus_per_gpu,
        priority=priority,
        mail_type=mail_type,
        mail_user=mail_user,
        slurm_resume=slurm_resume,
    )
    submit_env = os.environ.copy()
    for msg in maybe_autoset_auth_env(submit_env, enabled=auto_auth):
        typer.echo(msg, err=True)
    _submit_or_print(
        script_path,
        dry_run=dry_run,
        account=account,
        dependency=dependency,
        test_only=test_only,
        env=submit_env,
    )


if __name__ == "__main__":
    app()
