from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock

from pydantic_config import cli
from typer.testing import CliRunner

from medarc_rl.medarc_slurm import app
from prime_rl.configs.sft import SFTConfig


runner = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _bash_n(script_path: Path) -> None:
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _load_sft_config(config_path: Path) -> SFTConfig:
    return cli(SFTConfig, args=["@", str(config_path)])


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _cli_args(config_paths: tuple[Path, ...], *args: str) -> list[str]:
    return [*[item for path in config_paths for item in ("--config", str(path))], *args]


def _config_option_args(config_paths: tuple[Path, ...], *args: str) -> list[str]:
    return [*[item for path in config_paths for item in ("--config", str(path))], *args]


def _build_sft_inherited_config(tmp_path: Path) -> tuple[Path, Path]:
    base = _write(
        tmp_path / "sft_base.toml",
        """
        [model]
        name = "Qwen/Qwen2.5-3B"
        seq_len = 256

        [data]
        type = "fake"
        batch_size = 2
        micro_batch_size = 1
        seq_len = 256
        """,
    )
    child = _write(
        tmp_path / "sft_child.toml",
        """
        max_steps = 2
        """,
    )
    return base, child


def _build_rl_inherited_config(
    tmp_path: Path, *, weight_broadcast_type: str = "nccl", cp: int = 2, tp: int = 1
) -> tuple[Path, Path]:
    base = _write(
        tmp_path / "rl_base.toml",
        f"""
        [trainer.model]
        cp = {cp}
        tp = {tp}

        [orchestrator]

        [inference.parallel]
        tp = 1
        dp = 3
        """,
    )
    child = _write(
        tmp_path / "rl_child.toml",
        f"""

        [inference]
        api_server_count = 4

        [weight_broadcast]
        type = "{weight_broadcast_type}"
        """,
    )
    return base, child


def test_sft_dry_run_generates_script_and_resolved_toml(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out"

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "2", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch --account training {output_dir / 'sft.sh'}" in result.output

    script_path = output_dir / "sft.sh"
    trainer_toml = output_dir / "configs" / "trainer.toml"
    assert script_path.exists()
    assert trainer_toml.exists()

    _bash_n(script_path)
    cfg = _load_sft_config(trainer_toml)
    assert cfg.output_dir == output_dir

    script = script_path.read_text(encoding="utf-8")
    assert "--standalone" not in script
    assert "--rdzv-endpoint=127.0.0.1:$RDZV_PORT" in script
    assert "--nproc-per-node" in script
    assert "pick_free_ports" in script
    assert "uv sync" not in script


def test_sft_boundary_and_hf_env_flags_are_rendered(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_boundary"
    hf_cache_dir = tmp_path / "hf_cache"

    result = runner.invoke(
        app,
        [
            "sft",
            *_cli_args(
                config_paths,
                "--output-dir",
                str(output_dir),
                "--gpus",
                "1",
                "--hf-cache-dir",
                str(hf_cache_dir),
                "--hf-hub-offline",
                "--dry-run",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-task=1" in script
    assert f'export HF_CACHE_DIR="{hf_cache_dir.resolve()}"' in script
    assert 'export HF_HOME="$HF_CACHE_DIR"' in script
    assert "export HF_HUB_OFFLINE=1" in script


def test_sft_renders_priority_mail_and_requeue_flags(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_slurm_flags"

    result = runner.invoke(
        app,
        [
            "sft",
            *_cli_args(
                config_paths,
                "--output-dir",
                str(output_dir),
                "--gpus",
                "1",
                "--priority",
                "low",
                "--mail",
                "all",
                "--mail-user",
                "email@domain.com",
                "--slurm-resume",
                "--dry-run",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    config = _read_toml(output_dir / "configs" / "trainer.toml")

    assert "#SBATCH --qos=low" in script
    assert "#SBATCH --mail-type=all" in script
    assert "#SBATCH --mail-user=email@domain.com" in script
    assert "#SBATCH --requeue" in script
    assert config["ckpt"]["resume_step"] == -1


def test_rl_defaults_split_to_one_and_one(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_missing_split"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-task=2" in script


def test_rl_rejects_total_gpu_count_above_eight(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path)
    output_dir = tmp_path / "rl_out_bad_total"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "4", "--infer-gpus", "5", "--dry-run")],
    )

    assert result.exit_code != 0
    assert "between 2" in result.output
    assert "and 8" in result.output


def test_rl_rejects_train_gpus_above_four(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_bad_train_max"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "5", "--infer-gpus", "1", "--dry-run")],
    )

    assert result.exit_code != 0
    assert "train-gpus" in result.output


def test_rl_rejects_infer_gpus_above_seven(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_bad_infer_max"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "1", "--infer-gpus", "8", "--dry-run")],
    )

    assert result.exit_code != 0
    assert "infer-gpus" in result.output


def test_rl_dry_run_generates_normalized_subconfigs_and_safe_script(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path)
    output_dir = tmp_path / "rl_out"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "4", "--infer-gpus", "2", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch --account training {output_dir / 'rl.sh'}" in result.output

    script_path = output_dir / "rl.sh"
    rl_toml = output_dir / "configs" / "rl.toml"

    for path in [script_path, rl_toml]:
        assert path.exists(), str(path)

    _bash_n(script_path)

    rl_cfg = _read_toml(rl_toml)

    assert rl_cfg["deployment"]["num_train_gpus"] == 4
    assert rl_cfg["deployment"]["num_infer_gpus"] == 2
    assert rl_cfg["orchestrator"]["num_train_workers"] == 2
    assert rl_cfg["inference"]["parallel"]["tp"] == 1
    assert rl_cfg["inference"]["parallel"]["dp"] == 2
    assert rl_cfg["trainer"]["weight_broadcast"]["type"] == "nccl"
    assert rl_cfg["inference"]["weight_broadcast"]["type"] == "nccl"

    script = script_path.read_text(encoding="utf-8")
    assert 'export MEDARC_SINGLE_GPU=0' in script
    assert 'python -m medarc_rl.launchers.rl_local @ "$CONFIG_DIR/rl.toml"' in script
    assert "pick_free_ports" not in script
    assert "--rdzv-endpoint" not in script
    assert "-m prime_rl.trainer.rl.train" not in script
    assert "orchestrator @" not in script
    assert "inference @" not in script
    assert "uv sync" not in script


def test_rl_dry_run_train_gpu_path_and_filesystem_broadcast(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, weight_broadcast_type="filesystem")
    output_dir = tmp_path / "rl_out_train_split_fs"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "4", "--infer-gpus", "2", "--dry-run")],
    )

    assert result.exit_code == 0, result.output

    rl_cfg = _read_toml(output_dir / "configs" / "rl.toml")
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")

    assert rl_cfg["trainer"]["weight_broadcast"]["type"] == "filesystem"
    assert "inference_world_size" not in rl_cfg["trainer"]["weight_broadcast"]
    assert rl_cfg["inference"]["parallel"]["tp"] == 1
    assert rl_cfg["inference"]["parallel"]["dp"] == 2
    assert "WEIGHT_BROADCAST_PORT" not in script
    assert "pick_free_ports" not in script
    assert 'python -m medarc_rl.launchers.rl_local @ "$CONFIG_DIR/rl.toml"' in script


def test_sft_accepts_primerl_style_overrides_and_wrapper_output_dir_wins(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_overrides"
    ignored_output_dir = tmp_path / "ignored_by_wrapper"
    override_config = _write(tmp_path / "sft_override.toml", f'output_dir = "{ignored_output_dir}"')

    result = runner.invoke(
        app,
        [
            "sft",
            *_cli_args(
                (*config_paths, override_config),
                "--output-dir",
                str(output_dir),
                "--gpus",
                "1",
                "--dry-run",
                "--max-steps",
                "7",
                "--model.seq-len",
                "1024",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = _read_toml(output_dir / "configs" / "trainer.toml")
    assert cfg["max_steps"] == 7
    assert cfg["model"]["seq_len"] == 1024
    assert cfg["output_dir"] == str(output_dir.resolve())


def test_rl_accepts_primerl_style_overrides_and_wrapper_gpu_split_wins(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_overrides"

    result = runner.invoke(
        app,
        [
            "rl",
            *_cli_args(
                config_paths,
                "--output-dir",
                str(output_dir),
                "--train-gpus",
                "2",
                "--infer-gpus",
                "1",
                "--dry-run",
                "--inference.gpu-memory-utilization",
                "0.33",
                "--deployment.num-train-gpus",
                "4",
                "--deployment.num-infer-gpus",
                "4",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    rl_cfg = _read_toml(output_dir / "configs" / "rl.toml")
    assert rl_cfg["inference"]["gpu_memory_utilization"] == 0.33
    assert rl_cfg["deployment"]["num_train_gpus"] == 2
    assert rl_cfg["deployment"]["num_infer_gpus"] == 1


def test_rl_single_gpu_dry_run(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, weight_broadcast_type="filesystem", cp=1, tp=1)
    output_dir = tmp_path / "rl_out_single_gpu"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--single-gpu", "--dry-run")],
    )

    assert result.exit_code == 0, result.output

    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-task=1" in script
    assert "export MEDARC_SINGLE_GPU=1" in script
    assert 'python -m medarc_rl.launchers.rl_local @ "$CONFIG_DIR/rl.toml"' in script

    rl_cfg = _read_toml(output_dir / "configs" / "rl.toml")
    assert rl_cfg["deployment"]["num_train_gpus"] == 1
    assert rl_cfg["deployment"]["num_infer_gpus"] == 1


def test_rl_renders_priority_mail_and_requeue_flags(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_slurm_flags"

    result = runner.invoke(
        app,
        [
            "rl",
            *_cli_args(
                config_paths,
                "--output-dir",
                str(output_dir),
                "--priority",
                "normal",
                "--mail",
                "begin_end",
                "--mail-user",
                "email@domain.com",
                "--slurm-resume",
                "--dry-run",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    config = _read_toml(output_dir / "configs" / "rl.toml")

    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --mail-type=begin,end" in script
    assert "#SBATCH --mail-user=email@domain.com" in script
    assert "#SBATCH --requeue" in script
    assert config["ckpt"]["resume_step"] == -1


def test_sft_cpus_per_gpu_default(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_cpus_default"

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "2", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-gpu=16" in script


def test_sft_cpus_per_gpu_custom(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_cpus_custom"

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "2", "--cpus-per-gpu", "4", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-gpu=4" in script


def test_rl_cpus_per_gpu_default(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_cpus_default"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-gpu=16" in script


def test_rl_cpus_per_gpu_custom(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_cpus_custom"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--cpus-per-gpu", "12", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-gpu=12" in script


def test_sft_exclusive_omits_cpus_per_gpu(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_exclusive"

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "8", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" in script
    assert "--cpus-per-gpu" not in script


def test_rl_exclusive_omits_cpus_per_gpu(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=4, tp=1)
    output_dir = tmp_path / "rl_out_exclusive"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--train-gpus", "4", "--infer-gpus", "4", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" in script
    assert "--cpus-per-gpu" not in script


def test_dry_run_does_not_call_sbatch(tmp_path: Path, monkeypatch) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_no_submit"
    run_mock = Mock(side_effect=AssertionError("sbatch should not be called during --dry-run"))
    monkeypatch.setattr("medarc_rl.medarc_slurm.subprocess.run", run_mock)

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "1", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    run_mock.assert_not_called()


def test_sft_accepts_config_option(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_config_option"

    result = runner.invoke(
        app,
        ["sft", *_config_option_args(config_paths, "--output-dir", str(output_dir), "--gpus", "1", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "sft.sh").exists()


def test_sft_rejects_both_config_option_and_positional(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_config_conflict"

    result = runner.invoke(
        app,
        [
            "sft",
            "--config",
            str(config_paths[0]),
            str(config_paths[0]),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "Unrecognized options" in result.output


def test_sft_uses_toml_output_dir_when_output_dir_omitted(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    toml_output_dir = tmp_path / "sft_out_from_toml"
    config_with_output = _write(tmp_path / "sft_with_output.toml", f'output_dir = "{toml_output_dir}"')

    result = runner.invoke(
        app,
        ["sft", *_config_option_args((*config_paths, config_with_output), "--gpus", "1", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert (toml_output_dir / "sft.sh").exists()
    assert f"{toml_output_dir / 'sft.sh'}" in result.output


def test_rl_uses_toml_output_dir_when_output_dir_omitted(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    toml_output_dir = tmp_path / "rl_out_from_toml"
    config_with_output = _write(tmp_path / "rl_with_output.toml", f'output_dir = "{toml_output_dir}"')

    result = runner.invoke(
        app,
        ["rl", *_config_option_args((*config_paths, config_with_output), "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert (toml_output_dir / "rl.sh").exists()
    assert f"{toml_output_dir / 'rl.sh'}" in result.output


def test_sft_dry_run_renders_dependency_and_test_only_flags(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_dep_test_only"

    result = runner.invoke(
        app,
        [
            "sft",
            *_cli_args(
                config_paths,
                "--output-dir",
                str(output_dir),
                "--gpus",
                "1",
                "--dependency",
                "afterok:12345",
                "--test-only",
                "--dry-run",
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch --account training --dependency afterok:12345 --test-only {output_dir / 'sft.sh'}" in result.output


def test_rl_dry_run_renders_dependency_and_test_only_flags(tmp_path: Path) -> None:
    config_paths = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_dep_test_only"

    result = runner.invoke(
        app,
        ["rl", *_cli_args(config_paths, "--output-dir", str(output_dir), "--dependency", "singleton", "--test-only", "--dry-run")],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch --account training --dependency singleton --test-only {output_dir / 'rl.sh'}" in result.output


def test_sft_rejects_empty_dependency(tmp_path: Path) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_empty_dependency"

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "1", "--dependency", "   ", "--dry-run")],
    )

    assert result.exit_code != 0
    assert "--dependency must not be empty." in result.output


def test_sft_test_only_submits_sbatch_with_flag(tmp_path: Path, monkeypatch) -> None:
    config_paths = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_test_only_submit"
    run_mock = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout="sbatch: Job test successful\n",
            stderr="",
        )
    )
    monkeypatch.setattr("medarc_rl.medarc_slurm.subprocess.run", run_mock)

    result = runner.invoke(
        app,
        ["sft", *_cli_args(config_paths, "--output-dir", str(output_dir), "--gpus", "1", "--test-only")],
    )

    assert result.exit_code == 0, result.output
    run_mock.assert_called_once()
    sbatch_cmd = run_mock.call_args.args[0]
    assert sbatch_cmd[0] == "sbatch"
    assert "--test-only" in sbatch_cmd
