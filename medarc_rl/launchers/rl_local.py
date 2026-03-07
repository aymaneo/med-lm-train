# Minimally modified from PrimeIntellect's rl_local() in PRIME-RL.
# Apache License 2.0. Copyright PrimeIntellect AI.
# Source: https://github.com/PrimeIntellect-ai/prime-rl/blob/4825d76e7599/src/prime_rl/entrypoints/rl.py
#
# Modifications: SLURM shared-node GPU parsing, dynamic port allocation, and per-process cache isolation.

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

from pydantic_config import cli
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.utils import get_free_port


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_visible_gpus() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set; cannot determine shared-node GPU allocation.")
    gpu_ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES is invalid: {raw!r}")
    return gpu_ids


def _build_cache_env(cache_root: Path, name: str) -> dict[str, str]:
    xdg_cache = cache_root / f"xdg_{name}"
    triton_cache = cache_root / f"triton_{name}"
    torchinductor_cache = cache_root / f"torchinductor_{name}"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    return {
        "XDG_CACHE_HOME": str(xdg_cache),
        "TRITON_CACHE_DIR": str(triton_cache),
        "TORCHINDUCTOR_CACHE_DIR": str(torchinductor_cache),
    }


def rl_local(config: RLConfig) -> None:
    logger = setup_logger(
        config.log.level or "info",
        log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )

    if config.dry_run:
        logger.warning("--dry-run is set. No RL training will be started. Only writing resolved subconfigs to disk.")
        write_subconfigs(config, config.output_dir)
        logger.info(f"Dumping resolved subconfigs to {config.output_dir}")
        return

    if config.deployment.type != "single_node":
        raise ValueError("medarc_rl.launchers.rl_local only supports deployment.type=single_node")

    if getattr(config.deployment, "num_teacher_gpus", 0):
        raise ValueError("teacher_gpus are not supported by medarc_rl.launchers.rl_local")
    if getattr(config, "teacher_inference", None) is not None:
        raise ValueError("teacher_inference is not supported by medarc_rl.launchers.rl_local")

    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    num_train_gpus = config.deployment.num_train_gpus
    expected_visible = num_train_gpus + num_infer_gpus
    single_gpu_mode = _env_flag("MEDARC_SINGLE_GPU")

    visible_gpu_ids = _parse_visible_gpus()
    if single_gpu_mode:
        if len(visible_gpu_ids) != 1:
            raise RuntimeError(
                "MEDARC_SINGLE_GPU=1 requires exactly one visible GPU, got "
                f"{len(visible_gpu_ids)} (CUDA_VISIBLE_DEVICES={','.join(visible_gpu_ids)!r})"
            )
        infer_gpu_ids = visible_gpu_ids[:1] if config.inference is not None else []
        trainer_gpu_ids = visible_gpu_ids[:1]
    else:
        if len(visible_gpu_ids) != expected_visible:
            raise RuntimeError(
                f"Expected {expected_visible} visible GPUs (train={num_train_gpus}, infer={num_infer_gpus}), got "
                f"{len(visible_gpu_ids)} (CUDA_VISIBLE_DEVICES={','.join(visible_gpu_ids)!r})"
            )
        infer_gpu_ids = visible_gpu_ids[:num_infer_gpus]
        trainer_gpu_ids = visible_gpu_ids[num_infer_gpus : num_infer_gpus + num_train_gpus]

    if len(trainer_gpu_ids) != num_train_gpus:
        raise RuntimeError(f"Trainer GPU slice mismatch: expected {num_train_gpus}, got {len(trainer_gpu_ids)}")
    if config.inference is not None and len(infer_gpu_ids) != num_infer_gpus:
        raise RuntimeError(f"Inference GPU slice mismatch: expected {num_infer_gpus}, got {len(infer_gpu_ids)}")

    infer_port = get_free_port() if config.inference is not None else None
    weight_broadcast_is_nccl = getattr(config.trainer.weight_broadcast, "type", None) == "nccl"
    weight_broadcast_port = get_free_port() if weight_broadcast_is_nccl else None
    rdzv_port = get_free_port()

    start_command = sys.argv
    logger.info("Starting MedArc RL local launcher")
    logger.debug(f"Launcher command: {' '.join(start_command)}")
    logger.info(
        "GPU allocation: "
        f"infer={','.join(infer_gpu_ids) if infer_gpu_ids else '-'} "
        f"trainer={','.join(trainer_gpu_ids)} "
        f"(visible={','.join(visible_gpu_ids)}, single_gpu={single_gpu_mode})"
    )
    if infer_port is not None:
        logger.info(f"Selected inference port: {infer_port}")
    if weight_broadcast_port is not None:
        logger.info(f"Selected weight broadcast port: {weight_broadcast_port}")
    logger.info(f"Selected torchrun rendezvous port: {rdzv_port}")

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "torchrun").mkdir(parents=True, exist_ok=True)

    runtime_config_dir = config.output_dir / "configs"
    write_subconfigs(config, runtime_config_dir)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        cache_root = Path(slurm_tmpdir) / f"medarc_rl_launcher_{slurm_job_id}_{uuid.uuid4().hex[:8]}"
    else:
        cache_root = Path("/tmp/medarc") / slurm_job_id
    cache_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    trainer_process: Popen | None = None
    orchestrator_process: Popen | None = None

    try:
        if config.inference is not None:
            inference_cmd = [
                "inference",
                "@",
                (runtime_config_dir / "inference.toml").as_posix(),
                "--server.host",
                "127.0.0.1",
                "--server.port",
                str(infer_port),
            ]
            logger.info(f"Starting inference process on GPU(s) {' '.join(infer_gpu_ids)}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            with (log_dir / "inference.log").open("w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={
                        **base_env,
                        **_build_cache_env(cache_root, "infer"),
                        "CUDA_VISIBLE_DEVICES": ",".join(infer_gpu_ids),
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No inference config specified, skipping local inference server startup. "
                "Make sure orchestrator points to an existing server."
            )

        orchestrator_cmd = [
            "orchestrator",
            "@",
            (runtime_config_dir / "orchestrator.toml").as_posix(),
        ]
        if infer_port is not None:
            orchestrator_cmd.extend(["--client.base-url", f"http://127.0.0.1:{infer_port}/v1"])
        if weight_broadcast_port is not None:
            orchestrator_cmd.extend(
                ["--weight_broadcast.host", "127.0.0.1", "--weight_broadcast.port", str(weight_broadcast_port)]
            )

        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with (log_dir / "orchestrator.log").open("w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **base_env,
                    **_build_cache_env(cache_root, "orch"),
                    "WANDB_PROGRAM": "medarc_rl.launchers.rl_local",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        trainer_cmd = [
            "torchrun",
            f"--rdzv-endpoint=localhost:{rdzv_port}",
            f"--rdzv-id={uuid.uuid4().hex}",
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (runtime_config_dir / "trainer.toml").as_posix(),
        ]
        if weight_broadcast_port is not None:
            trainer_cmd.extend(
                ["--weight_broadcast.host", "127.0.0.1", "--weight_broadcast.port", str(weight_broadcast_port)]
            )

        logger.info(f"Starting trainer process on GPU(s) {' '.join(trainer_gpu_ids)}")
        logger.debug(f"Trainer start command: {' '.join(trainer_cmd)}")
        with (log_dir / "trainer.log").open("w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **base_env,
                    **_build_cache_env(cache_root, "train"),
                    "CUDA_VISIBLE_DEVICES": ",".join(trainer_gpu_ids),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "WANDB_PROGRAM": "medarc_rl.launchers.rl_local",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        logger.success("Startup complete. Showing trainer logs...")
        tail_process = Popen(["tail", "-F", str(log_dir / "trainer.log")])
        processes.append(tail_process)

        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                logger.error(f"Error: {error_queue[0]}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                raise SystemExit(1)
            time.sleep(1)

        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            raise SystemExit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            raise SystemExit(trainer_process.returncode)

        logger.success("RL training finished!")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise SystemExit(trainer_process.returncode or 0)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise SystemExit(1)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def main() -> None:
    config = cli(RLConfig)
    rl_local(config)


if __name__ == "__main__":
    main()
