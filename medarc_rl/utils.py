from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import tomli_w
import typer
from pydantic import ValidationError
from pydantic_config import ConfigFileError
from prime_rl.utils.config import cli

TYPER_PASSTHROUGH_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}
T = TypeVar("T")


def maybe_autoset_auth_env(env: dict[str, str], enabled: bool) -> list[str]:
    """Best-effort local auth discovery for tools that usually rely on env vars.

    This only affects the environment passed to `sbatch` (and therefore the job), and does
    not write secrets into the generated slurm script.
    """
    if not enabled:
        return []

    msgs: list[str] = []

    if not env.get("HF_TOKEN"):
        try:
            # huggingface_hub looks in env and its local token cache.
            from huggingface_hub.utils import get_token as _hf_get_token  # type: ignore[import-not-found]
        except Exception:
            _hf_get_token = None

        if _hf_get_token is not None:
            token = _hf_get_token()
            if token:
                env["HF_TOKEN"] = token
                msgs.append("Auto-auth: set HF_TOKEN from local Hugging Face credentials.")

    return msgs


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(data, f)


def _load_settings_from_toml(
    config_cls: type[T],
    config_paths: list[Path],
    *,
    extra_cli_args: list[str] | None = None,
    **overrides: Any,
) -> T:
    if not config_paths:
        raise typer.BadParameter("At least one config file is required.", param_hint="CONFIG_TOML")
    for config_path in config_paths:
        if not config_path.exists():
            raise typer.BadParameter(f"Config file does not exist: {config_path}", param_hint="CONFIG_TOML")

    reserved_roots = set(overrides)
    filtered_extra_args = filter_wrapper_owned_cli_args(extra_cli_args or [], override_roots=reserved_roots)
    try:
        return cli(
            config_cls,
            args=[
                *[item for config_path in config_paths for item in ("@", str(config_path))],
                *filtered_extra_args,
                *_overrides_to_cli_args(overrides),
            ],
        )
    except (ConfigFileError, ValidationError, SystemExit) as e:
        raise typer.BadParameter(str(e), param_hint="CONFIG_TOML") from e


def extra_config_args(ctx: typer.Context, *, positional_count: int = 1) -> list[str]:
    """Return unknown passthrough args captured by Typer/Click for PRIME config parsing.

    With ``allow_extra_args`` + ``ignore_unknown_options``, ``ctx.args`` includes consumed
    positional args. For our wrappers, we drop the leading positional(s) (e.g. CONFIG_TOML).
    """
    raw = [arg for arg in ctx.args if arg != "--"]
    if positional_count <= 0:
        return raw

    # Click may include consumed positionals in ctx.args in some command shapes, but not others.
    # Drop only leading non-option tokens, never blindly slice.
    args = raw[:]
    dropped = 0
    while dropped < positional_count and args and not args[0].startswith("-"):
        args.pop(0)
        dropped += 1
    return args


def filter_wrapper_owned_cli_args(cli_args: list[str], *, override_roots: set[str]) -> list[str]:
    """Drop passthrough CLI overrides that target wrapper-owned top-level config fields."""
    if not override_roots:
        return cli_args

    filtered: list[str] = []
    i = 0

    while i < len(cli_args):
        token = cli_args[i]
        if not token.startswith("--"):
            filtered.append(token)
            i += 1
            continue

        key_token, has_equals, _ = token.partition("=")
        normalized = key_token[2:].replace("-", "_")
        root = normalized.split(".", 1)[0]
        if root in override_roots:
            if has_equals:
                i += 1
                continue
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue

        if has_equals:
            filtered.append(token)
            i += 1
            continue

        filtered.append(token)
        if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("-"):
            filtered.append(cli_args[i + 1])
            i += 2
        else:
            i += 1

    return filtered


def _overrides_to_cli_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        args.extend(_flatten_override(key, value))
    return args


def _flatten_override(key: str, value: Any) -> list[str]:
    option = f"--{key.replace('_', '-')}"

    if value is None:
        return []

    if isinstance(value, dict):
        args: list[str] = []
        for subkey, subvalue in value.items():
            args.extend(_flatten_override(f"{key}.{subkey}", subvalue))
        return args

    if isinstance(value, bool):
        return [option] if value else [f"--no-{key.replace('_', '-')}"]

    if isinstance(value, Path):
        return [option, str(value)]

    if isinstance(value, (list, tuple)):
        return [option, json.dumps(value)]

    return [option, str(value)]
