from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import Any

from verifiers.types import State


def timing_state() -> dict[str, float]:
    return {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0, "start_time": time.time()}


def present_row(env, row: dict[str, Any]):
    return asyncio.run(env.setup_state(State(input=deepcopy(row))))


def completion_for_format(answer_format: str, answer: str) -> list[dict[str, str]]:
    if answer_format == "xml":
        return [{"role": "assistant", "content": f"<answer>{answer}</answer>"}]
    if answer_format == "boxed":
        return [{"role": "assistant", "content": f"\\boxed{{{answer}}}"}]
    if answer_format == "json":
        return [{"role": "assistant", "content": f'{{"answer":"{answer}"}}'}]
    raise AssertionError(answer_format)
