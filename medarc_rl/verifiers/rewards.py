from __future__ import annotations

import re
from typing import Any

import verifiers as vf
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy

from .think import strict_think_strip


def boxed_format_reward(*, use_think: bool):
    def reward(completion, **_: Any) -> float:
        text = vf.Parser().parse_answer(completion) or ""
        if use_think:
            text = strict_think_strip(text)
        if not text:
            return 0.0
        if len(re.findall(r"\\boxed\{", text)) != 1:
            return 0.0
        extracted = vf.extract_boxed_answer(text, strict=True)
        return 1.0 if extracted.strip() else 0.0

    return reward


def multiple_choice_accuracy_reward(
    completion,
    answer: str,
    parser: vf.Parser,
    info: dict[str, Any] | None = None,
    meta_parser=None,
    **kwargs,
) -> float:
    if meta_parser is not None:
        parsed = meta_parser.parse_for_row(completion, info) or ""
    else:
        parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def format_reward(completion, info: dict[str, Any] | None = None, meta_parser=None, **_: Any) -> float:
    if meta_parser is None:
        return 0.0
    if info is None:
        raise ValueError("Missing info for format-aware scoring")
    if info.get("dataset_split") != "train":
        return 0.0
    return meta_parser.format_reward_for_row(completion, info)
