from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice

from .prompts import get_system_prompt

TRAIN_MCQ = "TRAIN_MCQ"
TRAIN_ANSWER_KEY = "train_answer_reshuffle"


@dataclass(frozen=True)
class TrainingMcq:
    question_data: Any
    options: tuple[str, ...]
    labels: tuple[str, ...]
    answer_idx: int

    @classmethod
    def from_dict_choices(cls, *, question_data: Any, options: dict[str, str], answer: str) -> TrainingMcq:
        labels = tuple(options.keys())
        answer_idx = labels.index(answer)
        return cls(
            question_data=question_data,
            options=tuple(options.values()),
            labels=labels,
            answer_idx=answer_idx,
        )

    @classmethod
    def from_list_choices(
        cls,
        *,
        question_data: Any,
        options: list[str],
        answer_idx: int,
        labels: list[str],
    ) -> TrainingMcq:
        return cls(
            question_data=question_data,
            options=tuple(options),
            labels=tuple(labels),
            answer_idx=answer_idx,
        )

    @classmethod
    def from_value(cls, value: TrainingMcq | dict[str, Any]) -> TrainingMcq:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("TRAIN_MCQ payload must be a TrainingMcq or dict")
        raw_qd = value["question_data"]
        return cls(
            question_data=json.loads(raw_qd) if isinstance(raw_qd, str) else raw_qd,
            options=tuple(value["options"]),
            labels=tuple(value["labels"]),
            answer_idx=value["answer_idx"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "question_data": json.dumps(self.question_data),
            "options": list(self.options),
            "labels": list(self.labels),
            "answer_idx": self.answer_idx,
        }


def _should_reshuffle_rollout(state: dict[str, Any]) -> bool:
    info = state.get("info")
    if not isinstance(info, dict) or info.get("dataset_split") != "train":
        return False

    rollout_input = state.get("input")
    if not isinstance(rollout_input, dict):
        return False

    return bool(rollout_input.get(TRAIN_ANSWER_KEY))


def _present_training_mcq(training_mcq: TrainingMcq) -> tuple[dict[str, str], str, str]:
    shuffled_options, answer, answer_idx = randomize_multiple_choice(
        options=list(training_mcq.options),
        answer_choice=training_mcq.answer_idx,
        labels=list(training_mcq.labels),
        seed=-1,
    )
    shuffled_options = list(shuffled_options)
    option_map = dict(zip(training_mcq.labels, shuffled_options))
    return option_map, training_mcq.labels[answer_idx], shuffled_options[answer_idx]


def apply_train_answer_reshuffle(
    state: dict[str, Any],
    *,
    format_training_question: Callable[[TrainingMcq, dict[str, str]], str] | None,
    use_think: bool,
) -> dict[str, Any]:
    if format_training_question is None or not _should_reshuffle_rollout(state):
        return state

    rollout_input = state.get("input")
    if not isinstance(rollout_input, dict):
        raise ValueError("Missing rollout input for answer reshuffling")

    raw_training_mcq = rollout_input.get(TRAIN_MCQ)
    if raw_training_mcq is None:
        raise ValueError(f"Missing {TRAIN_MCQ} payload for answer reshuffling")
    training_mcq = TrainingMcq.from_value(raw_training_mcq)

    info = state.get("info")
    if not isinstance(info, dict):
        raise ValueError("Missing info metadata for answer reshuffling")

    options, answer, answer_text = _present_training_mcq(training_mcq)
    updated_info = dict(info)
    updated_info["answer_text"] = answer_text
    updated_info["options"] = options

    question = format_training_question(training_mcq, options)
    state["question"] = question
    state["answer"] = answer
    state["info"] = updated_info
    state["prompt"] = [
        {"role": "system", "content": get_system_prompt(updated_info["answer_format"], use_think=use_think)},
        {"role": "user", "content": question},
    ]
    return state
