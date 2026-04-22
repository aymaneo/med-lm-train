from __future__ import annotations

import random
from enum import StrEnum
from typing import Any, Sequence

from medarc_verifiers.prompts import AnswerFormat


class TrainingAnswerFormat(StrEnum):
    XML = AnswerFormat.XML.value
    BOXED = AnswerFormat.BOXED.value
    JSON = AnswerFormat.JSON.value
    RANDOM = "random"


DEFAULT_TRAIN_ANSWER_FORMATS: tuple[AnswerFormat, ...] = (
    AnswerFormat.XML,
    AnswerFormat.BOXED,
    AnswerFormat.JSON,
)


def normalize_answer_format(answer_format: AnswerFormat | str) -> AnswerFormat:
    if isinstance(answer_format, AnswerFormat):
        return answer_format
    normalized = answer_format.strip().lower()
    try:
        return AnswerFormat(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported answer format: {answer_format}") from exc


def normalize_training_answer_formats(
    answer_format: AnswerFormat | str,
    train_answer_formats: Sequence[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None,
) -> list[AnswerFormat]:
    if train_answer_formats is None:
        return [normalize_answer_format(answer_format)]

    if isinstance(train_answer_formats, (str, TrainingAnswerFormat)):
        raw_formats = [train_answer_formats]
    else:
        raw_formats = list(train_answer_formats)

    normalized_raw: list[AnswerFormat | TrainingAnswerFormat] = []
    for raw_format in raw_formats:
        if isinstance(raw_format, TrainingAnswerFormat):
            if raw_format == TrainingAnswerFormat.RANDOM:
                normalized_raw.append(raw_format)
            else:
                normalized_raw.append(normalize_answer_format(raw_format.value))
        elif isinstance(raw_format, str) and raw_format.strip().lower() == TrainingAnswerFormat.RANDOM.value:
            normalized_raw.append(TrainingAnswerFormat.RANDOM)
        else:
            normalized_raw.append(normalize_answer_format(raw_format))

    if TrainingAnswerFormat.RANDOM in normalized_raw:
        if len(normalized_raw) != 1:
            raise ValueError("Random training answer format cannot be combined with explicit formats")
        return list(DEFAULT_TRAIN_ANSWER_FORMATS)

    return list(dict.fromkeys(normalized_raw))


def choose_training_answer_format(
    *,
    row_format_key: str,
    train_answer_formats: Sequence[AnswerFormat],
    training_seed: int | None,
) -> AnswerFormat:
    if not train_answer_formats:
        raise ValueError("train_answer_formats must not be empty")

    if len(train_answer_formats) == 1:
        return train_answer_formats[0]

    if training_seed is None:
        return random.choice(train_answer_formats)

    rng = random.Random(f"{training_seed}:{row_format_key}")
    return rng.choice(train_answer_formats)


def add_answer_format_metadata(
    info: dict[str, Any] | None,
    *,
    answer_format: AnswerFormat,
    row_format_key: str,
    dataset_split: str,
    training_seed: int | None = None,
) -> dict[str, Any]:
    merged = dict(info or {})
    merged["answer_format"] = answer_format.value
    merged["row_format_key"] = row_format_key
    merged["dataset_split"] = dataset_split
    if training_seed is not None:
        merged["training_seed"] = training_seed
    return merged
