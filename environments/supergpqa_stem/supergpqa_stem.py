from enum import Enum
from typing import Dict, Optional

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import AnswerFormat
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice

from medarc_rl.verifiers import (
    MetaParser,
    TRAIN_MCQ,
    TRAIN_ANSWER_KEY,
    TrainingMcq,
    TrainingAnswerFormat,
    TrainEvalRoutingEnv,
    TrainEvalRoutingRubric,
    add_answer_format_metadata,
    build_parser_bundle,
    choose_training_answer_format,
    format_reward,
    get_system_prompt,
    multiple_choice_accuracy_reward,
    normalize_answer_format,
    normalize_training_answer_formats,
)

disable_progress_bar()

STEM_DISCIPLINES = {"Science", "Engineering"}
_VALID_LETTERS = set("ABCDEFGHIJ")


class Difficulty(str, Enum):
    ALL = "all"
    EASY = "easy"
    MIDDLE = "middle"
    HARD = "hard"


def _build_question(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in options.items() if v not in [None, ""])
    return f"{question}\n{opts}"


def _format_training_question(training_mcq: TrainingMcq, presented_options: Dict[str, str]) -> str:
    return _build_question(training_mcq.question_data, presented_options)


def load_environment(
    disciplines: list[str] | None = None,
    field: str | None = None,
    difficulty: str | Difficulty = Difficulty.ALL,
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    training_shuffle_answers: bool | None = None,
    training_seed: int | None = None,
    train_answer_formats: list[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None = None,
    test_size: float = 0.1,
    split_seed: int = 42,
) -> vf.Environment:
    """
    Single-turn STEM RL environment using non-medicine sections of m-a-p/SuperGPQA.

    Args:
        disciplines: Disciplines to include. Defaults to ['Science', 'Engineering'].
        field: Filter by field within a discipline (e.g. 'Physics', 'Chemistry'). None = all.
        difficulty: Filter by difficulty ('easy', 'middle', 'hard', 'all').
        use_think: Enable <think>...</think> reasoning format.
        system_prompt: Override eval system prompt. Defaults to format-appropriate prompt.
        shuffle_answers: Shuffle answer choices in the eval split to prevent positional bias.
        shuffle_seed: Seed for deterministic eval answer shuffling.
        answer_format: Answer format for eval ('xml', 'boxed', 'json'). Defaults to 'xml'.
        training_shuffle_answers: Shuffle answers at rollout time during training. Defaults to shuffle_answers.
        training_seed: Seed for training format rotation. Defaults to shuffle_seed.
        train_answer_formats: Formats to rotate during training. Defaults to [xml, boxed, json].
        test_size: Fraction of data reserved for eval (default 0.1).
        split_seed: Seed for train/eval split (default 42).
    """
    if disciplines is None:
        disciplines = list(STEM_DISCIPLINES)

    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    raw = load_dataset("m-a-p/SuperGPQA", split="train").filter(
        lambda row: row["discipline"] in disciplines
    )

    if field is not None:
        raw = raw.filter(lambda row: row["field"].lower() == field.lower())

    difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
    if difficulty != Difficulty.ALL:
        raw = raw.filter(lambda row: row["difficulty"] == difficulty.value)

    # Filter rows with invalid answer letters before mapping
    raw = raw.filter(lambda row: (row.get("answer_letter") or "").strip().upper() in _VALID_LETTERS)

    # Convert options from list to dict with letter keys
    def _convert_options(row: dict) -> dict:
        opts = row["options"]
        if isinstance(opts, list):
            row["options"] = {chr(ord("A") + i): v for i, v in enumerate(opts)}
        return row

    raw = raw.map(_convert_options, load_from_cache_file=False)

    splits = raw.train_test_split(test_size=test_size, seed=split_seed)
    train_raw = splits["train"]
    test_raw = splits["test"]

    def _map(ex: dict, idx: int, *, dataset_split: str) -> dict:
        question: str = ex.get("question", "") or ""
        options: Dict[str, str] = {k: v for k, v in (ex.get("options") or {}).items() if v not in [None, ""]}
        answer_letter: str = (ex.get("answer_letter") or "").strip().upper()
        row_format_key = str(ex.get("id", idx))

        if dataset_split == "eval" and shuffle_answers and answer_letter in options:
            options, answer_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=ex.get("id", idx),
            )

        info: Dict = {
            "answer_text": options.get(answer_letter, ""),
            **({"options": options} if dataset_split == "eval" and shuffle_answers else {}),
        }

        if dataset_split == "train":
            row_answer_format = choose_training_answer_format(
                row_format_key=row_format_key,
                train_answer_formats=training_formats,
                training_seed=training_seed,
            )
            prompt = [
                {"role": "system", "content": get_system_prompt(row_answer_format, use_think=use_think)},
                {"role": "user", "content": _build_question(question, options)},
            ]
        else:
            row_answer_format = eval_answer_format
            prompt = None

        mapped: Dict = {
            "question": _build_question(question, options),
            "answer": answer_letter,
            "info": add_answer_format_metadata(
                info,
                answer_format=row_answer_format,
                row_format_key=row_format_key,
                dataset_split=dataset_split,
                training_seed=training_seed if dataset_split == "train" else None,
            ),
        }
        if prompt is not None:
            mapped["prompt"] = prompt
            mapped[TRAIN_MCQ] = TrainingMcq.from_dict_choices(
                question_data=question,
                options=options,
                answer=answer_letter,
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    eval_load_from_cache_file = not shuffle_answers
    train_mapped = train_raw.map(
        lambda ex, idx: _map(ex, idx, dataset_split="train"),
        with_indices=True,
        remove_columns=train_raw.column_names,
        load_from_cache_file=False,
    )
    test_mapped = test_raw.map(
        lambda ex, idx: _map(ex, idx, dataset_split="eval"),
        with_indices=True,
        remove_columns=test_raw.column_names,
        load_from_cache_file=eval_load_from_cache_file,
    )
    del train_raw, test_raw

    train_rubric = vf.Rubric(
        funcs=[multiple_choice_accuracy_reward, format_reward],
        weights=[1.0, 0.1],
        parser=vf.Parser(),
    )
    train_rubric.add_class_object("meta_parser", meta_parser)

    eval_parser, _ = build_parser_bundle(eval_answer_format, use_think=use_think)
    eval_rubric = vf.Rubric(funcs=[multiple_choice_accuracy_reward], weights=[1.0], parser=eval_parser)
    eval_rubric.add_class_object("meta_parser", meta_parser)

    train_env = vf.SingleTurnEnv(
        dataset=train_mapped,
        system_prompt=None,
        parser=vf.Parser(),
        rubric=train_rubric,
        env_id="supergpqa_stem",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=test_mapped,
        system_prompt=system_prompt or get_system_prompt(eval_answer_format, use_think=use_think),
        parser=eval_parser,
        rubric=eval_rubric,
        env_id="supergpqa_stem",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        parser=eval_parser,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="supergpqa_stem",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
