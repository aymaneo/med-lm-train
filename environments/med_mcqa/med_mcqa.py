"""
MedMCQA Environment

This script defines a MedMCQA evaluation environment compatible with the Verifiers framework.

The `med_mcqa` function is adapted from LightEval's default prompts:
https://github.com/huggingface/lighteval/blob/ecef2c662b9418866b6447d33b5e7d5dedd74af8/src/lighteval/tasks/default_prompts.py

Originally licensed MIT, Copyright (c) 2024 Hugging Face

Reference:
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
"""

from typing import Any

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

disable_progress_bar()  # suppress datasets progress indicators

LETTER_INDICES = ["A", "B", "C", "D"]


def _row_key(question: str, options: list[str], answer_idx: int) -> str:
    ordered_options = "|".join(f"{label}:{text}" for label, text in zip(LETTER_INDICES, options))
    return f"{question}|{ordered_options}|{answer_idx}"


def med_mcqa(line: dict[str, Any]) -> dict[str, Any]:
    """Build the standard MedMCQA multiple-choice question prompt."""
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join(
        f"{key}. {choice}\n"
        for key, choice in zip(LETTER_INDICES, [line["opa"], line["opb"], line["opc"], line["opd"]])
    )
    query += "Answer:"

    return {
        "question": query,
        "answer": LETTER_INDICES[line["cop"] - 1],
        "choices": LETTER_INDICES,
        "gold_index": line["cop"] - 1,
        "instruction": "Give a letter answer among A, B, C or D.\n",
    }


def _format_training_question(training_mcq: TrainingMcq, presented_options: dict[str, str]) -> str:
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {training_mcq.question_data}\n"
    query += "".join(f"{label}. {presented_options[label]}\n" for label in LETTER_INDICES)
    query += "Answer:"
    return query


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    training_shuffle_answers: bool | None = None,
    training_seed: int | None = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    train_answer_formats: list[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None = None,
) -> vf.Environment:
    """
    Load the MedMCQA environment with train and validation splits.
    Supports reasoning (use_think=True) or standard evaluation.
    Returns a wrapped SingleTurnEnv with distinct training and eval sub-envs.
    """
    train_ds = load_dataset("lighteval/med_mcqa", split="train")
    val_ds = load_dataset("lighteval/med_mcqa", split="validation")
    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    def _map_example(example: dict[str, Any], idx: int | None = None, *, dataset_split: str) -> dict[str, Any] | None:
        cop = example.get("cop", -1)
        if not isinstance(cop, int) or cop not in [1, 2, 3, 4]:
            return None

        question = (example.get("question") or "").strip()
        choices = [(example.get(key) or "").strip() for key in ["opa", "opb", "opc", "opd"]]
        if not question or not any(choices):
            return None

        options = [choices[0], choices[1], choices[2], choices[3]]
        answer_idx = cop - 1
        row_format_key = _row_key(question, options, answer_idx)

        if dataset_split == "eval" and shuffle_answers:
            options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=LETTER_INDICES,
                seed=shuffle_seed,
                row_id=row_format_key,
            )

        line = {
            "question": question,
            "opa": options[0],
            "opb": options[1],
            "opc": options[2],
            "opd": options[3],
            "cop": answer_idx + 1,
        }
        mapped = med_mcqa(line)
        info = {
            "answer_text": options[answer_idx],
            **({"options": dict(zip(LETTER_INDICES, options))} if dataset_split == "eval" and shuffle_answers else {}),
        }

        if dataset_split == "train":
            row_answer_format = choose_training_answer_format(
                row_format_key=row_format_key,
                train_answer_formats=training_formats,
                training_seed=training_seed,
            )
            mapped["prompt"] = [
                {"role": "system", "content": get_system_prompt(row_answer_format, use_think=use_think)},
                {"role": "user", "content": mapped["question"]},
            ]
        else:
            row_answer_format = eval_answer_format

        mapped["info"] = add_answer_format_metadata(
            info,
            answer_format=row_answer_format,
            row_format_key=row_format_key,
            dataset_split=dataset_split,
            training_seed=training_seed if dataset_split == "train" else None,
        )
        if dataset_split == "train":
            mapped[TRAIN_MCQ] = TrainingMcq.from_list_choices(
                question_data=question,
                options=options,
                answer_idx=answer_idx,
                labels=LETTER_INDICES,
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    columns_to_remove = ["question", "opa", "opb", "opc", "opd", "cop"]
    train_mapped = train_ds.map(
        lambda example, idx: _map_example(example, idx, dataset_split="train"),
        with_indices=True,
        remove_columns=columns_to_remove,
        load_from_cache_file=False,
    ).filter(lambda example: example is not None, load_from_cache_file=False)
    val_mapped = val_ds.map(
        lambda example, idx: _map_example(example, idx, dataset_split="eval"),
        with_indices=True,
        remove_columns=columns_to_remove,
        load_from_cache_file=not shuffle_answers,
    ).filter(lambda example: example is not None, load_from_cache_file=not shuffle_answers)

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
        env_id="med_mcqa",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=val_mapped,
        system_prompt=system_prompt or get_system_prompt(eval_answer_format, use_think=use_think),
        parser=eval_parser,
        rubric=eval_rubric,
        env_id="med_mcqa",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="med_mcqa",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
