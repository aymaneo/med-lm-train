"""
HEAD-QA environment

This script defines an evaluation environment for HEAD-QA compatible with the Verifiers framework.

The prompts were adapted from the HEAD-QA v2 paper.
"""

from typing import Any

import verifiers as vf
from datasets import load_dataset
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


def _row_key(question_text: str, options: list[str], answer_idx: int) -> str:
    ordered_options = "|".join(f"{idx + 1}:{option}" for idx, option in enumerate(options))
    return f"{question_text}|{ordered_options}|{answer_idx}"


def zero_shot_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the zero-shot prompt."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join(f"{answer['aid']}. {answer['atext'].strip()}" for answer in answers)

    prompt = (
        "You are an expert in specialized scientific and health disciplines. "
        "Respond to the following multiple-choice question:\n"
        f"{question_text}\n{options_text}\n"
    )

    correct_answer = example.get("ra", -1)
    return {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(answer["aid"]) for answer in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }


def _format_training_question(training_mcq: TrainingMcq, presented_options: dict[str, str]) -> str:
    options_text = "\n".join(f"{label}. {presented_options[label]}" for label in training_mcq.labels)
    return (
        "You are an expert in specialized scientific and health disciplines. "
        "Respond to the following multiple-choice question:\n"
        f"{training_mcq.question_data}\n{options_text}\n"
    )


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
    Load the HEAD-QA environment with train and validation splits.
    Returns a wrapped SingleTurnEnv with distinct training and eval sub-envs.
    """
    train_ds = load_dataset("EleutherAI/headqa", "en", split="train")
    val_ds = load_dataset("EleutherAI/headqa", "en", split="validation")
    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    def _map_example(example: dict[str, Any], idx: int | None = None, *, dataset_split: str) -> dict[str, Any] | None:
        correct_answer = example.get("ra", -1)
        question_text = (example.get("qtext") or "").strip()
        answers = example.get("answers", [])

        if not question_text or not answers or not (1 <= correct_answer <= len(answers)):
            return None

        options = [answer["atext"].strip() for answer in answers]
        answer_idx = correct_answer - 1
        row_format_key = _row_key(question_text, options, answer_idx)

        if dataset_split == "eval" and shuffle_answers:
            indices = [str(i + 1) for i in range(len(options))]
            options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=indices,
                seed=shuffle_seed,
                row_id=row_format_key,
            )

        temp_example = {
            "qtext": question_text,
            "answers": [{"aid": i + 1, "atext": option} for i, option in enumerate(options)],
            "ra": answer_idx + 1,
        }
        mapped = zero_shot_prompt(temp_example)

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
            {
                **mapped["info"],
                **(
                    {"options": {str(i + 1): option for i, option in enumerate(options)}}
                    if dataset_split == "eval" and shuffle_answers
                    else {}
                ),
            },
            answer_format=row_answer_format,
            row_format_key=row_format_key,
            dataset_split=dataset_split,
            training_seed=training_seed if dataset_split == "train" else None,
        )
        if dataset_split == "train":
            mapped[TRAIN_MCQ] = TrainingMcq.from_list_choices(
                question_data=question_text,
                options=options,
                answer_idx=answer_idx,
                labels=[str(idx + 1) for idx in range(len(options))],
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    columns_to_remove = ["qtext", "answers", "ra"]
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
        env_id="head_qa",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=val_mapped,
        system_prompt=system_prompt or get_system_prompt(eval_answer_format, use_think=use_think),
        parser=eval_parser,
        rubric=eval_rubric,
        env_id="head_qa",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="head_qa",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
