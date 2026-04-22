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

disable_progress_bar()  # suppress datasets progress indicators


def _build_prompt(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question:{question}\n{opts}\nAnswer:"


def _format_training_question(training_mcq: TrainingMcq, presented_options: Dict[str, str]) -> str:
    return _build_prompt(training_mcq.question_data, presented_options)


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    training_shuffle_answers: bool | None = None,
    training_seed: int | None = None,
    train_answer_formats: list[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None = None,
) -> vf.Environment:
    """
    MedQA-USMLE-4-options multiple-choice evaluation
    - Train split = dataset
    - Test split = eval_dataset
    - Supports reasoning (use_think=True) or non-reasoning models
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    def _map(ex, idx=None, *, dataset_split: str):
        q: str = ex["question"]
        options: Dict[str, str] = dict(ex["options"])
        gold_letter: str = ex["answer_idx"].strip().upper()
        row_format_key = str(ex.get("id", idx))

        if dataset_split == "eval" and shuffle_answers and gold_letter in options:
            options, gold_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=gold_letter,
                seed=shuffle_seed,
                row_id=ex.get("id", idx),
            )

        info = {
            "answer_text": options.get(gold_letter, ""),
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
                {"role": "user", "content": _build_prompt(q, options)},
            ]
        else:
            row_answer_format = eval_answer_format
            prompt = None

        mapped = {
            "question": _build_prompt(q, options),
            "answer": gold_letter,
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
                question_data=q,
                options=options,
                answer=gold_letter,
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    eval_load_from_cache_file = not shuffle_answers
    train_mapped = ds["train"].map(
        lambda ex, idx: _map(ex, idx, dataset_split="train"),
        with_indices=True,
        remove_columns=ds["train"].column_names,
        load_from_cache_file=False,
    )
    test_mapped = ds["test"].map(
        lambda ex, idx: _map(ex, idx, dataset_split="eval"),
        with_indices=True,
        remove_columns=ds["test"].column_names,
        load_from_cache_file=eval_load_from_cache_file,
    )

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
        env_id="medqa",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=test_mapped,
        system_prompt=system_prompt or get_system_prompt(eval_answer_format, use_think=use_think),
        parser=eval_parser,
        rubric=eval_rubric,
        env_id="medqa",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        parser=eval_parser,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="medqa",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
