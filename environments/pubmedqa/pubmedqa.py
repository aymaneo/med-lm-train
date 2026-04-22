import json
import os

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

PROMPT_TEMPLATE = """Select the best answer.

Context: {abstract_as_context}

Question: {question}
{options_block}
Answer: """

BASE_OPTIONS = {"A": "Yes", "B": "No", "C": "Maybe"}


def map_row_to_mcq_prompt(
    row: dict,
    idx: int | None = None,
    *,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
):
    """Map dataset format for PubMedQA samples."""
    question_text = row.get("question")

    context_dict = row.get("context")
    labels = context_dict.get("labels")
    contexts = context_dict.get("contexts")

    final_decision = row.get("final_decision", "").lower()
    choices_map = {"yes": "A", "no": "B", "maybe": "C"}
    correct_answer_letter = choices_map[final_decision]

    options = dict(BASE_OPTIONS)

    if shuffle_answers:
        row_id = row.get("pubid", idx)
        shuffled, correct_answer_letter, _ = randomize_multiple_choice(
            options=options,
            answer_choice=correct_answer_letter,
            seed=shuffle_seed,
            row_id=row_id,
        )
        options = dict(shuffled)

    formatted_contexts = [f"{label}. {context}" for label, context in zip(labels, contexts)]
    context_text = "\n".join(formatted_contexts)
    options_block = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    complete_prompt = PROMPT_TEMPLATE.format(
        abstract_as_context=context_text,
        question=question_text,
        options_block=options_block,
    )

    info = {
        "answer_text": options.get(correct_answer_letter, final_decision),
    }
    if shuffle_answers:
        info["options"] = options

    return {
        "question": complete_prompt,
        "answer": correct_answer_letter,
        "task": "pubmedqa",
        "info": info,
    }


def _format_training_question(training_mcq: TrainingMcq, presented_options: dict[str, str]) -> str:
    question_data = training_mcq.question_data
    formatted_contexts = [
        f"{label}. {text}" for label, text in zip(question_data["context_labels"], question_data["context_texts"])
    ]
    options_block = "\n".join(f"{letter}. {text}" for letter, text in presented_options.items())
    question = PROMPT_TEMPLATE.format(
        abstract_as_context="\n".join(formatted_contexts),
        question=question_data["question"],
        options_block=options_block,
    )
    return question


def load_environment(
    use_think: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    training_shuffle_answers: bool | None = None,
    training_seed: int | None = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    train_answer_formats: list[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None = None,
) -> vf.Environment:
    """
    PubMedQA environment using classification-based evaluation.

    The training sub-env uses the artificial split; the eval sub-env uses the fixed labeled subset.
    """
    dataset_path = "qiaojin/PubMedQA"
    dataset_train = load_dataset(dataset_path, name="pqa_artificial", split="train")
    dataset_test = load_dataset(dataset_path, name="pqa_labeled", split="train")

    here = os.path.dirname(__file__)
    file_path = os.path.join(here, "data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = json.load(f)

    dataset_test = dataset_test.filter(lambda sample: str(sample["pubid"]) in test_ids)
    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    def _map_row(row: dict, idx: int | None = None, *, dataset_split: str) -> dict:
        mapped = map_row_to_mcq_prompt(
            row,
            idx,
            shuffle_answers=dataset_split == "eval" and shuffle_answers,
            shuffle_seed=shuffle_seed,
        )
        row_format_key = str(row.get("pubid", idx))

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
            mapped["info"],
            answer_format=row_answer_format,
            row_format_key=row_format_key,
            dataset_split=dataset_split,
            training_seed=training_seed if dataset_split == "train" else None,
        )
        if dataset_split == "train":
            context_dict = row.get("context") or {}
            mapped[TRAIN_MCQ] = TrainingMcq.from_dict_choices(
                question_data={
                    "question": row.get("question"),
                    "context_labels": list(context_dict.get("labels") or []),
                    "context_texts": list(context_dict.get("contexts") or []),
                },
                options=dict(BASE_OPTIONS),
                answer=mapped["answer"],
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    mapped_dataset_train = dataset_train.map(
        lambda row, idx: _map_row(row, idx, dataset_split="train"),
        with_indices=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    mapped_dataset_test = dataset_test.map(
        lambda row, idx: _map_row(row, idx, dataset_split="eval"),
        with_indices=True,
        load_from_cache_file=not shuffle_answers,
        keep_in_memory=True,
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
        dataset=mapped_dataset_train,
        system_prompt=None,
        parser=vf.Parser(),
        rubric=train_rubric,
        env_id="pubmedqa",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=mapped_dataset_test,
        system_prompt=get_system_prompt(eval_answer_format, use_think=use_think),
        rubric=eval_rubric,
        parser=eval_parser,
        env_id="pubmedqa",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="pubmedqa",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
