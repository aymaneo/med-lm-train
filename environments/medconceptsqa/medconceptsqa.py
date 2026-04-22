from enum import Enum
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
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


disable_progress_bar()  # suppress datasets mapping progress bar


class Vocab(str, Enum):
    ATC = "atc"
    ICD10CM = "icd10cm"
    ICD10PROC = "icd10proc"
    ICD9CM = "icd9cm"
    ICD9PROC = "icd9proc"
    ICD10CM_SAMPLE = "icd10cm_sample"
    ALL = "all"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def _extract_question_and_options(row: dict) -> tuple[str, dict[str, str]]:
    question = row.get("question", "") or ""
    options: dict[str, str] = {}
    for idx, label in enumerate(("A", "B", "C", "D"), start=1):
        value = row.get(f"option{idx}", "")
        if value not in ("", None):
            options[label] = value

    def looks_like_option(line: str) -> bool:
        candidate = line.strip()
        for label in ("A", "B", "C", "D"):
            for sep in (".", ")", ":", "-"):
                if candidate.startswith(f"{label}{sep}"):
                    return True
        return False

    question_lines = [line for line in question.splitlines() if not looks_like_option(line)]
    question_stem = "\n".join(question_lines).strip()
    return question_stem, options


def _format_stem_with_options(question: str, options: dict[str, str]) -> str:
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    return f"{question}\n{option_block}".strip()


def _build_training_question(question_stem: str, options: dict[str, str], few_shot_prompt: str) -> str:
    formatted_question = _format_stem_with_options(question_stem, options)
    return (
        "Answer A, B, C, D according to the answer to this multiple choice question.\n"
        + few_shot_prompt
        + ("\n" if few_shot_prompt else "")
        + formatted_question
        + "\nAnswer:"
    )


def _row_key(row: dict) -> str:
    concept_id = row.get("concept_id")
    if concept_id not in (None, ""):
        return f"concept:{concept_id}"

    question_stem, options = _extract_question_and_options(row)
    ordered_options = "|".join(f"{label}:{options.get(label, '')}" for label in ("A", "B", "C", "D"))
    return f"text:{question_stem}|{ordered_options}|{row.get('answer_id', '')}"


def _render_answer(answer_format: AnswerFormat, answer: str) -> str:
    if answer_format == AnswerFormat.XML:
        return f"<answer>{answer}</answer>"
    if answer_format == AnswerFormat.BOXED:
        return f"\\boxed{{{answer}}}"
    if answer_format == AnswerFormat.JSON:
        return f'{{"answer":"{answer}"}}'
    raise ValueError(f"Unsupported answer format: {answer_format}")


def _create_few_shot_data(few_shot_set: Dataset, num_few_shot: int, answer_format: AnswerFormat) -> dict[tuple, str]:
    few_shot_examples: dict[tuple, list[str]] = {}

    for row in few_shot_set:
        key = (row["vocab"], row["level"])
        few_shot_examples.setdefault(key, [])
        if len(few_shot_examples[key]) >= num_few_shot:
            continue

        question_stem, options = _extract_question_and_options(row)
        formatted_question = _format_stem_with_options(question_stem, options)
        prompt = f"{formatted_question}\nAnswer: {_render_answer(answer_format, row['answer_id'])}\n\n".replace(
            "  ", ""
        )
        few_shot_examples[key].append(prompt)

    return {key: "".join(value) for key, value in few_shot_examples.items()}


def _subset_name(vocab: Vocab, level: Difficulty) -> str:
    if vocab is Vocab.ALL:
        return "all"
    return f"{vocab.value}_{level.value}"


def _format_training_question(training_mcq: TrainingMcq, presented_options: dict[str, str]) -> str:
    question_data = training_mcq.question_data
    return _build_training_question(
        question_data["question_stem"],
        presented_options,
        question_data.get("few_shot_prompt", ""),
    )


def load_environment(
    num_few_shot: int = 4,
    use_think: bool = False,
    vocab: Vocab | str = Vocab.ICD10CM_SAMPLE,
    difficulty: Difficulty | str = Difficulty.EASY,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    training_shuffle_answers: bool | None = None,
    training_seed: int | None = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    train_answer_formats: list[TrainingAnswerFormat | str] | TrainingAnswerFormat | str | None = None,
) -> vf.Environment:
    """MedConceptsQA training/eval environment."""
    vocab = Vocab(vocab) if isinstance(vocab, str) else vocab
    level = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
    subset = _subset_name(vocab, level)
    eval_answer_format = normalize_answer_format(answer_format)
    training_formats = normalize_training_answer_formats(answer_format, train_answer_formats)
    meta_parser = MetaParser(use_think=use_think)
    training_shuffle_answers = shuffle_answers if training_shuffle_answers is None else training_shuffle_answers
    training_seed = shuffle_seed if training_seed is None else training_seed

    if vocab == Vocab.ICD10CM_SAMPLE:
        sample_subset = subset.replace("_sample", "")
        sample_ds = load_dataset("sameedkhan/medconceptsqa-sample_medarc_2k", sample_subset)
        full_ds = load_dataset("ofir408/MedConceptsQA", sample_subset)
        eval_source = sample_ds["test"]
        eval_few_shot_source = sample_ds["dev"]
        sample_keys = {_row_key(row) for row in eval_source}
        train_source = full_ds["test"].filter(lambda row: _row_key(row) not in sample_keys, load_from_cache_file=False)
        train_few_shot_source = full_ds["dev"]
    else:
        ds = load_dataset("ofir408/MedConceptsQA", subset)
        eval_source = ds["dev"]
        eval_few_shot_source = ds["dev"]
        train_source = ds["test"]
        train_few_shot_source = ds["dev"]

    eval_few_shot_data = (
        _create_few_shot_data(eval_few_shot_source, num_few_shot, eval_answer_format) if num_few_shot > 0 else {}
    )
    training_few_shot_data = (
        {
            format_value: _create_few_shot_data(train_few_shot_source, num_few_shot, format_value)
            for format_value in training_formats
        }
        if num_few_shot > 0
        else {}
    )

    def _map(row: dict, idx: int | None = None, *, dataset_split: str) -> dict:
        row_vocab = row["vocab"]
        row_level = row["level"]
        question_stem, options = _extract_question_and_options(row)
        row_key = _row_key(row)
        row_id = row.get("id") or row.get("concept_id") or row_key
        answer = row["answer_id"]
        row_format_key = row_key

        if dataset_split == "eval" and shuffle_answers and answer in options:
            options, answer, _ = randomize_multiple_choice(
                options=options,
                answer_choice=answer,
                seed=shuffle_seed,
                row_id=row_id,
            )

        if dataset_split == "train":
            row_answer_format = choose_training_answer_format(
                row_format_key=row_format_key,
                train_answer_formats=training_formats,
                training_seed=training_seed,
            )
            few_shot_prompt = training_few_shot_data.get(row_answer_format, {}).get((row_vocab, row_level), "")
        else:
            row_answer_format = eval_answer_format
            few_shot_prompt = eval_few_shot_data.get((row_vocab, row_level), "")

        full_question = _build_training_question(question_stem, options, few_shot_prompt)

        info: dict[str, Any] = {
            "answer_text": options.get(answer, row.get("answer")),
            **({"options": options} if dataset_split == "eval" and shuffle_answers else {}),
        }
        mapped = {
            "question": full_question,
            "answer": answer,
            "info": add_answer_format_metadata(
                info,
                answer_format=row_answer_format,
                row_format_key=row_format_key,
                dataset_split=dataset_split,
                training_seed=training_seed if dataset_split == "train" else None,
            ),
        }
        if dataset_split == "train":
            mapped["prompt"] = [
                {"role": "system", "content": get_system_prompt(row_answer_format, use_think=use_think)},
                {"role": "user", "content": full_question},
            ]
            mapped[TRAIN_MCQ] = TrainingMcq.from_dict_choices(
                question_data={"question_stem": question_stem, "few_shot_prompt": few_shot_prompt},
                options=options,
                answer=answer,
            ).to_payload()
            mapped[TRAIN_ANSWER_KEY] = training_shuffle_answers
        return mapped

    train_mapped = train_source.map(
        lambda row, idx: _map(row, idx, dataset_split="train"),
        with_indices=True,
        remove_columns=train_source.column_names,
        load_from_cache_file=False,
    )
    eval_mapped = eval_source.map(
        lambda row, idx: _map(row, idx, dataset_split="eval"),
        with_indices=True,
        remove_columns=eval_source.column_names,
        load_from_cache_file=not shuffle_answers,
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
        env_id="medconceptsqa",
    )
    eval_env = vf.SingleTurnEnv(
        eval_dataset=eval_mapped,
        rubric=eval_rubric,
        system_prompt=get_system_prompt(eval_answer_format, use_think=use_think),
        parser=eval_parser,
        env_id="medconceptsqa",
    )

    return TrainEvalRoutingEnv(
        train_env=train_env,
        eval_env=eval_env,
        rubric=TrainEvalRoutingRubric(train_rubric=train_rubric, eval_rubric=eval_rubric),
        env_id="medconceptsqa",
        format_training_question=_format_training_question,
        use_think=use_think,
    )
