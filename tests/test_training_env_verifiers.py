from copy import deepcopy

from datasets import Dataset
from medarc_verifiers.prompts import AnswerFormat
import medarc_rl.verifiers.presentation as presentation
from medarc_rl.verifiers import (
    MetaParser,
    StrictMaybeThinkParser,
    TRAIN_MCQ,
    TRAIN_ANSWER_KEY,
    TrainingMcq,
    apply_train_answer_reshuffle,
    build_parser_bundle,
    choose_training_answer_format,
    normalize_training_answer_formats,
)
from medarc_rl.verifiers.rewards import format_reward, multiple_choice_accuracy_reward
from verifiers.types import State


def test_strict_maybe_think_parser_rejects_unclosed_think() -> None:
    parser = StrictMaybeThinkParser()

    assert parser.parse("<think>unfinished") == ""
    assert parser.parse("<think>done</think>final") == "final"


def test_parser_bundle_returns_expected_reward_paths() -> None:
    xml_parser, xml_reward = build_parser_bundle(AnswerFormat.XML, use_think=False)
    json_parser, json_reward = build_parser_bundle(AnswerFormat.JSON, use_think=True)
    boxed_parser, boxed_reward = build_parser_bundle(AnswerFormat.BOXED, use_think=False)

    assert xml_parser.parse_answer("<answer>A</answer>") == "A"
    assert xml_reward([{"role": "assistant", "content": "<answer>A</answer>"}]) > 0.0
    assert json_parser.parse_answer('<think>x</think>{"answer":"B"}') == "B"
    assert json_reward([{"role": "assistant", "content": '{"answer":"B"}'}]) > 0.0
    assert boxed_parser.parse_answer(r"\boxed{C}") == "C"
    assert boxed_reward([{"role": "assistant", "content": r"\boxed{C}"}]) == 1.0


def test_seeded_training_format_selection_is_reproducible() -> None:
    formats = normalize_training_answer_formats(AnswerFormat.XML, "Random")

    first = [
        choose_training_answer_format(row_format_key=f"row-{idx}", train_answer_formats=formats, training_seed=7)
        for idx in range(6)
    ]
    second = [
        choose_training_answer_format(row_format_key=f"row-{idx}", train_answer_formats=formats, training_seed=7)
        for idx in range(6)
    ]

    assert first == second


def test_meta_parser_rejects_missing_and_unknown_formats() -> None:
    parser = MetaParser(use_think=False)

    try:
        parser.parse_for_row([{"role": "assistant", "content": "<answer>A</answer>"}], {})
    except ValueError as exc:
        assert "Missing answer_format" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing answer_format")

    try:
        parser.parse_for_format([{"role": "assistant", "content": "A"}], "yaml")
    except ValueError as exc:
        assert "Unsupported answer format" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported answer_format")


def test_rewards_reject_missing_answer_format_metadata() -> None:
    parser = MetaParser(use_think=False)

    try:
        multiple_choice_accuracy_reward(
            completion=[{"role": "assistant", "content": "<answer>A</answer>"}],
            answer="A",
            parser=build_parser_bundle(AnswerFormat.XML, use_think=False)[0],
            info={},
            meta_parser=parser,
        )
    except ValueError as exc:
        assert "Missing answer_format" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing answer_format")

    try:
        format_reward(
            completion=[{"role": "assistant", "content": "<answer>A</answer>"}],
            info=None,
            meta_parser=parser,
        )
    except ValueError as exc:
        assert "Missing info" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing info")


def _train_state_template(*, dataset_split: str = "train", enabled: bool = True) -> State:
    row = {
        "prompt": [
            {"role": "system", "content": "Please reason step by step, then give your final answer in <answer> tags."},
            {"role": "user", "content": "Question:Base question?\nA. Alpha\nB. Bravo\nC. Charlie\nD. Delta\nAnswer:"},
        ],
        "question": "Question:Base question?\nA. Alpha\nB. Bravo\nC. Charlie\nD. Delta\nAnswer:",
        "answer": "A",
        "example_id": "example-1",
        "info": {
            "answer_format": "xml",
            "row_format_key": "row-1",
            "dataset_split": dataset_split,
        },
        TRAIN_MCQ: TrainingMcq.from_dict_choices(
            question_data="Base question?",
            options={"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta"},
            answer="A",
        ).to_payload(),
        TRAIN_ANSWER_KEY: enabled,
    }
    return State(input=deepcopy(row))


def test_TRAIN_MCQ_from_dict_choices_sets_labels_and_answer_idx() -> None:
    TRAIN_MCQ = TrainingMcq.from_dict_choices(
        question_data="Base question?",
        options={"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta"},
        answer="C",
    )

    assert TRAIN_MCQ.labels == ("A", "B", "C", "D")
    assert TRAIN_MCQ.options == ("Alpha", "Bravo", "Charlie", "Delta")
    assert TRAIN_MCQ.answer_idx == 2


def test_TRAIN_MCQ_from_list_choices_sets_labels_and_answer_idx() -> None:
    TRAIN_MCQ = TrainingMcq.from_list_choices(
        question_data="Base question?",
        options=["Alpha", "Bravo", "Charlie", "Delta"],
        answer_idx=1,
        labels=["A", "B", "C", "D"],
    )

    assert TRAIN_MCQ.labels == ("A", "B", "C", "D")
    assert TRAIN_MCQ.options == ("Alpha", "Bravo", "Charlie", "Delta")
    assert TRAIN_MCQ.answer_idx == 1


def test_training_mcq_payload_normalizes_question_data_schema_for_mixed_rows() -> None:
    text_payload = TrainingMcq.from_dict_choices(
        question_data="Base question?",
        options={"A": "Alpha", "B": "Bravo"},
        answer="A",
    ).to_payload()
    dict_payload = TrainingMcq.from_dict_choices(
        question_data={"question": "Base question?", "context": ["ctx"]},
        options={"A": "Alpha", "B": "Bravo"},
        answer="A",
    ).to_payload()

    dataset = Dataset.from_list([{TRAIN_MCQ: text_payload}, {TRAIN_MCQ: dict_payload}])

    assert dataset.features[TRAIN_MCQ]["question_data"].dtype == "string"
    assert TrainingMcq.from_value(text_payload).question_data == "Base question?"
    assert TrainingMcq.from_value(dict_payload).question_data == {"question": "Base question?", "context": ["ctx"]}


def test_apply_train_answer_reshuffle_updates_answer_alignment(monkeypatch) -> None:
    def fake_randomize_multiple_choice(*, options, answer_choice, seed, **_):
        assert options == ["Alpha", "Bravo", "Charlie", "Delta"]
        assert answer_choice == 0
        assert seed == -1
        return ["Bravo", "Alpha", "Charlie", "Delta"], "B", 1

    monkeypatch.setattr(presentation, "randomize_multiple_choice", fake_randomize_multiple_choice)

    state = _train_state_template()
    original_input_question = state["input"]["question"]
    apply_train_answer_reshuffle(
        state,
        format_training_question=lambda TRAIN_MCQ, presented_options: (
            f"Question:{TRAIN_MCQ.question_data}\n"
            + "\n".join(f"{label}. {text}" for label, text in presented_options.items())
            + "\nAnswer:"
        ),
        use_think=False,
    )

    assert state["example_id"] == "example-1"
    assert state["answer"] == "B"
    assert state["info"]["answer_format"] == "xml"
    assert state["info"]["answer_text"] == state["info"]["options"][state["answer"]]
    assert state["info"]["options"]["B"] == "Alpha"
    assert state["input"]["question"] == original_input_question


def test_apply_train_answer_reshuffle_leaves_non_train_rows_unchanged() -> None:
    state = _train_state_template(dataset_split="eval")
    original_prompt = deepcopy(state["prompt"])
    original_info = deepcopy(state["info"])

    apply_train_answer_reshuffle(
        state,
        format_training_question=lambda *_: "changed",
        use_think=False,
    )

    assert state["prompt"] == original_prompt
    assert state["answer"] == "A"
    assert state["info"] == original_info
