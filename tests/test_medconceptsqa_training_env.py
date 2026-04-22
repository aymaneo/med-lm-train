import asyncio
from copy import deepcopy
import importlib.util
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from medarc_verifiers.prompts import AnswerFormat
from medarc_rl.verifiers import TRAIN_MCQ, TRAIN_ANSWER_KEY

from tests.training_env_test_utils import completion_for_format, present_row, timing_state


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDCONCEPTSQA_PATH = REPO_ROOT / "environments" / "medconceptsqa" / "medconceptsqa.py"


def _load_medconceptsqa_module():
    spec = importlib.util.spec_from_file_location("local_medconceptsqa_env", MEDCONCEPTSQA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load medconceptsqa module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("local_medconceptsqa_env", module)
    spec.loader.exec_module(module)
    return module


def _make_row(concept_id: str, stem: str) -> dict:
    return {
        "concept_id": concept_id,
        "vocab": "icd10cm",
        "level": "easy",
        "question": stem,
        "option1": f"{stem} A",
        "option2": f"{stem} B",
        "option3": f"{stem} C",
        "option4": f"{stem} D",
        "answer_id": "A",
        "answer": f"{stem} A",
    }


def _sample_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "dev": Dataset.from_list(
                [
                    _make_row("sample-dev-1", "Sample dev 1?"),
                    _make_row("sample-dev-2", "Sample dev 2?"),
                ]
            ),
            "test": Dataset.from_list(
                [
                    _make_row("sample-test-1", "Sample test 1?"),
                    _make_row("sample-test-2", "Sample test 2?"),
                ]
            ),
        }
    )


def _full_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "dev": Dataset.from_list(
                [
                    _make_row("full-dev-1", "Full dev 1?"),
                    _make_row("full-dev-2", "Full dev 2?"),
                ]
            ),
            "test": Dataset.from_list(
                [_make_row("sample-test-1", "Sample test 1?"), _make_row("sample-test-2", "Sample test 2?")]
                + [_make_row(f"full-train-{idx}", f"Full train {idx}?") for idx in range(1, 10)]
            ),
        }
    )


def test_medconceptsqa_sample_training_is_full_minus_sample_eval(monkeypatch) -> None:
    medconceptsqa = _load_medconceptsqa_module()
    sample_ds = _sample_dataset()
    full_ds = _full_dataset()

    def fake_load_dataset(path: str, subset: str):
        assert subset == "icd10cm_easy"
        if path == "sameedkhan/medconceptsqa-sample_medarc_2k":
            return sample_ds
        if path == "ofir408/MedConceptsQA":
            return full_ds
        raise AssertionError(path)

    monkeypatch.setattr(medconceptsqa, "load_dataset", fake_load_dataset)

    env = medconceptsqa.load_environment(vocab="icd10cm_sample", num_few_shot=1)
    train_ids = {row["info"]["row_format_key"] for row in env.get_dataset()}
    eval_ids = {row["info"]["row_format_key"] for row in env.get_eval_dataset()}

    assert train_ids == {f"concept:full-train-{idx}" for idx in range(1, 10)}
    assert eval_ids == {"concept:sample-test-1", "concept:sample-test-2"}
    assert train_ids.isdisjoint(eval_ids)


def test_medconceptsqa_few_shot_and_train_formats_follow_selected_format(monkeypatch) -> None:
    medconceptsqa = _load_medconceptsqa_module()

    def fake_load_dataset(path: str, subset: str):
        assert subset == "icd10cm_easy"
        if path == "sameedkhan/medconceptsqa-sample_medarc_2k":
            return _sample_dataset()
        if path == "ofir408/MedConceptsQA":
            return _full_dataset()
        raise AssertionError(path)

    monkeypatch.setattr(medconceptsqa, "load_dataset", fake_load_dataset)

    env = medconceptsqa.load_environment(
        vocab="icd10cm_sample",
        num_few_shot=1,
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats="random",
        training_seed=11,
    )
    train_dataset = env.get_dataset()
    formats = {row["info"]["answer_format"] for row in train_dataset}

    assert formats == {"xml", "boxed", "json"}

    prompt_by_format = {}
    for row in train_dataset:
        prompt_by_format.setdefault(row["info"]["answer_format"], row["prompt"][1]["content"])

    assert "<answer>A</answer>" in prompt_by_format["xml"]
    assert "\\boxed{A}" in prompt_by_format["boxed"]
    assert '{"answer":"A"}' in prompt_by_format["json"]


def test_medconceptsqa_eval_remains_fixed_format_and_correctness_only(monkeypatch) -> None:
    medconceptsqa = _load_medconceptsqa_module()

    def fake_load_dataset(path: str, subset: str):
        assert subset == "icd10cm_easy"
        if path == "sameedkhan/medconceptsqa-sample_medarc_2k":
            return _sample_dataset()
        if path == "ofir408/MedConceptsQA":
            return _full_dataset()
        raise AssertionError(path)

    monkeypatch.setattr(medconceptsqa, "load_dataset", fake_load_dataset)

    env = medconceptsqa.load_environment(
        vocab="icd10cm_sample",
        use_think=False,
        answer_format=AnswerFormat.XML,
        train_answer_formats=[AnswerFormat.XML, AnswerFormat.JSON],
        training_seed=3,
    )
    eval_row = env.get_eval_dataset()[0]

    assert eval_row["info"]["answer_format"] == "xml"

    state = {
        "prompt": eval_row["prompt"],
        "completion": [{"role": "assistant", "content": "<answer>A</answer>"}],
        "answer": eval_row["answer"],
        "task": eval_row["task"],
        "info": eval_row["info"],
        "timing": timing_state(),
    }
    asyncio.run(env.rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert "format_reward" not in state["metrics"]


def test_medconceptsqa_non_sample_does_not_train_on_eval_split(monkeypatch) -> None:
    medconceptsqa = _load_medconceptsqa_module()

    full_ds = _full_dataset()

    def fake_load_dataset(path: str, subset: str):
        assert path == "ofir408/MedConceptsQA"
        assert subset == "icd10cm_easy"
        return full_ds

    monkeypatch.setattr(medconceptsqa, "load_dataset", fake_load_dataset)

    env = medconceptsqa.load_environment(vocab="icd10cm", num_few_shot=0)
    train_ids = {row["info"]["row_format_key"] for row in env.get_dataset()}
    eval_ids = {row["info"]["row_format_key"] for row in env.get_eval_dataset()}

    assert train_ids == {"concept:sample-test-1", "concept:sample-test-2"} | {
        f"concept:full-train-{idx}" for idx in range(1, 10)
    }
    assert eval_ids == {"concept:full-dev-1", "concept:full-dev-2"}
    assert train_ids.isdisjoint(eval_ids)


def test_medconceptsqa_training_shuffle_and_group_scoring(monkeypatch) -> None:
    medconceptsqa = _load_medconceptsqa_module()

    def fake_load_dataset(path: str, subset: str):
        assert subset == "icd10cm_easy"
        if path == "sameedkhan/medconceptsqa-sample_medarc_2k":
            return _sample_dataset()
        if path == "ofir408/MedConceptsQA":
            return _full_dataset()
        raise AssertionError(path)

    monkeypatch.setattr(medconceptsqa, "load_dataset", fake_load_dataset)

    env = medconceptsqa.load_environment(
        vocab="icd10cm_sample",
        num_few_shot=0,
        use_think=True,
        shuffle_answers=False,
        training_shuffle_answers=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats=[AnswerFormat.XML, AnswerFormat.BOXED, AnswerFormat.JSON],
        training_seed=5,
    )

    train_row = env.get_dataset()[0]
    eval_row = env.get_eval_dataset()[0]
    assert TRAIN_MCQ in train_row
    assert train_row[TRAIN_ANSWER_KEY] is True
    assert "options" not in train_row["info"]
    assert "options" not in eval_row["info"]

    original_prompt = deepcopy(train_row["prompt"])
    original_answer = train_row["answer"]

    presented_state = present_row(env, train_row)

    assert train_row["prompt"] == original_prompt
    assert train_row["answer"] == original_answer
    assert presented_state["info"]["answer_format"] == train_row["info"]["answer_format"]
    assert presented_state["info"]["answer_text"] == presented_state["info"]["options"][presented_state["answer"]]
    assert presented_state["prompt"][0]["content"] == train_row["prompt"][0]["content"]

    presented_state["completion"] = completion_for_format(
        presented_state["info"]["answer_format"],
        presented_state["answer"],
    )
    presented_state["timing"] = timing_state()
    asyncio.run(env.rubric.score_rollout(presented_state))

    assert presented_state["reward"] == 1.1

    states = []
    seen = set()
    for row in env.get_dataset():
        row_format = row["info"]["answer_format"]
        if row_format in seen:
            continue
        answer = row["answer"]
        completions = {
            "xml": [{"role": "assistant", "content": f"<think>x</think><answer>{answer}</answer>"}],
            "boxed": [{"role": "assistant", "content": f"<think>x</think>\\boxed{{{answer}}}"}],
            "json": [{"role": "assistant", "content": f'<think>x</think>{{"answer":"{answer}"}}'}],
        }
        states.append(
            {
                "prompt": row["prompt"],
                "completion": completions[row_format],
                "answer": answer,
                "task": row["task"],
                "info": row["info"],
                "timing": timing_state(),
                "trajectory": [],
            }
        )
        seen.add(row_format)

    asyncio.run(env.rubric.score_group(states))

    for state in states:
        assert state["reward"] == 1.1
        assert state["metrics"]["multiple_choice_accuracy_reward"] == 1.0
        assert state["metrics"]["format_reward"] == 1.0

    assert env.get_env_for_task("medconceptsqa") is env.train_env
