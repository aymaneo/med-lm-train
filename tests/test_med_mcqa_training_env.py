import asyncio
from copy import deepcopy
import importlib.util
import sys
from pathlib import Path

from datasets import Dataset
from medarc_verifiers.prompts import AnswerFormat
from medarc_rl.verifiers import TRAIN_MCQ, TRAIN_ANSWER_KEY

from tests.training_env_test_utils import completion_for_format, present_row, timing_state


REPO_ROOT = Path(__file__).resolve().parents[1]
MED_MCQA_PATH = REPO_ROOT / "environments" / "med_mcqa" / "med_mcqa.py"


def _load_med_mcqa_module():
    spec = importlib.util.spec_from_file_location("local_med_mcqa_env", MED_MCQA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load med_mcqa module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("local_med_mcqa_env", module)
    spec.loader.exec_module(module)
    return module


def _fake_split(prefix: str, size: int) -> Dataset:
    rows = []
    for idx in range(size):
        rows.append(
            {
                "question": f"{prefix} question {idx}?",
                "opa": f"{prefix} A {idx}",
                "opb": f"{prefix} B {idx}",
                "opc": f"{prefix} C {idx}",
                "opd": f"{prefix} D {idx}",
                "cop": 1,
            }
        )
    return Dataset.from_list(rows)


def test_med_mcqa_training_dataset_supports_xml_boxed_and_json(monkeypatch) -> None:
    med_mcqa = _load_med_mcqa_module()

    def fake_load_dataset(name: str, split: str):
        assert name == "lighteval/med_mcqa"
        return _fake_split(split, 9 if split == "train" else 1)

    monkeypatch.setattr(med_mcqa, "load_dataset", fake_load_dataset)

    env = med_mcqa.load_environment(
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats="random",
        training_seed=11,
    )
    train_dataset = env.get_dataset()
    formats = {row["info"]["answer_format"] for row in train_dataset}

    assert formats == {"xml", "boxed", "json"}


def test_med_mcqa_training_shuffle_can_differ_from_eval_shuffle(monkeypatch) -> None:
    med_mcqa = _load_med_mcqa_module()

    def fake_load_dataset(name: str, split: str):
        assert name == "lighteval/med_mcqa"
        return _fake_split(split, 2 if split == "train" else 1)

    monkeypatch.setattr(med_mcqa, "load_dataset", fake_load_dataset)

    env = med_mcqa.load_environment(
        shuffle_answers=False,
        training_shuffle_answers=True,
        training_seed=23,
    )

    train_row = env.get_dataset()[0]
    eval_row = env.get_eval_dataset()[0]

    assert TRAIN_MCQ in train_row
    assert train_row[TRAIN_ANSWER_KEY] is True
    assert "options" not in train_row["info"]
    assert "options" not in eval_row["info"]

    original_prompt = deepcopy(train_row["prompt"])
    original_answer = train_row["answer"]

    state = present_row(env, train_row)

    assert train_row["prompt"] == original_prompt
    assert train_row["answer"] == original_answer
    assert state["info"]["answer_format"] == train_row["info"]["answer_format"]
    assert state["info"]["answer_text"] == state["info"]["options"][state["answer"]]
    assert state["prompt"][0]["content"] == train_row["prompt"][0]["content"]

    state["completion"] = completion_for_format(state["info"]["answer_format"], state["answer"])
    state["timing"] = timing_state()
    asyncio.run(env.rubric.score_rollout(state))

    assert state["reward"] == 1.1


def test_med_mcqa_eval_remains_fixed_format_and_correctness_only(monkeypatch) -> None:
    med_mcqa = _load_med_mcqa_module()

    def fake_load_dataset(name: str, split: str):
        assert name == "lighteval/med_mcqa"
        return _fake_split(split, 2 if split == "train" else 1)

    monkeypatch.setattr(med_mcqa, "load_dataset", fake_load_dataset)

    env = med_mcqa.load_environment(
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


def test_med_mcqa_group_scoring_and_task_routing(monkeypatch) -> None:
    med_mcqa = _load_med_mcqa_module()

    def fake_load_dataset(name: str, split: str):
        assert name == "lighteval/med_mcqa"
        return _fake_split(split, 9 if split == "train" else 1)

    monkeypatch.setattr(med_mcqa, "load_dataset", fake_load_dataset)

    env = med_mcqa.load_environment(
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats=[AnswerFormat.XML, AnswerFormat.BOXED, AnswerFormat.JSON],
        training_seed=5,
    )

    completions = {
        "xml": [{"role": "assistant", "content": "<think>x</think><answer>A</answer>"}],
        "boxed": [{"role": "assistant", "content": "<think>x</think>\\boxed{A}"}],
        "json": [{"role": "assistant", "content": '<think>x</think>{"answer":"A"}'}],
    }

    states = []
    seen = set()
    for row in env.get_dataset():
        row_format = row["info"]["answer_format"]
        if row_format in seen:
            continue
        states.append(
            {
                "prompt": row["prompt"],
                "completion": completions[row_format],
                "answer": row["answer"],
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

    assert env.get_env_for_task("med_mcqa") is env.train_env
