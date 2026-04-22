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
MEDQA_PATH = REPO_ROOT / "environments" / "medqa" / "medqa.py"


def _load_medqa_module():
    spec = importlib.util.spec_from_file_location("local_medqa_env", MEDQA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load medqa module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("local_medqa_env", module)
    spec.loader.exec_module(module)
    return module


def _fake_medqa_dataset() -> DatasetDict:
    train_rows = []
    for idx in range(9):
        train_rows.append(
            {
                "id": f"train-{idx}",
                "question": f"Question {idx}?",
                "options": {
                    "A": f"Alpha {idx}",
                    "B": f"Bravo {idx}",
                    "C": f"Charlie {idx}",
                    "D": f"Delta {idx}",
                },
                "answer_idx": "A",
            }
        )
    test_rows = [
        {
            "id": "test-0",
            "question": "Eval question?",
            "options": {"A": "Eval A", "B": "Eval B", "C": "Eval C", "D": "Eval D"},
            "answer_idx": "B",
        }
    ]
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def test_medqa_training_dataset_supports_xml_boxed_and_json(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment(
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats="Random",
        training_seed=11,
    )
    train_dataset = env.get_dataset()
    formats = {row["info"]["answer_format"] for row in train_dataset}

    assert formats == {"xml", "boxed", "json"}

    prompt_by_format = {}
    for row in train_dataset:
        prompt_by_format.setdefault(row["info"]["answer_format"], row["prompt"][0]["content"])

    assert "<answer>" in prompt_by_format["xml"]
    assert "\\boxed{}" in prompt_by_format["boxed"]
    assert '{"answer": "<final_answer>"}' in prompt_by_format["json"]


def test_medqa_eval_remains_fixed_format_and_correctness_only(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment(
        use_think=False,
        answer_format=AnswerFormat.XML,
        train_answer_formats=[AnswerFormat.XML, AnswerFormat.JSON],
        training_seed=3,
    )

    eval_row = env.get_eval_dataset()[0]
    assert eval_row["info"]["answer_format"] == "xml"
    assert eval_row["prompt"][0]["content"].startswith("Please reason step by step")

    state = {
        "prompt": eval_row["prompt"],
        "completion": [{"role": "assistant", "content": "<answer>B</answer>"}],
        "answer": eval_row["answer"],
        "task": eval_row["task"],
        "info": eval_row["info"],
        "timing": timing_state(),
    }
    asyncio.run(env.rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert "format_reward" not in state["metrics"]


def test_medqa_training_shuffle_can_differ_from_eval_shuffle(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment(
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


def test_medqa_training_scoring_routes_by_row_format(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment(
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

    checked_formats = set()
    for row in env.get_dataset():
        row_format = row["info"]["answer_format"]
        if row_format in checked_formats:
            continue
        state = {
            "prompt": row["prompt"],
            "completion": completions[row_format],
            "answer": row["answer"],
            "task": row["task"],
            "info": row["info"],
            "timing": timing_state(),
        }
        asyncio.run(env.rubric.score_rollout(state))
        assert state["reward"] == 1.1
        checked_formats.add(row_format)

    assert checked_formats == {"xml", "boxed", "json"}


def test_medqa_group_scoring_routes_training_states(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment(
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


def test_medqa_get_env_for_task_returns_training_env(monkeypatch) -> None:
    medqa = _load_medqa_module()
    monkeypatch.setattr(medqa, "load_dataset", lambda _: _fake_medqa_dataset())

    env = medqa.load_environment()

    routed_env = env.get_env_for_task("medqa")
    assert routed_env is env.train_env
    reward_names = routed_env.rubric._get_reward_func_names()
    assert "multiple_choice_accuracy_reward" in reward_names
    assert "format_reward" in reward_names
