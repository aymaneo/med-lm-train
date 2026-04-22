from __future__ import annotations

from typing import Any, Callable

import verifiers as vf

from .presentation import TrainingMcq, apply_train_answer_reshuffle


class TrainEvalRoutingRubric(vf.Rubric):
    def __init__(
        self,
        *,
        train_rubric: vf.Rubric,
        eval_rubric: vf.Rubric,
        split_field: str = "dataset_split",
    ) -> None:
        super().__init__(funcs=[], parser=train_rubric.parser)
        self.train_rubric = train_rubric
        self.eval_rubric = eval_rubric
        self.split_field = split_field

    def _select_rubric(self, state: dict[str, Any]) -> vf.Rubric:
        info = state.get("info") or {}
        if isinstance(info, dict) and info.get(self.split_field) == "eval":
            return self.eval_rubric
        return self.train_rubric

    async def score_rollout(self, state):
        await self._select_rubric(state).score_rollout(state)

    async def dummy_score_rollout(self, state):
        await self._select_rubric(state).dummy_score_rollout(state)

    async def score_group(self, states):
        partitions: dict[vf.Rubric, list[dict[str, Any]]] = {}
        for state in states:
            rubric = self._select_rubric(state)
            partitions.setdefault(rubric, []).append(state)
        for rubric, rubric_states in partitions.items():
            await rubric.score_group(rubric_states)

    async def dummy_score_group(self, states):
        partitions: dict[vf.Rubric, list[dict[str, Any]]] = {}
        for state in states:
            rubric = self._select_rubric(state)
            partitions.setdefault(rubric, []).append(state)
        for rubric, rubric_states in partitions.items():
            await rubric.dummy_score_group(rubric_states)

    async def cleanup(self, state):
        await self._select_rubric(state).cleanup(state)

    async def teardown(self):
        await self.train_rubric.teardown()
        await self.eval_rubric.teardown()


class TrainEvalRoutingEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        *,
        train_env: vf.Environment,
        eval_env: vf.Environment,
        parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        env_id: str | None = None,
        env_args: dict[str, Any] | None = None,
        split_field: str = "dataset_split",
        use_think: bool = False,
        format_training_question: Callable[[TrainingMcq, dict[str, str]], str] | None = None,
    ) -> None:
        self.train_env = train_env
        self.eval_env = eval_env
        self.split_field = split_field
        self.use_think = use_think
        self.format_training_question = format_training_question
        resolved_parser = parser
        if rubric is not None:
            resolved_parser = rubric.parser
        elif resolved_parser is None:
            resolved_parser = eval_env.parser
        super().__init__(
            dataset=lambda: self._require_dataset(self.train_env.build_dataset(), "train_env did not provide a dataset"),
            eval_dataset=lambda: self._require_dataset(
                self.eval_env.build_eval_dataset(),
                "eval_env did not provide an eval_dataset",
            ),
            system_prompt=None,
            parser=resolved_parser,
            rubric=rubric or eval_env.rubric,
            env_id=env_id or train_env.env_id or eval_env.env_id,
            env_args=env_args or {},
            score_rollouts=train_env.score_rollouts,
            pass_threshold=train_env.pass_threshold,
        )

    @staticmethod
    def _require_dataset(dataset, error_message: str):
        if dataset is None:
            raise ValueError(error_message)
        return dataset

    def _select_env_for_state(self, state: dict[str, Any]) -> vf.Environment:
        info = state.get("info") or {}
        if isinstance(info, dict) and info.get(self.split_field) == "eval":
            return self.eval_env
        return self.train_env

    async def setup_state(self, state):
        state = apply_train_answer_reshuffle(
            state,
            format_training_question=self.format_training_question,
            use_think=self.use_think,
        )
        return await self._select_env_for_state(state).setup_state(state)

    def get_env_for_task(self, task_name: str) -> vf.Environment:
        valid_names = {
            self.env_id,
            self.train_env.env_id,
            self.eval_env.env_id,
            "default",
            "",
        }
        if task_name in valid_names:
            return self.train_env
        raise KeyError(f"Unknown task: {task_name}")
