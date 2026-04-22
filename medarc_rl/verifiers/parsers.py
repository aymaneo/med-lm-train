from __future__ import annotations

from typing import Any, Callable

import verifiers as vf
from medarc_verifiers.parsers import JSONParser, XMLParser
from medarc_verifiers.prompts import AnswerFormat
from verifiers.utils.data_utils import extract_boxed_answer

from .formats import normalize_answer_format
from .rewards import boxed_format_reward


class StrictMaybeThinkParser(vf.MaybeThinkParser):
    def parse(self, text: str) -> str:
        if "<think>" in text and "</think>" not in text:
            return ""
        return super().parse(text)


def build_parser_bundle(
    answer_format: AnswerFormat | str,
    *,
    use_think: bool,
) -> tuple[vf.Parser, Callable[..., float]]:
    normalized = normalize_answer_format(answer_format)

    if normalized == AnswerFormat.XML:
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = XMLParser(fields=parser_fields, answer_field="answer")
        return parser, parser.get_format_reward_func()

    if normalized == AnswerFormat.JSON:
        extract_fn: Callable[[str], str] = str if not use_think else StrictMaybeThinkParser().parse
        parser = JSONParser(fields=["answer"], answer_field="answer", extract_fn=extract_fn)
        return parser, parser.get_format_reward_func()

    if normalized == AnswerFormat.BOXED:
        def extract_fn(text: str) -> str:
            return extract_boxed_answer(text, strict=True)

        if use_think:
            extract_fn = StrictMaybeThinkParser(extract_fn).parse
        parser = vf.Parser(extract_fn=extract_fn)
        return parser, boxed_format_reward(use_think=use_think)

    raise ValueError(f"Unsupported answer format: {answer_format}")


class MetaParser:
    def __init__(self, *, use_think: bool, formats: list[AnswerFormat] | None = None) -> None:
        supported_formats = formats or list(AnswerFormat)
        self.parsers: dict[AnswerFormat, vf.Parser] = {}
        self.format_rewards: dict[AnswerFormat, Callable[..., float]] = {}
        for answer_format in supported_formats:
            parser, format_reward = build_parser_bundle(answer_format, use_think=use_think)
            self.parsers[answer_format] = parser
            self.format_rewards[answer_format] = format_reward

    def _resolve_format(self, answer_format: AnswerFormat | str | None) -> AnswerFormat:
        if answer_format is None:
            raise ValueError("Missing answer_format in row metadata")
        normalized = normalize_answer_format(answer_format)
        if normalized not in self.parsers:
            raise ValueError(f"Unsupported answer format: {answer_format}")
        return normalized

    def parse_for_format(self, completion: Any, answer_format: AnswerFormat | str) -> str | None:
        resolved = self._resolve_format(answer_format)
        return self.parsers[resolved].parse_answer(completion)

    def parse_for_row(self, completion: Any, info: dict[str, Any] | None) -> str | None:
        row_format = None if info is None else info.get("answer_format")
        return self.parse_for_format(completion, row_format)

    def format_reward_for_row(self, completion: Any, info: dict[str, Any] | None) -> float:
        row_format = None if info is None else info.get("answer_format")
        resolved = self._resolve_format(row_format)
        return float(self.format_rewards[resolved](completion))
