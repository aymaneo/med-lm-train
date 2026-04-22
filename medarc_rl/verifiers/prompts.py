from __future__ import annotations

from medarc_verifiers.prompts import AnswerFormat, THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT

from .formats import normalize_answer_format

JSON_SYSTEM_PROMPT = (
    'Please reason step by step, then respond with a JSON object matching {"answer": "<final_answer>"}.'
)
THINK_JSON_SYSTEM_PROMPT = (
    'Think step-by-step inside <think>...</think> tags. Then respond with a JSON object matching {"answer": "<final_answer>"}.'
)


def get_system_prompt(answer_format: AnswerFormat | str, *, use_think: bool) -> str:
    normalized = normalize_answer_format(answer_format)
    if normalized == AnswerFormat.XML:
        return THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT
    if normalized == AnswerFormat.BOXED:
        return THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
    if normalized == AnswerFormat.JSON:
        return THINK_JSON_SYSTEM_PROMPT if use_think else JSON_SYSTEM_PROMPT
    raise ValueError(f"Unsupported answer format: {answer_format}")
