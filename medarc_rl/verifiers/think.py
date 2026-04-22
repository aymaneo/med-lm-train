from __future__ import annotations


def strict_think_strip(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" not in text:
        return text.strip()
    return text.split("</think>", 1)[1].strip()
