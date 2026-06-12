"""Backend interfaces.

Two narrow capabilities the pipeline depends on:

* ``VLBackend.describe`` — look at one or more board images plus the spoken-text
  context and return a JSON string describing the visual structure.
* ``LLMBackend.complete`` — text-only chat completion used by the synthesis stage.

A single concrete backend (local Qwen, or Claude) may implement both.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class VLBackend(Protocol):
    def describe(self, image_paths: List[str], prompt: str) -> str:
        """Return the model's raw text response for the given image(s) + prompt."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response for a text-only system+user prompt."""
        ...


def extract_json(text: str) -> str:
    """Best-effort pull of a JSON object/array out of a model response.

    Models often wrap JSON in prose or ```json fences. This returns the substring
    from the first ``{``/``[`` to its matching close, or the original text if none.
    """
    fences = ("```json", "```JSON", "```")
    for f in fences:
        if f in text:
            text = text.split(f, 1)[1]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
            break

    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return text[start:].strip()
