"""Backend interfaces.

Two narrow capabilities the pipeline depends on:

* ``VLBackend.describe`` — look at one or more board images plus the spoken-text
  context and return a JSON string describing the visual structure.
* ``LLMBackend.complete`` — text-only chat completion used by the synthesis stage.

A single concrete backend (local Qwen, or Claude) may implement both.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Protocol, runtime_checkable


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


def _basic_repair(s: str) -> str:
    """Best-effort fixes for the ways local models break JSON.

    Board content is LaTeX-heavy (\\frac, \\gcd, \\underline ...). Models emit those
    backslashes raw, which is invalid JSON. We escape every backslash that isn't already
    ``\\\\``/``\\"``/``\\/`` — i.e. treat backslashes as literal LaTeX rather than control
    escapes (so \\frac stays \\frac instead of becoming a form-feed).
    """
    s = re.sub(r'\\(?![\\/"])', r"\\\\", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # drop trailing commas before } or ]
    return s


def loads_lenient(text: str) -> Any:
    """Parse model JSON, tolerating LaTeX backslashes, trailing commas, and truncation.

    Tries stdlib first, then ``json_repair`` (handles unterminated strings/braces from a
    cut-off response), then a regex repair — so a slightly malformed response still yields
    usable data instead of dropping the whole episode.
    """
    s = extract_json(text)
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        from json_repair import repair_json

        return json.loads(repair_json(s))
    except Exception:
        pass
    return json.loads(_basic_repair(s))  # last resort; may still raise, caught by caller
