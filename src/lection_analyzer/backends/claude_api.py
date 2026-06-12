"""Claude backend — vision (VL) + text (synthesis). Requires ANTHROPIC_API_KEY.

Implements both VLBackend.describe and LLMBackend.complete, so it can be selected for
either or both stages via config.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, List


class ClaudeBackend:
    def __init__(self, cfg: Dict[str, Any]):
        from anthropic import Anthropic  # local import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "claude backend selected but ANTHROPIC_API_KEY is not set in the environment"
            )
        self.client = Anthropic(api_key=api_key)
        self.model = cfg.get("model", "claude-opus-4-8")
        self.max_tokens = int(cfg.get("max_tokens", 4096))

    # -- VLBackend ------------------------------------------------------------
    def describe(self, image_paths: List[str], prompt: str) -> str:
        content: List[Dict[str, Any]] = []
        for p in image_paths:
            media_type = mimetypes.guess_type(p)[0] or "image/jpeg"
            with open(p, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        content.append({"type": "text", "text": prompt})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    # -- LLMBackend -----------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
