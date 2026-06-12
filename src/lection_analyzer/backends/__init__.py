"""Model backends. Selected per-stage via config (local | claude)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import LLMBackend, VLBackend


def build_backends(backends_cfg: Dict[str, Any]) -> Tuple[VLBackend, LLMBackend]:
    """Construct the VL and synthesis backends named in config.

    The local backend is a single Qwen2.5-VL instance shared for both roles, so we
    only build/load it once even when both stages are set to ``local``.
    """
    vl_name = backends_cfg.get("vl_backend", "local")
    syn_name = backends_cfg.get("synthesis_backend", "local")

    local_shared = None

    def get_local():
        nonlocal local_shared
        if local_shared is None:
            from .local_qwen import LocalQwenBackend

            local_shared = LocalQwenBackend(backends_cfg.get("local", {}))
        return local_shared

    def get_claude():
        from .claude_api import ClaudeBackend

        return ClaudeBackend(backends_cfg.get("claude", {}))

    vl: VLBackend = get_local() if vl_name == "local" else get_claude()
    llm: LLMBackend = get_local() if syn_name == "local" else get_claude()
    return vl, llm
