"""Stage 5 — synthesize: per-episode -> one TaskSpec JSON per task class.

This is the final stage. It makes ONE small LLM call per board episode (each ≈ an
independent task), turning that episode's structured visual data + local transcript
into TaskSpec(s). No global pass over the whole lecture. Each spec is written as both
``<task_class>.json`` (machine) and ``<task_class>.md`` (eyeball).
"""

from __future__ import annotations

import json
import re

from .backends.base import LLMBackend, extract_json
from .config import Config
from .prompts import SYN_SYSTEM, synthesis_prompt
from .schemas import Moment, TaskSource, TaskSpec, Transcript, VLReport


def _slugify(name: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or fallback


def _to_md(spec: TaskSpec) -> str:
    lines = [f"# {spec.title}"]
    if spec.title_source:
        lines.append(f"*source term:* {spec.title_source}")
    lines += ["", spec.description, "", "## Algorithm"]
    lines += [f"{i}. {s}" for i, s in enumerate(spec.algorithm_steps, 1)]
    if spec.formulas:
        lines += ["", "## Formulas"] + [f"- `{f}`" for f in spec.formulas]
    if spec.worked_example.given or spec.worked_example.calculation:
        lines += ["", "## Worked example", f"**Given:** {spec.worked_example.given}", ""]
        lines += [f"- {c}" for c in spec.worked_example.calculation]
        lines += ["", f"**Result:** {spec.worked_example.result}"]
    if spec.output_format.appearance or spec.output_format.template:
        lines += ["", "## Output format", spec.output_format.appearance, "",
                  "```", spec.output_format.template, "```"]
    if spec.source.timestamps:
        lines += ["", f"*source: {spec.source.lecture} @ {', '.join(spec.source.timestamps)}*"]
    return "\n".join(lines)


def _clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _spec_from_item(cfg: Config, item: dict, moment: Moment, idx: int) -> TaskSpec | None:
    """Build a validated TaskSpec from one raw model item, injecting the source from
    the episode itself (timestamps/frames are authoritative from the moment, not the LLM)."""
    if not isinstance(item, dict):
        return None
    item["source"] = TaskSource(
        lecture=cfg.lecture,
        timestamps=[f"{_clock(moment.timestamp_start)}-{_clock(moment.timestamp_end)}"],
        frames=list(moment.frame_paths),
    ).model_dump()
    item.setdefault("task_class", f"task_{idx}")
    item["task_class"] = _slugify(item["task_class"], f"task_{idx}")
    try:
        return TaskSpec.model_validate(item)
    except Exception as e:
        print(f"[synthesize]   ! skipping malformed task: {e}")
        return None


def run(cfg: Config, report: VLReport, transcript: Transcript, llm: LLMBackend) -> list[TaskSpec]:
    """One small LLM call PER episode (independent task), never the whole lecture at once.

    Each call sees only that episode's structured visual data + its local transcript, so
    the prompt is bounded regardless of lecture length — no O(n^2) attention blow-up, cheap
    enough to run locally. An episode may yield several tasks (multi-task board) or none
    (pure narration). Duplicate task slugs across episodes are merged at the end.
    """
    cfg.ensure_dirs()

    by_slug: dict[str, TaskSpec] = {}
    for i, moment in enumerate(report.moments, 1):
        moment_json = json.dumps(moment.model_dump(), ensure_ascii=False, indent=2)
        user = synthesis_prompt(moment.transcript_excerpt, moment_json)
        print(f"[synthesize] episode {i}/{len(report.moments)} "
              f"@ {_clock(moment.timestamp_start)} ({moment.kind})")
        try:
            resp = llm.complete(SYN_SYSTEM, user)
            raw = json.loads(extract_json(resp))
        except Exception as e:
            print(f"[synthesize]   ! call/parse failed, skipping episode: {e}")
            continue
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw if isinstance(raw, list) else []:
            spec = _spec_from_item(cfg, item, moment, i)
            if spec is None:
                continue
            if spec.task_class in by_slug:  # same task taught twice -> keep the richer one
                if len(spec.algorithm_steps) <= len(by_slug[spec.task_class].algorithm_steps):
                    continue
            by_slug[spec.task_class] = spec

    specs = list(by_slug.values())
    for spec in specs:
        (cfg.output_dir / f"{spec.task_class}.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (cfg.output_dir / f"{spec.task_class}.md").write_text(_to_md(spec), encoding="utf-8")

    print(f"[synthesize] wrote {len(specs)} task specs -> {cfg.output_dir}")
    for s in specs:
        print(f"  - {s.task_class}: {s.title}")
    return specs
