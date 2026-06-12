"""Stage 5 — synthesize: moments + transcript -> one TaskSpec JSON per task class.

This is the final stage. The synthesis LLM clusters the visual moments into task
classes and emits the structured specs your Agent loads. Each spec is written as both
``<task_class>.json`` (machine) and ``<task_class>.md`` (eyeball).
"""

from __future__ import annotations

import json
import re

from .backends.base import LLMBackend, extract_json
from .config import Config
from .prompts import SYN_SYSTEM, synthesis_prompt
from .schemas import TaskSource, TaskSpec, Transcript, VLReport


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


def run(cfg: Config, report: VLReport, transcript: Transcript, llm: LLMBackend) -> list[TaskSpec]:
    cfg.ensure_dirs()
    moments_json = json.dumps(
        [m.model_dump() for m in report.moments], ensure_ascii=False, indent=2
    )
    transcript_text = "\n".join(s.text.strip() for s in transcript.segments)
    user = synthesis_prompt(transcript_text, moments_json)

    print(f"[synthesize] clustering {len(report.moments)} moments into task classes...")
    resp = llm.complete(SYN_SYSTEM, user)
    raw = json.loads(extract_json(resp))
    if isinstance(raw, dict):
        raw = [raw]

    specs: list[TaskSpec] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        src = item.get("source", {}) or {}
        item["source"] = TaskSource(
            lecture=cfg.lecture,
            timestamps=src.get("timestamps", []),
            frames=src.get("frames", []),
        ).model_dump()
        item.setdefault("task_class", f"task_{i}")
        item["task_class"] = _slugify(item["task_class"], f"task_{i}")
        try:
            spec = TaskSpec.model_validate(item)
        except Exception as e:
            print(f"[synthesize]   ! skipping malformed task class #{i}: {e}")
            continue
        specs.append(spec)

    for spec in specs:
        (cfg.output_dir / f"{spec.task_class}.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (cfg.output_dir / f"{spec.task_class}.md").write_text(_to_md(spec), encoding="utf-8")

    print(f"[synthesize] wrote {len(specs)} task specs -> {cfg.output_dir}")
    for s in specs:
        print(f"  - {s.task_class}: {s.title}")
    return specs
