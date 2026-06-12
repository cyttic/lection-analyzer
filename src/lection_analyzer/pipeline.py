"""Pipeline orchestrator.

Runs the five stages in order. Every stage caches its artifact to disk and skips
work if the artifact already exists, so a crashed/long run resumes cheaply and you
can re-run a single stage by deleting just its artifact.

CLI:
    python -m lection_analyzer.pipeline --config config.yaml
    python -m lection_analyzer.pipeline --config config.yaml --only transcribe
    python -m lection_analyzer.pipeline --config config.yaml --from keyframes
"""

from __future__ import annotations

import argparse
from typing import List

from . import ingest, keyframes, synthesize, transcribe, vl_report
from .backends import build_backends
from .config import Config, load_config

STAGES = ["ingest", "transcribe", "keyframes", "vl_report", "synthesize"]


def run_pipeline(cfg: Config, stages: List[str]) -> None:
    cfg.ensure_dirs()

    transcript = None
    index = None
    report = None

    if "ingest" in stages:
        ingest.run(cfg)

    if "transcribe" in stages:
        transcript = transcribe.run(cfg)

    if "keyframes" in stages:
        if transcript is None:
            transcript = transcribe.run(cfg)  # cached load
        index = keyframes.run(cfg, transcript)

    # Backends are only needed (and only loaded) for the model stages.
    vl = llm = None
    if "vl_report" in stages or "synthesize" in stages:
        vl, llm = build_backends(cfg.backends)

    if "vl_report" in stages:
        if index is None:
            from .schemas import KeyframeIndex

            index = KeyframeIndex.model_validate_json(
                cfg.keyframe_index_json.read_text("utf-8")
            )
        report = vl_report.run(cfg, index, vl)

    if "synthesize" in stages:
        if report is None:
            report = _load_report(cfg)
        if transcript is None:
            transcript = transcribe.run(cfg)
        synthesize.run(cfg, report, transcript, llm)

    print("\n[pipeline] done.")


def _load_report(cfg: Config):
    from .schemas import VLReport

    return VLReport.model_validate_json(cfg.vl_report_json.read_text("utf-8"))


def _select_stages(only: str | None, start: str | None) -> List[str]:
    if only:
        if only not in STAGES:
            raise SystemExit(f"unknown stage '{only}'; choose from {STAGES}")
        return [only]
    if start:
        if start not in STAGES:
            raise SystemExit(f"unknown stage '{start}'; choose from {STAGES}")
        return STAGES[STAGES.index(start):]
    return list(STAGES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lecture Analyzer pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", help="run a single stage", choices=STAGES)
    ap.add_argument("--from", dest="start", help="run from this stage onward", choices=STAGES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    stages = _select_stages(args.only, args.start)
    print(f"[pipeline] lecture={cfg.lecture} stages={stages}")
    run_pipeline(cfg, stages)


if __name__ == "__main__":
    main()
