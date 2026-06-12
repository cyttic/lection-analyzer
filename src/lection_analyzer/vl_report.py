"""Stage 4 — vl_report: analyze each keyframe into a structured visual Moment.

The VL backend reads one board frame at a time, anchored by the transcript window, and
returns JSON describing the visual structure (tables / formulas / schemes). Robust to
malformed JSON: a frame that fails to parse becomes a minimal Moment rather than crashing
the run.
"""

from __future__ import annotations

import json

from .backends.base import VLBackend, extract_json
from .config import Config
from .prompts import VL_SYSTEM, vl_prompt
from .schemas import (
    Keyframe,
    KeyframeIndex,
    Moment,
    Scheme,
    Table,
    VLReport,
)


def _moment_from_raw(kf: Keyframe, raw: dict) -> Moment:
    tables = [Table(**t) for t in raw.get("tables", []) if isinstance(t, dict)]
    schemes = [Scheme(**s) for s in raw.get("schemes", []) if isinstance(s, dict)]
    formulas = [str(f) for f in raw.get("formulas", []) if f]
    return Moment(
        timestamp_start=kf.timestamp,
        timestamp_end=kf.timestamp,
        frame_paths=[kf.path],
        kind=raw.get("kind", "other"),
        description=raw.get("description", ""),
        board_text=raw.get("board_text", ""),
        formulas=formulas,
        tables=tables,
        schemes=schemes,
        transcript_excerpt=kf.transcript_window,
    )


def run(cfg: Config, index: KeyframeIndex, vl: VLBackend) -> VLReport:
    cfg.ensure_dirs()
    if cfg.vl_report_json.exists():
        print(f"[vl_report] cached: {cfg.vl_report_json}")
        return VLReport.model_validate_json(cfg.vl_report_json.read_text("utf-8"))

    moments = []
    for i, kf in enumerate(index.frames, 1):
        abs_path = str(cfg.data_dir / kf.path)
        prompt = VL_SYSTEM + "\n\n" + vl_prompt(kf.transcript_window)
        print(f"[vl_report] {i}/{len(index.frames)} t={kf.timestamp:.1f}s ({kf.reason})")
        try:
            resp = vl.describe([abs_path], prompt)
            raw = json.loads(extract_json(resp))
            if isinstance(raw, list):  # model returned a list; take first object
                raw = next((x for x in raw if isinstance(x, dict)), {})
            moment = _moment_from_raw(kf, raw)
        except Exception as e:  # keep going; a bad frame shouldn't sink the run
            print(f"[vl_report]   ! parse/describe failed: {e}")
            moment = Moment(
                timestamp_start=kf.timestamp,
                timestamp_end=kf.timestamp,
                frame_paths=[kf.path],
                transcript_excerpt=kf.transcript_window,
            )
        moments.append(moment)

    report = VLReport(lecture=cfg.lecture, moments=moments)
    cfg.vl_report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"[vl_report] {len(moments)} moments -> {cfg.vl_report_json}")
    return report
