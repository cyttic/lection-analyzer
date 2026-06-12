"""Offline smoke test: exercises every stage's logic with fake backends and a
synthetic transcript/keyframe index. No GPU, no models, no video needed.

Run: PYTHONPATH=src python scripts/smoke_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lection_analyzer import synthesize, vl_report  # noqa: E402
from lection_analyzer.backends.base import extract_json  # noqa: E402
from lection_analyzer.config import load_config  # noqa: E402
from lection_analyzer.schemas import (  # noqa: E402
    Keyframe,
    KeyframeIndex,
    Segment,
    Transcript,
)

# --- 1. extract_json robustness -------------------------------------------------
fenced = "prose before\n```json\n{\"a\": [1, 2], \"b\": {\"c\": 3}}\n```\nafter"
assert json.loads(extract_json(fenced)) == {"a": [1, 2], "b": {"c": 3}}
arr = "here [ {\"x\": 1}, {\"y\": 2} ] done"
assert json.loads(extract_json(arr)) == [{"x": 1}, {"y": 2}]
print("extract_json: ok")

# --- 2. fake backends -----------------------------------------------------------
class FakeVL:
    def describe(self, image_paths, prompt):
        return json.dumps({
            "kind": "algorithm",
            "description": "Euclid's algorithm for GCD",
            "board_text": "gcd(a,b)=gcd(b, a mod b)",
            "formulas": ["\\gcd(a,b)=\\gcd(b, a \\bmod b)"],
            "tables": [{"caption": "trace", "header": ["a", "b"], "rows": [["48", "18"], ["18", "12"]]}],
            "schemes": [],
        })


class FakeLLM:
    def complete(self, system, user):
        return json.dumps([{
            "task_class": "Euclid GCD",
            "title": "Greatest Common Divisor via Euclid",
            "title_source": "אלגוריתם אוקלידס",
            "description": "Find gcd of two integers.",
            "trigger_keywords": ["gcd", "מחלק משותף"],
            "algorithm_steps": ["Divide a by b, take remainder r", "Replace (a,b)=(b,r)", "Repeat until r=0"],
            "formulas": ["\\gcd(a,b)=\\gcd(b, a \\bmod b)"],
            "tables": [], "schemes": [],
            "worked_example": {"given": "a=48, b=18", "calculation": ["48 mod 18 = 12", "18 mod 12 = 6", "12 mod 6 = 0"], "result": "gcd = 6"},
            "output_format": {"appearance": "two-column trace table", "template": "gcd({a},{b}) = {result}"},
            "source": {"timestamps": ["00:01-00:03"], "frames": ["frames/00001000.jpg"]},
        }])


# --- 3. run vl_report + synthesize on synthetic inputs -------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    # minimal config pointing into the temp dir
    cfg_text = (
        "lecture: lecT\n"
        f"paths: {{data_dir: {root}/data, output_dir: {root}/output}}\n"
        "backends: {vl_backend: local, synthesis_backend: local}\n"
    )
    cfg_path = root / "config.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()

    transcript = Transcript(
        lecture="lecT",
        segments=[Segment(start=0, end=3, text="היום נלמד אלגוריתם אוקלידס", lang="he")],
    )
    # fake two episode frames so paths resolve; both teach the same task -> must merge to 1
    (cfg.frames_dir / "00001000.jpg").write_bytes(b"\xff\xd8\xff")
    (cfg.frames_dir / "00009000.jpg").write_bytes(b"\xff\xd8\xff")
    index = KeyframeIndex(lecture="lecT", frames=[
        Keyframe(timestamp=1.0, path="frames/00001000.jpg", reason="episode_end",
                 transcript_window="אלגוריתם אוקלידס"),
        Keyframe(timestamp=9.0, path="frames/00009000.jpg", reason="episode_end",
                 transcript_window="עוד דוגמה לאלגוריתם אוקלידס"),
    ])

    report = vl_report.run(cfg, index, FakeVL())
    assert len(report.moments) == 2
    assert report.moments[0].kind == "algorithm"
    assert report.moments[0].tables[0].rows[0] == ["48", "18"]
    print("vl_report: ok")

    specs = synthesize.run(cfg, report, transcript, FakeLLM())
    assert len(specs) == 1, f"two episodes of the same task should merge to 1, got {len(specs)}"
    s0 = specs[0]
    assert s0.task_class == "euclid_gcd", s0.task_class  # slugified
    assert s0.source.lecture == "lecT"  # injected
    assert s0.source.timestamps == ["00:01-00:01"], s0.source.timestamps  # from the moment, not LLM
    assert s0.source.frames == ["frames/00001000.jpg"], s0.source.frames
    assert s0.worked_example.result == "gcd = 6"
    out_json = cfg.output_dir / "euclid_gcd.json"
    out_md = cfg.output_dir / "euclid_gcd.md"
    assert out_json.exists() and out_md.exists()
    print("synthesize: ok")
    print("\nGenerated TaskSpec JSON:\n", out_json.read_text()[:400], "...")

print("\nALL SMOKE TESTS PASSED")
