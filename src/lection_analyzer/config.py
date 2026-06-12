"""Config loading and run paths.

A single Config object is threaded through the pipeline. It resolves all the
per-run directories so stages never hard-code paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, raw: Dict[str, Any], root: Path):
        self.raw = raw
        self.root = root
        self.lecture: str = raw.get("lecture", "lecture")

        paths = raw.get("paths", {})
        self.data_dir = root / paths.get("data_dir", "data") / self.lecture
        self.output_dir = root / paths.get("output_dir", "output") / self.lecture
        self.frames_dir = self.data_dir / "frames"

    # -- convenience accessors -------------------------------------------------
    @property
    def ingest(self) -> Dict[str, Any]:
        return self.raw.get("ingest", {})

    @property
    def transcribe(self) -> Dict[str, Any]:
        return self.raw.get("transcribe", {})

    @property
    def keyframes(self) -> Dict[str, Any]:
        return self.raw.get("keyframes", {})

    @property
    def backends(self) -> Dict[str, Any]:
        return self.raw.get("backends", {})

    # -- artifact paths --------------------------------------------------------
    @property
    def raw_video(self) -> Path:
        return self.data_dir / "raw.mp4"

    @property
    def audio_wav(self) -> Path:
        return self.data_dir / "audio.wav"

    @property
    def transcript_json(self) -> Path:
        return self.data_dir / "transcript.json"

    @property
    def transcript_srt(self) -> Path:
        return self.data_dir / "transcript.srt"

    @property
    def transcript_txt(self) -> Path:
        return self.data_dir / "transcript.txt"

    @property
    def keyframe_index_json(self) -> Path:
        return self.data_dir / "frames_index.json"

    @property
    def vl_report_json(self) -> Path:
        return self.data_dir / "vl_report.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(raw, root=path.resolve().parent)
