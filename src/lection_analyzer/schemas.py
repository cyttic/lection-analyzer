"""Pydantic models shared across pipeline stages.

These double as the on-disk JSON contract: every stage validates its output against
one of these models, so a downstream stage can trust the shape of what it reads.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# transcribe stage
# ---------------------------------------------------------------------------
class Segment(BaseModel):
    """One Whisper transcript segment."""

    start: float
    end: float
    text: str
    lang: Optional[str] = None  # detected language for the segment (he/ru/...)


class Transcript(BaseModel):
    lecture: str
    language_summary: dict = Field(default_factory=dict)  # {"he": 0.8, "ru": 0.2}
    segments: List[Segment] = Field(default_factory=list)

    def window(self, start: float, end: float, pad: float = 8.0) -> str:
        """Concatenated transcript text overlapping [start-pad, end+pad]."""
        lo, hi = start - pad, end + pad
        parts = [s.text.strip() for s in self.segments if s.end >= lo and s.start <= hi]
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# keyframes stage
# ---------------------------------------------------------------------------
class Keyframe(BaseModel):
    timestamp: float            # seconds into the video
    path: str                   # relative path under data_dir, e.g. frames/000123.jpg
    reason: str                 # why selected: "scene_cut" | "cue:<word>" | "interval"
    transcript_window: str = "" # spoken text around this frame (anchoring ground truth)


class KeyframeIndex(BaseModel):
    lecture: str
    frames: List[Keyframe] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# vl_report stage
# ---------------------------------------------------------------------------
class Table(BaseModel):
    caption: str = ""
    header: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)


class Scheme(BaseModel):
    caption: str = ""
    nodes: List[str] = Field(default_factory=list)
    edges: List[List[str]] = Field(default_factory=list)  # [["a","b"], ...]
    notes: str = ""


class Moment(BaseModel):
    """One analyzed visual moment, grouped from one or more keyframes."""

    timestamp_start: float
    timestamp_end: float
    frame_paths: List[str] = Field(default_factory=list)
    kind: str = "other"  # algorithm | worked_example | definition | table | scheme | other
    description: str = ""               # English summary of what is shown
    board_text: str = ""               # text read off the board (English, terms verbatim)
    formulas: List[str] = Field(default_factory=list)  # LaTeX
    tables: List[Table] = Field(default_factory=list)
    schemes: List[Scheme] = Field(default_factory=list)
    transcript_excerpt: str = ""


class VLReport(BaseModel):
    lecture: str
    moments: List[Moment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# synthesize stage — the final artifact your Agent loads
# ---------------------------------------------------------------------------
class WorkedExample(BaseModel):
    given: str = ""
    calculation: List[str] = Field(default_factory=list)  # numeric steps
    result: str = ""


class OutputFormat(BaseModel):
    appearance: str = ""  # exact board layout (table/formula as shown)
    template: str = ""    # fill-in template the Agent should emit


class TaskSource(BaseModel):
    lecture: str
    timestamps: List[str] = Field(default_factory=list)  # ["00:12:30-00:15:10"]
    frames: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    """A single task class, ready to register as an Agent sub-agent."""

    task_class: str                       # machine slug, e.g. "modular_inverse"
    title: str                            # English title
    title_source: str = ""               # original Hebrew/Russian term, verbatim
    description: str = ""                 # when/why to use it
    trigger_keywords: List[str] = Field(default_factory=list)
    algorithm_steps: List[str] = Field(default_factory=list)
    formulas: List[str] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    schemes: List[Scheme] = Field(default_factory=list)
    worked_example: WorkedExample = Field(default_factory=WorkedExample)
    output_format: OutputFormat = Field(default_factory=OutputFormat)
    source: TaskSource
