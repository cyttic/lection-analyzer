"""Pydantic models shared across pipeline stages.

These double as the on-disk JSON contract: every stage validates its output against
one of these models, so a downstream stage can trust the shape of what it reads.

Local VL/LLM models are sloppy about types — they'll return a dict where we asked for a
string, or numbers where we asked for text. Rather than drop a whole episode over that,
the string-ish fields use ``Str``/``StrList``/``StrGrid`` which *coerce* such values
(dict/list -> compact JSON, number -> str) instead of rejecting them.
"""

from __future__ import annotations

import json
from typing import Annotated, List, Optional

from pydantic import BaseModel, BeforeValidator, Field


# ---------------------------------------------------------------------------
# tolerant coercion helpers
# ---------------------------------------------------------------------------
def _as_str(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v if isinstance(v, str) else str(v)


def _as_str_list(v):
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return [_as_str(x) for x in v]


def _as_str_grid(v):
    if not isinstance(v, list):
        return []
    rows = []
    for row in v:
        rows.append([_as_str(c) for c in row] if isinstance(row, list) else [_as_str(row)])
    return rows


Str = Annotated[str, BeforeValidator(_as_str)]
StrList = Annotated[List[str], BeforeValidator(_as_str_list)]
StrGrid = Annotated[List[List[str]], BeforeValidator(_as_str_grid)]


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
    reason: str                 # why selected: "episode_end" | ...
    transcript_window: str = "" # spoken text around this frame (anchoring ground truth)


class KeyframeIndex(BaseModel):
    lecture: str
    frames: List[Keyframe] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# vl_report stage
# ---------------------------------------------------------------------------
class Table(BaseModel):
    caption: Str = ""
    header: StrList = Field(default_factory=list)
    rows: StrGrid = Field(default_factory=list)


class Scheme(BaseModel):
    caption: Str = ""
    nodes: StrList = Field(default_factory=list)
    edges: StrGrid = Field(default_factory=list)  # [["a","b"], ...]
    notes: Str = ""


class Moment(BaseModel):
    """One analyzed visual moment, grouped from one or more keyframes."""

    timestamp_start: float
    timestamp_end: float
    frame_paths: List[str] = Field(default_factory=list)
    kind: Str = "other"  # algorithm | worked_example | definition | table | scheme | other
    description: Str = ""               # English summary of what is shown
    board_text: Str = ""               # text read off the board (English, terms verbatim)
    formulas: StrList = Field(default_factory=list)  # LaTeX
    tables: List[Table] = Field(default_factory=list)
    schemes: List[Scheme] = Field(default_factory=list)
    transcript_excerpt: Str = ""


class VLReport(BaseModel):
    lecture: str
    moments: List[Moment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# synthesize stage — the final artifact your Agent loads
# ---------------------------------------------------------------------------
class WorkedExample(BaseModel):
    given: Str = ""
    calculation: StrList = Field(default_factory=list)  # numeric steps
    result: Str = ""


class OutputFormat(BaseModel):
    appearance: Str = ""  # exact board layout (table/formula as shown)
    template: Str = ""    # fill-in template the Agent should emit


class TaskSource(BaseModel):
    lecture: str
    timestamps: StrList = Field(default_factory=list)  # ["00:12:30-00:15:10"]
    frames: StrList = Field(default_factory=list)


class TaskSpec(BaseModel):
    """A single task class, ready to register as an Agent sub-agent."""

    task_class: str                       # machine slug, e.g. "modular_inverse"
    title: Str                            # English title
    title_source: Str = ""               # original Hebrew/Russian term, verbatim
    description: Str = ""                 # when/why to use it
    trigger_keywords: StrList = Field(default_factory=list)
    algorithm_steps: StrList = Field(default_factory=list)
    formulas: StrList = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    schemes: List[Scheme] = Field(default_factory=list)
    worked_example: WorkedExample = Field(default_factory=WorkedExample)
    output_format: OutputFormat = Field(default_factory=OutputFormat)
    source: TaskSource
