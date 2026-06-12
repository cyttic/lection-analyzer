"""Durable transcript cache — transcribe a lecture once, ever (not once per session).

``/kaggle/working`` is wiped when a Kaggle session ends, so the on-disk transcript cache
there does not survive into the next day's session. This module syncs transcripts to a
persistent **Kaggle Dataset**:

* RESTORE — if that dataset is attached to the notebook (mounted read-only under
  ``/kaggle/input/<name>``), copy this lecture's transcript into the working data dir
  *before* transcription runs, so the normal cache check reuses it.
* SAVE — after a fresh transcription, push the transcript files up as a new dataset
  version via the Kaggle CLI (needs your Kaggle API token in the environment).

Both halves are no-ops unless configured under ``persist:`` in config.yaml, so the
pipeline works fine without any of this.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import Config

TRANSCRIPT_FILES = ("transcript.json", "transcript.srt", "transcript.txt")


# Files are stored flat in the dataset as "<lecture>__transcript.json" etc. — no subdirs,
# so the dataset stays directly readable when attached (no zip/extract step).
def _flat(lecture: str, name: str) -> str:
    return f"{lecture}__{name}"


def restore_transcript(cfg: Config) -> bool:
    """Copy this lecture's transcript from the attached dataset into the working dir.

    Returns True if a usable transcript is now present locally.
    """
    if cfg.transcript_json.exists():
        return True
    src_root = cfg.persist.get("transcript_dataset_dir")
    if not src_root:
        return False
    src = Path(src_root)
    if not (src / _flat(cfg.lecture, "transcript.json")).exists():
        return False
    cfg.ensure_dirs()
    for name in TRANSCRIPT_FILES:
        f = src / _flat(cfg.lecture, name)
        if f.exists():
            shutil.copy(f, cfg.data_dir / name)
    print(f"[persist] restored transcript for '{cfg.lecture}' from {src}")
    return True


def save_transcript(cfg: Config) -> None:
    """Push this lecture's transcript to the persistent Kaggle Dataset (new version)."""
    slug = cfg.persist.get("transcript_dataset_slug")
    if not slug:
        return

    staging = Path(cfg.persist.get("staging_dir", "/kaggle/working/_transcript_store"))
    staging.mkdir(parents=True, exist_ok=True)

    # A new dataset version replaces ALL files, so carry forward the previous contents
    # (other lectures, stored flat) from the currently-attached version first.
    prev = cfg.persist.get("transcript_dataset_dir")
    if prev and Path(prev).exists():
        for item in Path(prev).iterdir():
            if item.is_file() and item.name != "dataset-metadata.json":
                shutil.copy(item, staging / item.name)

    for name in TRANSCRIPT_FILES:
        f = cfg.data_dir / name
        if f.exists():
            shutil.copy(f, staging / _flat(cfg.lecture, name))

    _, _, name = slug.partition("/")
    meta = {"title": name or slug, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    msg = f"add/update {cfg.lecture}"
    # Try to add a version; if the dataset doesn't exist yet, create it.
    if not _kaggle(["datasets", "version", "-p", str(staging), "-m", msg]):
        _kaggle(["datasets", "create", "-p", str(staging)])


def _kaggle(args: list[str]) -> bool:
    try:
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    except FileNotFoundError:
        print("[persist] kaggle CLI not found; skipping transcript upload")
        return False
    out = (r.stdout or r.stderr).strip()
    print(f"[persist] kaggle {args[0]} {args[1]}: {out.splitlines()[-1] if out else r.returncode}")
    return r.returncode == 0
