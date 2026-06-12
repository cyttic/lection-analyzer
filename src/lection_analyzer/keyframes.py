"""Stage 3 — keyframes: pick the frames worth analyzing.

Combines three signals:
  * scene cuts (PySceneDetect content detector) — board wiped / slide changed,
  * transcript cue words (algorithm / formula / example / table ...) — boost sampling
    where the lecturer flags something important,
then keeps the *fullest* frame of each stable board segment (the frame just before
the next change), dedups near-identical frames with a perceptual hash, and enforces
a minimum spacing. Output: jpgs under ``cfg.frames_dir`` + ``frames_index.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2

from .config import Config
from .schemas import Keyframe, KeyframeIndex, Transcript


def _scene_candidates(video: Path, threshold: float) -> List[Tuple[float, str]]:
    """Timestamps just before each detected scene cut, plus the very first frame."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=threshold))
    cands: List[Tuple[float, str]] = [(0.5, "scene_start")]
    for _start, end in scenes:
        # 'end' is the cut point; the frame just before it is the fullest board.
        cands.append((max(0.0, end.get_seconds() - 0.4), "scene_cut"))
    return cands


def _cue_candidates(transcript: Transcript, keywords: List[str]) -> List[Tuple[float, str]]:
    lowered = [k.lower() for k in keywords]
    cands: List[Tuple[float, str]] = []
    for seg in transcript.segments:
        text = seg.text.lower()
        for kw in lowered:
            if kw in text:
                cands.append((seg.end, f"cue:{kw}"))
                break
    return cands


def _grab_frame(cap: "cv2.VideoCapture", t: float, out_path: Path) -> bool:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out_path), frame)
    return True


def run(cfg: Config, transcript: Transcript) -> KeyframeIndex:
    cfg.ensure_dirs()
    if cfg.keyframe_index_json.exists():
        print(f"[keyframes] cached: {cfg.keyframe_index_json}")
        return KeyframeIndex.model_validate_json(cfg.keyframe_index_json.read_text("utf-8"))

    kf = cfg.keyframes
    threshold = float(kf.get("scene_threshold", 27.0))
    min_gap = float(kf.get("min_seconds_between", 4))
    phash_dist = int(kf.get("phash_distance", 6))
    keywords = kf.get("cue_keywords", [])

    print("[keyframes] detecting scenes...")
    candidates = _scene_candidates(cfg.raw_video, threshold)
    candidates += _cue_candidates(transcript, keywords)
    candidates.sort(key=lambda c: c[0])

    # Enforce minimum spacing on candidate timestamps (greedy, earliest wins).
    spaced: List[Tuple[float, str]] = []
    last_t = -1e9
    for t, reason in candidates:
        if t - last_t >= min_gap:
            spaced.append((t, reason))
            last_t = t

    import imagehash
    from PIL import Image

    cap = cv2.VideoCapture(str(cfg.raw_video))
    kept: List[Keyframe] = []
    kept_hashes: List["imagehash.ImageHash"] = []
    for t, reason in spaced:
        tmp = cfg.frames_dir / f"{int(t * 1000):08d}.jpg"
        if not _grab_frame(cap, t, tmp):
            continue
        h = imagehash.phash(Image.open(tmp))
        if any((h - kh) <= phash_dist for kh in kept_hashes):
            tmp.unlink(missing_ok=True)  # near-duplicate board, skip
            continue
        kept_hashes.append(h)
        kept.append(
            Keyframe(
                timestamp=t,
                path=str(tmp.relative_to(cfg.data_dir)),
                reason=reason,
                transcript_window=transcript.window(t, t),
            )
        )
    cap.release()

    index = KeyframeIndex(lecture=cfg.lecture, frames=kept)
    cfg.keyframe_index_json.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    print(f"[keyframes] kept {len(kept)} frames -> {cfg.keyframe_index_json}")
    return index
