"""Stage 3 — keyframes: segment the lecture into board EPISODES, keep one frame each.

A lecture is a sequence of mostly independent tasks. Each task lives on the board as
an "episode": the lecturer builds it up, then wipes/changes the board to start the next.
Scene cuts (PySceneDetect content detector) mark those wipes, so each scene ≈ one episode.

We only want the *result* — the completed board — so for each episode we keep a single
frame just before its end cut (the fullest state), NOT the in-progress frames. Each kept
frame also carries that episode's full local transcript, which is all the synthesis stage
needs for that task (no global transcript, no giant attention pass).

Output: one jpg per episode under ``cfg.frames_dir`` + ``frames_index.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2

from .config import Config
from .schemas import Keyframe, KeyframeIndex, Transcript


def _episodes(video: Path, threshold: float, min_seconds: float) -> List[Tuple[float, float]]:
    """Return (start, end) spans between board changes, dropping too-short ones."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=threshold))
    spans: List[Tuple[float, float]] = []
    for start, end in scenes:
        s, e = start.get_seconds(), end.get_seconds()
        if e - s >= min_seconds:
            spans.append((s, e))
    if not spans:  # no cuts detected (single static board) -> treat whole video as one episode
        cap = cv2.VideoCapture(str(video))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        cap.release()
        spans = [(0.0, dur)]
    return spans


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
    min_seconds = float(kf.get("min_episode_seconds", 20))
    phash_dist = int(kf.get("phash_distance", 6))

    print("[keyframes] segmenting into board episodes...")
    spans = _episodes(cfg.raw_video, threshold, min_seconds)

    import imagehash
    from PIL import Image

    cap = cv2.VideoCapture(str(cfg.raw_video))
    kept: List[Keyframe] = []
    kept_hashes: List["imagehash.ImageHash"] = []
    for start, end in spans:
        t_final = max(0.0, end - 0.4)  # just before the wipe = fullest board
        tmp = cfg.frames_dir / f"{int(t_final * 1000):08d}.jpg"
        if not _grab_frame(cap, t_final, tmp):
            continue
        h = imagehash.phash(Image.open(tmp))
        if any((h - kh) <= phash_dist for kh in kept_hashes):
            tmp.unlink(missing_ok=True)  # board barely changed across episodes; merge
            continue
        kept_hashes.append(h)
        kept.append(
            Keyframe(
                timestamp=t_final,
                path=str(tmp.relative_to(cfg.data_dir)),
                reason="episode_end",
                # the WHOLE spoken content of this episode = this task's local context
                transcript_window=transcript.window(start, end, pad=2.0),
            )
        )
    cap.release()

    index = KeyframeIndex(lecture=cfg.lecture, frames=kept)
    cfg.keyframe_index_json.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    print(f"[keyframes] {len(spans)} episodes -> kept {len(kept)} frames -> {cfg.keyframe_index_json}")
    return index
