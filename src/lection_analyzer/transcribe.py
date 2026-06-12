"""Stage 2 — transcribe: audio -> timestamped transcript.

ffmpeg extracts 16 kHz mono PCM, then faster-whisper (CTranslate2) produces segments
with timestamps. The transcript is kept in the ORIGINAL language (Hebrew/Russian) so
verbatim terms survive into later stages; translation to English happens downstream.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from .config import Config
from .schemas import Segment, Transcript


def _extract_audio(video: Path, wav: Path) -> None:
    if wav.exists() and wav.stat().st_size > 0:
        print(f"[transcribe] audio already extracted: {wav}")
        return
    print(f"[transcribe] extracting audio -> {wav}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _srt_time(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(transcript: Transcript, path: Path) -> None:
    lines = []
    for i, seg in enumerate(transcript.segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _clock(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _write_txt(transcript: Transcript, path: Path) -> None:
    """Readable transcript: one line per segment, prefixed with [HH:MM:SS-HH:MM:SS]."""
    lines = [
        f"[{_clock(seg.start)}-{_clock(seg.end)}] {seg.text.strip()}"
        for seg in transcript.segments
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> Transcript:
    cfg.ensure_dirs()
    if cfg.transcript_json.exists():
        print(f"[transcribe] cached: {cfg.transcript_json}")
        return Transcript.model_validate_json(cfg.transcript_json.read_text("utf-8"))

    _extract_audio(cfg.raw_video, cfg.audio_wav)

    from faster_whisper import WhisperModel  # heavy import, only here

    tcfg = cfg.transcribe
    device = tcfg.get("device", "auto")
    if device == "auto":
        try:
            import torch  # noqa

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute = tcfg.get("compute_type", "default")
    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"

    print(f"[transcribe] loading whisper '{tcfg.get('model')}' on {device}/{compute}")
    model = WhisperModel(tcfg.get("model", "large-v3"), device=device, compute_type=compute)

    language = tcfg.get("language")  # None => autodetect
    segments_iter, info = model.transcribe(
        str(cfg.audio_wav),
        language=language,
        vad_filter=tcfg.get("vad_filter", True),
        word_timestamps=False,
    )

    segments = []
    lang_counter: Counter = Counter()
    default_lang = getattr(info, "language", None)
    for s in segments_iter:
        seg_lang = getattr(s, "language", None) or default_lang
        segments.append(Segment(start=s.start, end=s.end, text=s.text, lang=seg_lang))
        if seg_lang:
            lang_counter[seg_lang] += 1
        print(f"[transcribe] {s.start:7.1f}s [{seg_lang}] {s.text.strip()[:70]}")

    total = sum(lang_counter.values()) or 1
    transcript = Transcript(
        lecture=cfg.lecture,
        language_summary={k: round(v / total, 3) for k, v in lang_counter.items()},
        segments=segments,
    )

    cfg.transcript_json.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    _write_srt(transcript, cfg.transcript_srt)
    _write_txt(transcript, cfg.transcript_txt)
    print(
        f"[transcribe] {len(segments)} segments, langs={transcript.language_summary}\n"
        f"[transcribe]   json: {cfg.transcript_json}\n"
        f"[transcribe]   srt:  {cfg.transcript_srt}\n"
        f"[transcribe]   txt:  {cfg.transcript_txt}  (text + timecodes)"
    )
    return transcript
