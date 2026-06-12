"""Stage 1 — ingest: get the lecture video onto local disk.

Source can be a local/mounted path (e.g. a Kaggle dataset under /kaggle/input) or a
Google Drive file id/URL (requires internet on). Output: ``cfg.raw_video``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config


def run(cfg: Config) -> Path:
    cfg.ensure_dirs()
    out = cfg.raw_video
    if out.exists() and out.stat().st_size > 0:
        print(f"[ingest] already present: {out}")
        return out

    ing = cfg.ingest
    source_path = ing.get("source_path")
    gdrive_id = ing.get("gdrive_id")
    gdrive_url = ing.get("gdrive_url")

    if source_path:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"[ingest] source_path not found: {src}")
        print(f"[ingest] copying local file {src} -> {out}")
        shutil.copy(src, out)
    elif gdrive_id or gdrive_url:
        import gdown  # local import: only needed for the Drive path

        if gdrive_id:
            print(f"[ingest] downloading Google Drive id {gdrive_id}")
            gdown.download(id=gdrive_id, output=str(out), quiet=False)
        else:
            print(f"[ingest] downloading {gdrive_url}")
            gdown.download(url=gdrive_url, output=str(out), quiet=False, fuzzy=True)
    else:
        raise ValueError(
            "[ingest] set one of ingest.source_path / ingest.gdrive_id / ingest.gdrive_url"
        )

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"[ingest] download produced no file at {out}")
    print(f"[ingest] ready: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
