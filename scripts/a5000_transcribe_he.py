#!/usr/bin/env python3
"""Hebrew transcription of one Google Drive audio/video file on the A5000 server, result pushed to Hugging Face.

Steps: download the file from Google Drive (gdown; the file must be shared "Anyone with the link") -> transcribe with faster-whisper
and the Hebrew fine-tune of Whisper large-v3 by ivrit.ai (ivrit-ai/whisper-large-v3-ct2, CTranslate2, float16 on the GPU) ->
write transcript_he.{txt,srt,json} + meta.json -> upload them to the PUBLIC HF dataset repo (folder <name>/) -> delete the
downloaded media file and the local outputs (also when a step fails).

One-time setup on the server (venv ~/env, already logged in with `hf auth login`):
    pip install faster-whisper gdown

Run:
    python ~/lection-analyzer/scripts/a5000_transcribe_he.py          # the Drive link (URL) and HF folder (NAME) are set below
-> https://huggingface.co/datasets/cyttic/lection-transcripts/tree/main/lecture1
"""
import argparse, ctypes, glob, json, os, re, shutil, site, sys, tempfile, time

MODEL = "ivrit-ai/whisper-large-v3-ct2"
REPO = "cyttic/lection-transcripts"
URL = "https://drive.google.com/file/d/1a5FI63_PAHYamef3GAfFbnBeipELWw2k/view?usp=drive_link"
NAME = "lecture1"


def preload_cuda_libs():
    """CTranslate2 needs cuBLAS 12 + cuDNN 9; the pip wheels that come with torch have them, but not on the loader path. Load them globally."""
    for sp in site.getsitepackages():
        for pat in ("nvidia/cublas/lib/libcublas*.so.*", "nvidia/cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(os.path.join(sp, pat))):
                try: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError: pass


def srt_time(t):
    h, rem = divmod(int(t), 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{int((t - int(t)) * 1000):03d}"


def clock(t):
    h, rem = divmod(int(t), 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=URL, help="Google Drive share link of the audio/video file (default: URL above)")
    ap.add_argument("--name", default=NAME, help="folder in the HF repo")
    ap.add_argument("--repo", default=REPO, help="HF dataset repo (created PUBLIC if missing)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--no-push", action="store_true", help="test: keep the outputs locally, do not upload")
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix="transcribe_", dir=os.path.expanduser("~"))
    out = os.path.join(work, "out"); os.makedirs(out)
    media = None
    try:
        import gdown
        fid = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", a.url); assert fid, f"no file id in {a.url}"
        # plain uc?id= link: works with every gdown version (newer ones dropped the fuzzy= argument)
        media = gdown.download(f"https://drive.google.com/uc?id={fid.group(1) or fid.group(2)}", os.path.join(work, ""), quiet=False)
        assert media and os.path.exists(media), "download failed - is the file shared 'Anyone with the link'?"
        name = a.name or os.path.splitext(os.path.basename(media))[0]
        print(f"downloaded {media} ({os.path.getsize(media) / 1e6:.1f} MB) -> HF folder '{name}'", flush=True)

        preload_cuda_libs()
        import av   # PyAV >= 15 removed av.open(metadata_errors=...), which faster-whisper still passes: drop it
        _av_open = av.open
        av.open = lambda *x, **k: _av_open(*x, **{kk: v for kk, v in k.items() if kk != "metadata_errors"})
        from faster_whisper import WhisperModel
        t0 = time.time()
        asr = WhisperModel(a.model, device="cuda", compute_type="float16")
        seg_iter, info = asr.transcribe(media, language="he", beam_size=a.beam, vad_filter=True)
        print(f"audio {clock(info.duration)}, model {a.model}; decoding ...", flush=True)
        segs = []
        for s in seg_iter:
            segs.append({"id": len(segs), "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
            if len(segs) % 50 == 0: print(f"  {clock(s.end)} / {clock(info.duration)}", flush=True)
        dt = time.time() - t0
        print(f"{len(segs)} segments in {dt / 60:.1f} min", flush=True)

        json.dump(segs, open(os.path.join(out, "transcript_he.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        srt = []
        for n, s in enumerate(segs, 1): srt += [str(n), f"{srt_time(s['start'])} --> {srt_time(s['end'])}", s["text"], ""]
        open(os.path.join(out, "transcript_he.srt"), "w", encoding="utf-8").write("\n".join(srt))
        open(os.path.join(out, "transcript_he.txt"), "w", encoding="utf-8").write(
            "\n".join(f"[{clock(s['start'])}-{clock(s['end'])}] {s['text']}" for s in segs) + "\n")
        json.dump({"source_file": os.path.basename(media), "model": a.model, "language": "he", "beam_size": a.beam, "vad_filter": True,
                   "audio_seconds": round(info.duration, 1), "segments": len(segs), "decode_minutes": round(dt / 60, 2)},
                  open(os.path.join(out, "meta.json"), "w"), indent=1)

        if a.no_push:
            keep = os.path.abspath(f"transcript_{name}"); shutil.copytree(out, keep, dirs_exist_ok=True); print("kept locally:", keep)
        else:
            from huggingface_hub import HfApi
            api = HfApi(); api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
            for i in range(5):
                try:
                    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=out, path_in_repo=name,
                                      commit_message=f"Hebrew transcript of {os.path.basename(media)} ({a.model})"); break
                except Exception as e:
                    print(f"upload failed ({e}); retry {i + 1}/5", flush=True); time.sleep(30)
            else:
                sys.exit("upload failed 5 times")
            print(f"pushed: https://huggingface.co/datasets/{a.repo}/tree/main/{name}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        print("removed from the server:", media or "(nothing downloaded)", "and", work)


if __name__ == "__main__":
    main()
