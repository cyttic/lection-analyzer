# Hebrew lecture → subtitles (A5000 + Whisper)

## 1. Extract the audio (local)
```bash
ffmpeg -i lecture.mp4 -ac 1 -ar 16000 lecture_audio.flac
```
16 kHz mono FLAC: the format Whisper uses, much smaller than the video.

## 2. Upload to Google Drive
Upload `lecture_audio.flac` → Share → **Anyone with the link**.

## 3. Set the link and name in the script (local)
In `scripts/a5000_transcribe_he.py`:
```python
URL = "https://drive.google.com/file/d/<FILE_ID>/view"
NAME = "lecture_name"          # folder in the HF repo
```
Push:
```bash
git add scripts/a5000_transcribe_he.py && git commit -m "transcribe lecture_name" && git push
```

## 4. Transcribe (A5000 server)
One-time setup:
```bash
pip install faster-whisper gdown nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```
Each lecture:
```bash
cd ~/lection-analyzer && git pull
tmux new -s whisper
python ~/lection-analyzer/scripts/a5000_transcribe_he.py
```
Model: `ivrit-ai/whisper-large-v3-ct2` (GPU, float16). Writes `transcript_he.{txt,srt,json}` + `meta.json`, uploads them to
`https://huggingface.co/datasets/cyttic/lection-transcripts/tree/main/lecture_name` (**public**), deletes the audio from the server.

## 5. Download the subtitles (local)
```bash
hf download cyttic/lection-transcripts lecture_name/transcript_he.srt --repo-type dataset --local-dir .
```

## 6. Burn the subtitles into the video (local)
```bash
ffmpeg -i lecture.mp4 -vf subtitles=lecture_name/transcript_he.srt lecture_subtitled.mp4
```
