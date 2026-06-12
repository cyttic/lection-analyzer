# Lecture Analyzer

Turn lecture videos (mostly **Hebrew**, some **Russian**, handwritten board) into
**structured task-class specs** you can load into your Agent. For each recurring
procedure the lecturer teaches, you get one JSON file with the algorithm steps, a
worked numeric example, and the exact output appearance shown on the board.

## Pipeline

```
video ─▶ ingest ─▶ transcribe ─▶ keyframes ─▶ vl_report ─▶ synthesize ─▶ output/*.json
        (drive/    (Whisper      (board        (VL: tables/  (one small call
         local)     medium)       episodes:     formulas/     PER episode →
                                  1 final frame  schemes)      task specs)
                                  each)
```

**Per-task, not whole-lecture.** A lecture is a sequence of mostly independent tasks.
`keyframes` splits it into board *episodes* (one per board-wipe) and keeps only each
episode's **final frame** — the completed result, not the in-progress writing. `synthesize`
then makes **one small LLM call per episode** (that frame's data + its local transcript),
so prompt size is bounded by a single task, never the whole lecture. This keeps it within a
single T4's memory and cheap enough to run **locally** — no giant context, no per-lecture API bill.

Each stage **caches its artifact** under `data/<lecture>/`, so a crashed or long run
resumes cheaply, and you can re-run just one stage by deleting its artifact.

### What each stage trusts
- The **transcript is ground truth for words/terms** (it back-stops weak Hebrew
  handwriting OCR).
- The **VL model owns 2D structure**: tables (grid), formulas (LaTeX), schemes
  (nodes/edges). When they disagree on a *word* → trust transcript; on *layout* → trust VL.

## Models — free by default, Claude optional

Set independently in `config.yaml` under `backends`:

| Stage      | `local` (free, offline)        | `claude` (best quality, needs key + internet) |
|------------|--------------------------------|-----------------------------------------------|
| `vl_backend`        | Qwen2.5-VL-7B (4-bit) | `claude-opus-4-8` vision |
| `synthesis_backend` | Qwen2.5-VL-7B (text)  | `claude-opus-4-8`        |

Start with `local`. If tables/formulas come out garbled, flip **just** `vl_backend`
to `claude` and re-run `--only vl_report` (set `ANTHROPIC_API_KEY` in the env).

## Run on Kaggle (intended path)

1. Upload your video to Google Drive (or as a Kaggle dataset).
2. Open `notebooks/kaggle_run.ipynb`. Set Accelerator = **GPU T4**, Internet = **On**.
3. Point `config.yaml` `ingest.gdrive_id` or `ingest.source_path` at your video; set `lecture`.
4. Run the cells. Download `output/<lecture>/*.json` — those are your Agent specs.

For the Claude backend, add `ANTHROPIC_API_KEY` as a Kaggle **Secret** (cell 2 loads it).

## Run from the CLI

```bash
pip install -r requirements.txt          # + ffmpeg on PATH
export PYTHONPATH=src
python -m lection_analyzer.pipeline --config config.yaml
python -m lection_analyzer.pipeline --only vl_report     # one stage
python -m lection_analyzer.pipeline --from keyframes     # this stage onward
```

## Output: `TaskSpec`

```jsonc
{
  "task_class": "euclid_gcd",
  "title": "Greatest Common Divisor via Euclid",
  "title_source": "אלגוריתם אוקלידס",
  "description": "When/why to use it.",
  "algorithm_steps": ["..."],
  "formulas": ["\\gcd(a,b)=\\gcd(b, a \\bmod b)"],
  "tables": [{ "caption": "", "header": ["a","b"], "rows": [["48","18"]] }],
  "schemes": [{ "nodes": [], "edges": [], "notes": "" }],
  "worked_example": { "given": "a=48,b=18", "calculation": ["48 mod 18 = 12", "..."], "result": "gcd = 6" },
  "output_format": { "appearance": "two-column trace table", "template": "gcd({a},{b}) = {result}" },
  "source": { "lecture": "lec01", "timestamps": ["00:12:30-00:15:10"], "frames": ["frames/...jpg"] }
}
```

Each `*.json` is written alongside a human-readable `*.md`.

## Layout

```
src/lection_analyzer/
  config.py schemas.py prompts.py
  ingest.py transcribe.py keyframes.py vl_report.py synthesize.py pipeline.py
  backends/{base,local_qwen,claude_api}.py
notebooks/kaggle_run.ipynb
scripts/smoke_test.py        # offline, no GPU — validates stage logic with fake backends
config.yaml requirements.txt
```

## Test

```bash
PYTHONPATH=src python scripts/smoke_test.py   # no models/video needed
```
