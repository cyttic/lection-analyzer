"""Prompt templates for the VL and synthesis stages.

Kept in one place so they're easy to tune — prompt quality is the main lever on
output quality, especially for the Hebrew-handwriting + table/formula case.
"""

# ---------------------------------------------------------------------------
# VL stage: one board frame (+ transcript window) -> structured visual JSON
# ---------------------------------------------------------------------------
VL_SYSTEM = (
    "You analyze a single frame from a university lecture. The board may be in Hebrew "
    "(mostly) or Russian and is often HANDWRITTEN. You are given the lecturer's spoken "
    "words around this moment as GROUND TRUTH for wording and terminology — use them to "
    "disambiguate hard-to-read handwriting. Your unique job is to capture the 2D VISUAL "
    "STRUCTURE that the audio cannot convey: tables, formulas, and schemes/diagrams."
)

VL_INSTRUCTIONS = """\
Spoken context around this frame (ground truth for words/terms):
\"\"\"{transcript}\"\"\"

Look at the board image and return ONE JSON object, nothing else:

{{
  "kind": "algorithm | worked_example | definition | table | scheme | other",
  "description": "1-3 sentences in ENGLISH: what is shown and what it is for",
  "board_text": "the text written on the board, in ENGLISH; keep math/Hebrew/Russian terms verbatim",
  "formulas": ["each formula as LaTeX"],
  "tables": [{{"caption": "", "header": ["col1","col2"], "rows": [["a","b"]]}}],
  "schemes": [{{"caption": "", "nodes": ["n1","n2"], "edges": [["n1","n2"]], "notes": "arrows/flow"}}]
}}

Rules:
- If the board has a table, RECONSTRUCT the grid faithfully (rows/columns), don't flatten it.
- Render every formula as LaTeX. Math is language-neutral; get it visually exact.
- For diagrams/flowcharts, list nodes and directed edges and describe the flow in "notes".
- When the handwriting and the spoken context disagree on a WORD, trust the spoken context.
- Use [] for any field with nothing to report. Output JSON only.
"""


def vl_prompt(transcript_window: str) -> str:
    return VL_INSTRUCTIONS.format(transcript=transcript_window or "(no speech captured)")


# ---------------------------------------------------------------------------
# Synthesis stage: all moments + transcript -> list of TaskSpec
# ---------------------------------------------------------------------------
SYN_SYSTEM = (
    "You convert analyzed lecture moments into reusable TASK-CLASS specifications that will "
    "train an autonomous Agent to solve homework/exam problems of that kind. A 'task class' is "
    "a recurring procedure the lecturer teaches (e.g. an algorithm or a calculation method). "
    "Write everything in ENGLISH, keeping Hebrew/Russian source terms and all formulas verbatim. "
    "FUSION RULE: for wording trust the transcript; for tables/formulas/diagram layout trust the "
    "visual report."
)

SYN_INSTRUCTIONS = """\
You are given ONE lecture episode: the completed board (analyzed below as structured JSON) and
the lecturer's spoken words DURING that episode. This episode usually teaches a single task
(occasionally a few; sometimes none).

SPOKEN WORDS during this episode (original language, may be Hebrew/Russian):
\"\"\"{transcript}\"\"\"

THIS EPISODE'S BOARD (analyzed JSON):
{moments}

Extract the reproducible TASK CLASS(es) this episode teaches — a procedure/algorithm/calculation
a student must be able to redo. If the episode is pure narration with no reproducible method,
return an empty array [].

Return ONE JSON array (0, 1, or a few elements). Each element:

{{
  "task_class": "snake_case_slug",
  "title": "English title",
  "title_source": "original Hebrew/Russian term, verbatim (or empty)",
  "description": "when and why to use this method",
  "trigger_keywords": ["signals in a problem statement that this task class applies"],
  "algorithm_steps": ["ordered, imperative steps to solve it"],
  "formulas": ["LaTeX formulas used"],
  "tables": [{{"caption": "", "header": [], "rows": []}}],
  "schemes": [{{"caption": "", "nodes": [], "edges": [], "notes": ""}}],
  "worked_example": {{
    "given": "the example inputs exactly as on the board",
    "calculation": ["each numeric step, matching the board"],
    "result": "final answer as shown"
  }},
  "output_format": {{
    "appearance": "how the answer should LOOK (table/formula layout as on the board)",
    "template": "a fill-in template the Agent should emit for new problems"
  }},
  "source": {{"timestamps": ["MM:SS-MM:SS"], "frames": ["frames/...jpg"]}}
}}

Rules:
- Use the worked example from THIS board; preserve its numbers exactly.
- output_format.appearance must reflect the real board layout (use the visual tables/formulas).
- "source" is filled in automatically — you may output {{}} for it or omit it.
- Output the JSON array only, no prose.
"""


def synthesis_prompt(transcript_text: str, moments_json: str) -> str:
    return SYN_INSTRUCTIONS.format(transcript=transcript_text, moments=moments_json)
