"""Offline test for the JSON-robustness fixes (lenient parse + type coercion)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lection_analyzer.backends.base import loads_lenient  # noqa: E402
from lection_analyzer.schemas import Moment, TaskSource, TaskSpec  # noqa: E402

# 1. LaTeX backslashes + trailing comma + ```json fence (the "Invalid \escape" failures)
fenced = 'pre ```json\n{"formulas": ["\\gcd(a,b)"], "board_text": "x = \\frac{1}{2}",}\n``` post'
r = loads_lenient(fenced)
assert r["formulas"] == ["\\gcd(a,b)"], r["formulas"]
assert r["board_text"] == "x = \\frac{1}{2}", r["board_text"]
print("lenient LaTeX/trailing-comma parse: ok ->", r)

# 2. truncated/unterminated object (the "Unterminated string" failures) — repaired, not dropped
truncated = '{"a": "hello", "b": "world'
r2 = loads_lenient(truncated)
assert r2.get("a") == "hello", r2
print("truncation repair: ok ->", r2)

# 3. dict where a string is expected -> coerced (the pydantic string_type errors you hit)
m = Moment(timestamp_start=0, timestamp_end=1, board_text={"keywording": ["a", "b"]})
assert isinstance(m.board_text, str), m.board_text
t = TaskSpec(
    task_class="tfidf", title="TF-IDF",
    worked_example={"given": {"doc1": {"English": 27}}, "result": {"TFIDF": "0.2"}},
    source=TaskSource(lecture="lec01"),
)
assert isinstance(t.worked_example.given, str) and isinstance(t.worked_example.result, str)
print("dict->str coercion: ok ->", t.worked_example.given, "/", t.worked_example.result)

# 4. numbers inside a table grid -> coerced to str
m2 = Moment(timestamp_start=0, timestamp_end=1, tables=[{"header": ["a", "b"], "rows": [[1, 2], [3, 4]]}])
assert m2.tables[0].rows == [["1", "2"], ["3", "4"]], m2.tables[0].rows
print("table-grid coercion: ok ->", m2.tables[0].rows)

print("\nALL ROBUSTNESS CHECKS PASSED")
