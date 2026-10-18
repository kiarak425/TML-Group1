import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize import scores, table  # noqa: E402


def write(root, dataset, rows):
    # eval.py saves results under the last 3 parts of the model path
    d = root / "p1/models/random" / dataset
    d.mkdir(parents=True)
    (d / "test_qwen-instruct_t0.6_k4_s0_e2.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def write_chunk(root, dataset, start, rows):
    d = root / "p1/models/random" / dataset
    d.mkdir(parents=True, exist_ok=True)
    name = f"test_qwen-instruct_t0.6_k8_s{start}_e{start + len(rows)}.jsonl"
    (d / name).write_text("\n".join(json.dumps(r) for r in rows))


def test_scores_computes_pass1_pass8_and_no_answer_rate(tmp_path):
    write(tmp_path, "amc", [
        {"answers_correctness": [True, False, False, False], "generated_answers": ["1", "", "2", "3"]},
        {"answers_correctness": [False] * 4, "generated_answers": [""] * 4},
    ])
    expected = {"amc": {"pass@1": 0.125, "pass@8": 0.5, "no_answer": 0.625}}
    assert scores(tmp_path, n_questions={"amc": 2}.get) == expected


def test_table_marks_missing_benchmarks_and_only_averages_full_rows():
    full = {d: {"pass@1": 0.5} for d in ["aime", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"]}
    t = table({"base": full, "random": {"amc": {"pass@1": 0.25}}}, "pass@1")
    base_row, random_row = (line for line in t.splitlines() if line.startswith(("| base", "| random")))
    cells = lambda line: [c.strip() for c in line.strip("|").split("|")]
    assert cells(base_row)[-1] == "50.0"                                # average of a complete row
    assert "25.0" in cells(random_row) and cells(random_row)[-1] == "—"  # incomplete row, so no average
    assert len({len(line) for line in t.splitlines()[2:]}) == 1         # columns are aligned


def test_scores_merges_chunks_without_double_counting(tmp_path):
    q = lambda ok: {"question": "same text", "answers_correctness": [ok, ok], "generated_answers": ["1", "1"]}
    write_chunk(tmp_path, "aime", 0, [q(True)] * 15)                      # questions 0-14
    write_chunk(tmp_path, "aime", 15, [q(False)] * 15)                    # questions 15-29
    write_chunk(tmp_path, "aime", 0, [q(True)] * 15 + [q(False)] * 15)   # an older full run covering both
    # 30 questions in total, not 60. Repeated question text doesn't matter since we match by position.
    assert scores(tmp_path)["aime"]["pass@1"] == 0.5


def test_scores_skips_benchmarks_with_missing_chunks(tmp_path):
    rows = [{"question": f"q{i}", "answers_correctness": [True], "generated_answers": ["1"]} for i in range(15)]
    write_chunk(tmp_path, "aime", 0, rows)  # aime has 30 questions, so this is only half
    assert "aime" not in scores(tmp_path)
