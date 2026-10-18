"""Print markdown score tables from the eval outputs, one row per model.

Usage:
  python scripts/summarize.py Base=/scratch/$USER/p1/eval/base Random=/scratch/$USER/p1/eval/random_e3

Metrics:
  pass@1     average accuracy over the 8 samples per question (what the eval README asks us to report)
  pass@8     a question counts as solved if any of its 8 samples is right (eval.py calls this "Acc")
  no_answer  fraction of samples with no \\boxed{} answer at all (usually ran out of tokens)

slurm/eval_all.sh splits the big benchmarks into chunks, so this script merges the chunk files back
together. If some chunks of a benchmark are still missing, that benchmark is left out of the table.
"""
import json
import re
import sys
from pathlib import Path

DATASETS = ["aime", "amc", "cn_math_2024", "gpqa", "kaoyan", "math", "minerva", "olympiadbench"]
EVAL_DATA = Path(__file__).resolve().parents[1] / "eval" / "data"
SHORT = {"cn_math_2024": "cn_math", "olympiadbench": "olympiad"}  # shorter column names


def n_questions(dataset):
    return sum(1 for line in open(EVAL_DATA / dataset / "test.jsonl") if line.strip())


def scores(root, n_questions=n_questions):
    out = {}
    for d in DATASETS:
        # Key results by question position instead of question text, because Kaoyan has a few
        # questions with identical text.
        by_index = {}
        for f in sorted(Path(root).glob(f"**/{d}/*.jsonl")):
            start = int(re.search(r"_s(\d+)_e\d+\.jsonl$", f.name).group(1))
            for k, line in enumerate(open(f)):
                by_index[start + k] = json.loads(line)
        if not by_index:
            continue
        if len(by_index) < n_questions(d):
            print(f"partial: {root} {d} {len(by_index)}/{n_questions(d)} questions", file=sys.stderr)
            continue
        rows = list(by_index.values())
        out[d] = {
            "pass@1": sum(sum(r["answers_correctness"]) / len(r["answers_correctness"]) for r in rows) / len(rows),
            "pass@8": sum(any(r["answers_correctness"]) for r in rows) / len(rows),
            "no_answer": sum(a == "" for r in rows for a in r["generated_answers"])
                         / sum(len(r["generated_answers"]) for r in rows),
        }
    return out


def table(results, metric):
    header = ["model"] + [SHORT.get(d, d) for d in DATASETS] + ["AVG"]
    rows = []
    for name, s in results.items():
        vals = [s[d][metric] * 100 if d in s else None for d in DATASETS]
        avg = f"{sum(vals) / len(vals):.1f}" if None not in vals else "—"
        rows.append([name] + ["—" if v is None else f"{v:.1f}" for v in vals] + [avg])

    # Pad every column to the same width so the table lines up in a terminal.
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    def line(r):
        cells = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))]
        return "| " + " | ".join(cells) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([f"**{metric}** (%)", "", line(header), sep] + [line(r) for r in rows])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = {name: scores(path) for name, path in (arg.split("=", 1) for arg in sys.argv[1:])}
    print("\n\n".join(table(results, m) for m in ("pass@1", "pass@8", "no_answer")))
