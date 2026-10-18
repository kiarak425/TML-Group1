"""Select 15K training samples from redsgnaoh/acereason11_100k.

Usage:
  python scripts/select_data.py --strategy random   --push <hf_user>/ece6514-p1-random-15k
  python scripts/select_data.py --strategy advanced --push <hf_user>/ece6514-p1-advanced-15k
  python scripts/select_data.py --strategy concise  --push <hf_user>/ece6514-p1-concise-15k

Each run writes data/<strategy>_15k.json (in the format data/dataset_info.json expects) plus a
<strategy>_15k.stats.json file with the filter counts and token statistics.

Strategies:
  random    15K rows sampled uniformly from the full pool (our baseline).
  advanced  Math only, exactly one non-empty \\boxed{} after </think>, no proof problems, short enough to
            train on without truncation, and no duplicate problems. We then drop the shortest 20% (mostly
            easy problems) and sample uniformly from what's left.
  concise   Same filters as advanced, but we keep only samples between 1,000 and 6,000 tokens. Models trained
            on longer solutions often ran out of tokens during eval without giving an answer.
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

SOURCE = "redsgnaoh/acereason11_100k"
CUTOFF = 16384         # same as cutoff_len in configs/train.yaml
TEMPLATE_TOKENS = 64   # rough upper bound on the extra tokens the Qwen chat template adds
PROOF = re.compile(r"\b(prove|show that|demonstrate)\b", re.I)
TEXT = re.compile(r"\\text\{([^{}]*)\}")
PRESETS = {
    "advanced": dict(drop_short=0.2),
    "concise": dict(drop_short=0.0, min_tokens=1000, max_tokens=6000),
}


def final_answer(output):
    """Return what's inside the single \\boxed{} after </think>, or None.

    The eval grader only looks at the last \\boxed{}, so we want exactly one clear final answer.
    """
    _, sep, tail = output.rpartition("</think>")
    if not sep or tail.count("\\boxed{") != 1:
        return None
    i = j = tail.index("\\boxed{") + len("\\boxed{")
    depth = 1
    while j < len(tail) and depth:
        depth += {"{": 1, "}": -1}.get(tail[j], 0)
        j += 1
    return (tail[i:j - 1].strip() or None) if depth == 0 else None


def is_clean_math(r):
    if r["extra_info"]["category"] != "math" or PROOF.search(r["instruction"]):
        return False
    a = final_answer(r["output"])
    # A boxed sentence like "\text{The pentagon is regular.}" is really a proof, but units like
    # "5 \text{ cm}" are fine, so we only reject \text{} with three or more words.
    return a is not None and not any(len(t.split()) >= 3 for t in TEXT.findall(a))


def prompt_text(r):
    return r["system"] + r["instruction"] + r["input"] + r["output"]


def select_random(rows, n, seed):
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), n))]


def select_advanced(rows, n, seed, count_tokens, cutoff=CUTOFF, drop_short=0.2, funnel=None,
                    min_tokens=0, max_tokens=None):
    funnel = {} if funnel is None else funnel
    seen, buf, pool = set(), [], []  # pool holds (token count, row index)

    def flush():
        # Tokenize in batches of 1,000 so we never keep all 100K long responses in memory.
        for (i, _), t in zip(buf, count_tokens([text for _, text in buf])):
            if t + TEMPLATE_TOKENS <= cutoff:
                pool.append((t, i))
        buf.clear()

    for i, r in enumerate(rows):
        key = " ".join(r["instruction"].lower().split())
        if key in seen or not is_clean_math(r):
            continue
        seen.add(key)
        buf.append((i, prompt_text(r)))
        if len(buf) == 1000:
            flush()
    flush()
    funnel.update(rows=len(rows), clean_math_unique=len(seen), fits_cutoff=len(pool))

    pool = [(t, i) for t, i in pool if t >= min_tokens and (max_tokens is None or t <= max_tokens)]
    funnel["in_length_band"] = len(pool)

    # Sort by length (ties go by row index, so results are reproducible) and drop the shortest part.
    pool.sort()
    pool = [i for _, i in pool[int(len(pool) * drop_short):]]
    funnel["after_drop_short"] = len(pool)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} rows left in the pool, need {n}")
    return [rows[i] for i in sorted(random.Random(seed).sample(pool, n))]


def qwen_token_counter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    return lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=["random", *PRESETS], required=True)
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows", help="read rows from a local JSON list instead of Hugging Face (for testing)")
    p.add_argument("--out", help="output path (default: data/<strategy>_15k.json)")
    p.add_argument("--push", metavar="HF_REPO", help="also upload the result to this public HF dataset repo")
    a = p.parse_args()

    from datasets import Dataset, load_dataset
    rows = json.load(open(a.rows)) if a.rows else load_dataset(SOURCE, split="train")
    count = qwen_token_counter()
    funnel = {}
    print(f"selecting {a.n} rows ({a.strategy}) from {len(rows)}...", flush=True)
    if a.strategy == "random":
        picked = select_random(rows, a.n, a.seed)
    else:
        picked = select_advanced(rows, a.n, a.seed, count, funnel=funnel, **PRESETS[a.strategy])

    recs = [{k: r[k] for k in ("instruction", "input", "output", "system")}
            | {"source": r["extra_info"]["source"], "category": r["extra_info"]["category"]} for r in picked]
    toks = [t for k in range(0, len(recs), 1000) for t in count([prompt_text(r) for r in recs[k:k + 1000]])]
    stats = {
        "strategy": a.strategy, "seed": a.seed, "n": len(recs), "selection": PRESETS.get(a.strategy), "funnel": funnel,
        "categories": Counter(r["category"] for r in recs), "sources": Counter(r["source"] for r in recs),
        "tokens": {"mean": round(statistics.mean(toks)), "median": statistics.median(toks),
                   "p90": sorted(toks)[int(0.9 * len(toks))], "max": max(toks),
                   "truncated_at_cutoff": sum(t + TEMPLATE_TOKENS > CUTOFF for t in toks)},
    }
    out = Path(a.out or Path(__file__).resolve().parents[1] / "data" / f"{a.strategy}_15k.json")
    out.write_text(json.dumps(recs, ensure_ascii=False))
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {out}")
    if a.push:
        Dataset.from_list(recs).push_to_hub(a.push)
        print(f"pushed https://huggingface.co/datasets/{a.push}")


if __name__ == "__main__":
    main()
