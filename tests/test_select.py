import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_data import final_answer, select_advanced, select_random  # noqa: E402

SYS = "Please reason step by step, and put your final answer within \\boxed{}."


def row(q, out, category="math", source="OpenMathReasoning"):
    return {"instruction": q, "input": "", "output": out, "system": SYS,
            "extra_info": {"category": category, "source": source, "index": 0}}


def ans(a, think="x"):
    return f"<think>\n{think}\n</think>\n\nSo the answer is \\boxed{{{a}}}."


def n_tokens(texts):
    # Count characters instead of real tokens so the length checks are easy to follow.
    return [len(t) for t in texts]


def test_final_answer_handles_nested_braces_after_think():
    assert final_answer(ans("\\frac{1}{2}")) == "\\frac{1}{2}"
    assert final_answer("<think>\\boxed{1}</think> no box") is None        # box only inside the thinking
    assert final_answer("no think tag \\boxed{3}") is None                 # no closing </think>
    assert final_answer("<think>x</think> \\boxed{1} and \\boxed{2}") is None  # two boxes, ambiguous
    assert final_answer("<think>x</think> \\boxed{}") is None              # empty box


def test_random_is_reproducible_and_has_no_duplicates():
    rows = [row(f"q{i}", ans(i)) for i in range(50)]
    a, b = select_random(rows, 10, seed=1), select_random(rows, 10, seed=1)
    assert a == b and len(a) == 10 and len({r["instruction"] for r in a}) == 10
    assert select_random(rows, 10, seed=2) != a


def test_advanced_filters_out_bad_samples():
    good = row("Find x.", ans("7"))
    rows = [
        good,
        row("Write a function.", ans("7"), category="code"),
        row("Find y.", "<think>x</think> the answer is 7"),                 # no boxed answer
        row("Prove that x is even.", ans("7")),                             # proof problem
        row("Find z.", ans("\\text{The pentagon is regular.}")),            # answer is a sentence
        row("Find w.", ans("7", think="y" * 500)),                          # too long for the cutoff
        row("find   X.", ans("7")),                                         # duplicate of the good one
    ]
    got = select_advanced(rows, n=1, seed=0, count_tokens=n_tokens, cutoff=400, drop_short=0.0)
    assert got == [good]


def test_advanced_keeps_answers_with_units():
    r = row("How long?", ans("5 \\text{ cm}"))
    assert select_advanced([r], n=1, seed=0, count_tokens=n_tokens, cutoff=400, drop_short=0.0) == [r]


def test_advanced_drops_the_shortest_samples():
    rows = [row(f"q{i}", ans("1", think="t" * (10 * i))) for i in range(10)]
    got = select_advanced(rows, n=8, seed=0, count_tokens=n_tokens, cutoff=10_000, drop_short=0.2)
    assert sorted(r["instruction"] for r in got) == sorted(f"q{i}" for i in range(2, 10))


def test_advanced_errors_when_not_enough_samples():
    try:
        select_advanced([row("Find x.", ans("7"))], n=5, seed=0, count_tokens=n_tokens)
    except ValueError as e:
        assert "pool" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_concise_keeps_only_the_length_range():
    rows = [row(f"q{i}", ans("1", think="t" * (100 * i))) for i in range(10)]  # about 100*i + 110 chars each
    got = select_advanced(rows, n=3, seed=0, count_tokens=n_tokens, cutoff=10_000, drop_short=0.0,
                          min_tokens=400, max_tokens=700)
    assert sorted(r["instruction"] for r in got) == ["q3", "q4", "q5"]

