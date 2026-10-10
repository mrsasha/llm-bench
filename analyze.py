"""Summarize quality-*.json files into one table, plus a head-to-head on the
smaller subset that the 27B model also ran (same doc ids for every model)."""

import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from mem_monitor import peaks

RESULTS = Path(__file__).parent / "results"

def as_float(v):
    return float(v == "True") if v in ("True", "False") else float(v)


def sample_score(task, s):
    if task.startswith("ifeval"):
        return as_float(s["prompt_level_strict_acc"])
    return as_float(s["exact_match"])


def family(task):
    if task.startswith("mmlu"):
        return "mmlu_cot"
    return task


def load():
    """model -> family -> {(task, doc_id): score}; also empty-answer counts."""
    scores = defaultdict(lambda: defaultdict(dict))
    empty = defaultdict(lambda: [0, 0])
    for f in sorted(RESULTS.glob("quality-*.json")):
        model = f.stem.removeprefix("quality-").rsplit("-", 1)[0]
        model = model.removesuffix("-full").removesuffix("-sub")
        data = json.loads(f.read_text(encoding="utf-8"))
        for task, samples in data.get("samples", {}).items():
            by_doc = defaultdict(list)
            for s in samples:
                by_doc[s["doc_id"]].append(s)
            for doc_id, group in by_doc.items():
                # mmlu/gsm8k log one entry per filter; prefer flexible-extract
                pick = next((g for g in group if g.get("filter") == "flexible-extract"), group[0])
                scores[model][family(task)][(task, doc_id)] = sample_score(task, pick)
                resps = pick["resps"]
                resps = ast.literal_eval(resps) if isinstance(resps, str) else resps
                resp = resps[0][0] if resps and resps[0] else ""
                empty[model][0] += not str(resp).strip()
                empty[model][1] += 1
    return scores, empty


def fmt(vals):
    if not vals:
        return "-"
    p = sum(vals) / len(vals)
    se = math.sqrt(p * (1 - p) / len(vals))
    return f"{p:.2f} +/-{1.96 * se:.2f} (n={len(vals)})"


def main():
    scores, empty = load()
    fams = sorted({f for m in scores.values() for f in m})
    models = sorted(scores)
    print("## Full run (95% margin shown)\n")
    print("| model | " + " | ".join(fams) + " | empty answers |")
    print("|---" * (len(fams) + 2) + "|")
    for m in models:
        row = [fmt(list(scores[m][f].values())) for f in fams]
        e, n = empty[m]
        print(f"| {m} | " + " | ".join(row) + f" | {e}/{n} |")

    ref = next((m for m in models if "27b" in m), None)
    if ref:
        print(f"\n## Same questions as {ref}\n")
        print("| model | " + " | ".join(fams) + " |")
        print("|---" * (len(fams) + 1) + "|")
        for m in models:
            row = []
            for f in fams:
                keys = scores[ref][f].keys() & scores[m][f].keys()
                row.append(fmt([scores[m][f][k] for k in keys]))
            print(f"| {m} | " + " | ".join(row) + " |")

    # written by run_all.sh through mem_monitor.py; absent for older runs
    mem = peaks(RESULTS / "mem-quality.csv")
    if mem:
        gbs = lambda v: f"{v:.1f} GB" if v is not None else "-"
        print("\n## Memory used (on top of the idle machine)\n")
        print("| model | Ollama size | VRAM used | RAM used at load | RAM used, peak |")
        print("|---|---|---|---|---|")
        for m in sorted(mem):
            p = mem[m]
            print(f"| {m} | {gbs(p['size'])} | {gbs(p['vram'])} | {gbs(p['ram_load'])} | {gbs(p['ram'])} |")


if __name__ == "__main__":
    sys.exit(main())
