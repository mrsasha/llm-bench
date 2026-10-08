"""Build the coding comparison table from run_coding.sh output.

    python analyze_coding.py [results-dir]      (default results/m4)
"""

import json
import math
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results" / "m4")


def k(n):
    return f"{n // 1024}K" if n else "-"


def rate(vals):
    if not vals:
        return "-"
    p = sum(vals) / len(vals)
    return f"{p:.2f} ±{1.96 * math.sqrt(p * (1 - p) / len(vals)):.2f} (n={len(vals)})"


rows = []
for ctx_file in sorted(d.glob("ctx-*.json")):
    c = json.loads(ctx_file.read_text())
    slug = ctx_file.stem.removeprefix("ctx-")
    code_file = d / f"code-{slug}.jsonl"
    code = [json.loads(l) for l in code_file.read_text().splitlines() if l.strip()] if code_file.exists() else []
    he = [r["passed"] for r in code if r["id"].startswith("HumanEval")]
    mb = [r["passed"] for r in code if r["id"].startswith("Mbpp")]
    sp = sorted(c["speed"], key=lambda s: s["prompt_tokens"])
    short, deep = (sp[0], sp[-1]) if sp else (None, None)
    rows.append({
        "model": c["model"],
        "all": rate(he + mb), "he": rate(he), "mb": rate(mb),
        "score": sum(he + mb) / len(he + mb) if code else -1,
        "t": f"{statistics.median(r['wall_s'] for r in code):.0f}s" if code else "-",
        "gen": f"{short['gen_tps']:.0f}" if short else "-",
        "gen_deep": f"{deep['gen_tps']:.0f} @{deep['prompt_tokens'] // 1000}K" if deep else "-",
        "pp": f"{deep['prompt_tps']:.0f}" if deep else "-",
        # what a coding agent pays before the first token of a fresh ~16K-token session
        "t16k": f"{16000 / deep['prompt_tps'] / 60:.1f} min" if deep else "-",
        "ctx": k(c["est_max_ctx"]),
        "native": k(c["native_ctx"]),
        "mem": f"{c['size_32k_gb']:.1f} GB",
        "tools": "yes" if "tools" in c["capabilities"] else "no",
        "think": "yes" if "thinking" in c["capabilities"] else "no",
        "trunc": f"{sum(r['done_reason'] == 'length' for r in code)}/{len(code)}" if code else "-",
    })

rows.sort(key=lambda r: r["score"], reverse=True)
cols = [
    ("model", "Model"), ("all", "Coding pass@1"), ("he", "HumanEval+"), ("mb", "MBPP+"),
    ("t", "Median time / problem"), ("gen", "Gen tok/s"), ("gen_deep", "Gen tok/s, long prompt"),
    ("pp", "Prompt tok/s"), ("t16k", "16K prompt wait"), ("ctx", "Est. max ctx (11.8 GB)"),
    ("native", "Native ctx"), ("mem", "Memory"), ("tools", "Tools"), ("think", "Thinking"),
    ("trunc", "Hit token limit"),
]
print("| " + " | ".join(h for _, h in cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r[c]) for c, _ in cols) + " |")
