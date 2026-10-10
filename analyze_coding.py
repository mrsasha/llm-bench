"""Build the coding tables from run_coding.sh output.

    python analyze_coding.py [results-dir]          one machine (default results/m4)
    python analyze_coding.py DIR DIR [DIR ...]      machines side by side, e.g.
                                                    results/m4 results/rtx5070

Each results dir is one machine's run; its name labels the machine.
"""

import json
import math
import statistics
import sys
from pathlib import Path

from mem_monitor import peaks


def k(n):
    return f"{n // 1024}K" if n else "-"


def gbs(v):
    return f"{v:.1f} GB" if v is not None else "-"


def rate(vals):
    if not vals:
        return "-"
    p = sum(vals) / len(vals)
    return f"{p:.2f} ±{1.96 * math.sqrt(p * (1 - p) / len(vals)):.2f} (n={len(vals)})"


def mem_used(m):
    """Memory the model took on top of the idle machine. RAM shows at load and at
    peak; the peak includes llama.cpp's prompt cache (see mem_monitor.peaks)."""
    if not m or m.get("ram") is None:
        return "-"
    ram = f"{gbs(m['ram_load'])} at load, {gbs(m['ram'])} peak"
    if m.get("vram") is None:  # no discrete GPU (Apple): RAM is the whole story
        return ram
    return f"{gbs(m['vram'])} VRAM + RAM {ram}"


def load(d):
    """One row per model in results dir d."""
    mem = peaks(d / "mem.csv")  # empty for runs made before mem_monitor existed
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
        m = mem.get(c["model"], {})
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
            "vram_used": gbs(m.get("vram")),
            "ram_used": f"{gbs(m['ram_load'])} / {gbs(m['ram'])}" if m.get("ram") is not None else "-",
            "mem_used": mem_used(m),
            "runner": c.get("runner", "-"),
            "tools": "yes" if "tools" in c["capabilities"] else "no",
            "think": "yes" if "thinking" in c["capabilities"] else "no",
            "trunc": f"{sum(r['done_reason'] == 'length' for r in code)}/{len(code)}" if code else "-",
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def table(cols, rows):
    print("| " + " | ".join(h for _, h in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c, _ in cols) + " |")


def one(d):
    # older ctx files predate gpu_budget_gb; those were all M4 runs at the 11.84 default
    budget = max((json.loads(f.read_text()).get("gpu_budget_gb", 11.84) for f in d.glob("ctx-*.json")), default=11.84)
    table([
        ("model", "Model"), ("all", "Coding pass@1"), ("he", "HumanEval+"), ("mb", "MBPP+"),
        ("t", "Median time / problem"), ("gen", "Gen tok/s"), ("gen_deep", "Gen tok/s, long prompt"),
        ("pp", "Prompt tok/s"), ("t16k", "16K prompt wait"), ("ctx", f"Est. max ctx ({budget:.1f} GB)"),
        ("native", "Native ctx"), ("mem", "Memory"),
        ("vram_used", "VRAM used"), ("ram_used", "RAM used (load / peak)"),
        ("tools", "Tools"), ("think", "Thinking"),
        ("trunc", "Hit token limit"),
    ], load(d))


def compare(dirs):
    by_machine = {d.name: {r["model"]: r for r in load(d)} for d in dirs}
    models = {m for rows in by_machine.values() for m in rows}
    # best first, by the mean score over the machines that ran the model
    mean = lambda m: statistics.mean(rows[m]["score"] for rows in by_machine.values() if m in rows)
    out = []
    for model in sorted(models, key=mean, reverse=True):
        for i, (machine, rows) in enumerate(by_machine.items()):
            if model in rows:
                out.append({**rows[model], "model": model if i == 0 else "", "machine": machine})
    table([
        ("model", "Model"), ("machine", "Machine"), ("runner", "Runner"), ("all", "Coding pass@1"),
        ("t", "Median time / problem"), ("gen", "Gen tok/s"), ("pp", "Prompt tok/s"),
        ("t16k", "16K prompt wait"), ("mem_used", "Memory used"), ("trunc", "Hit token limit"),
    ], out)


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]] or [Path(__file__).parent / "results" / "m4"]
    one(dirs[0]) if len(dirs) == 1 else compare(dirs)
