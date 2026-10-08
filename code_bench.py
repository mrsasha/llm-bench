"""Coding benchmark: HumanEval+ and MBPP+ through Ollama's native chat API.

lm-eval's code tasks prefill the assistant turn (gen_prefix), which doesn't mix
well with thinking models behind Ollama. This asks for a complete solution in a
```python block, runs it against the EvalPlus tests in a subprocess, and records
wall time and token counts so quality and speed come from the same answers.

Each finished problem is appended to the --out JSONL file; a rerun skips
problems already in it, so an interrupted run resumes.

WARNING: this executes model-generated code on this machine (subprocess, temp
dir, timeout). Only run it with models you trust to that degree.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from datasets import load_dataset

OLLAMA_URL = "http://localhost:11434/api/chat"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True, help="JSONL file, appended to")
ap.add_argument("--n-humaneval", type=int, default=40, help="problems, evenly spaced over 164")
ap.add_argument("--n-mbpp", type=int, default=40, help="problems, evenly spaced over 378")
ap.add_argument("--max-tokens", type=int, default=8192, help="thinking + answer budget")
ap.add_argument("--test-timeout", type=int, default=60, help="seconds per test run")
args = ap.parse_args()


def spaced(n_total, n):
    """n evenly spaced indices, the same for every model."""
    n = min(n, n_total)
    return [round(i * (n_total - 1) / max(n - 1, 1)) for i in range(n)]


def problems():
    he = load_dataset("evalplus/humanevalplus", split="test")
    for i in spaced(len(he), args.n_humaneval):
        d = he[i]
        yield {
            "id": d["task_id"],
            "prompt": (
                "Complete this Python function. Reply with the full function "
                "(including the signature and any imports) in one ```python code block.\n\n"
                f"```python\n{d['prompt']}```"
            ),
            "test": f"{d['test']}\n\ncheck({d['entry_point']})\n",
        }
    mb = load_dataset("evalplus/mbppplus", split="test")
    for i in spaced(len(mb), args.n_mbpp):
        d = mb[i]
        yield {
            "id": f"Mbpp/{d['task_id']}",
            "prompt": (
                f"{d['prompt']}\nYour code should pass this test:\n{d['test_list'][0]}\n\n"
                "Reply with the solution in one ```python code block."
            ),
            "test": "\n".join(d["test_imports"]) + "\n" + d["test"] + "\n",
        }


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        # the solution is usually the longest block; examples/tests are shorter
        return max(blocks, key=len)
    return text


def run_tests(code, test):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "t.py"
        f.write_text(code + "\n\n" + test, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(f)],
                cwd=tmp, capture_output=True, text=True, timeout=args.test_timeout,
            )
            return r.returncode == 0, r.stderr[-500:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def chat(prompt):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # sampling comes from the model's own defaults (Modelfile / library params)
        "options": {"num_predict": args.max_tokens, "num_ctx": 16384},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}

for p in problems():
    if p["id"] in done:
        continue
    t0 = time.time()
    resp = chat(p["prompt"])
    wall = time.time() - t0
    msg = resp["message"]
    answer = msg.get("content", "")
    passed, err = run_tests(extract_code(answer), p["test"]) if answer.strip() else (False, "empty answer")
    rec = {
        "id": p["id"],
        "passed": passed,
        "error": "" if passed else err,
        "wall_s": round(wall, 2),
        "eval_count": resp.get("eval_count", 0),
        "eval_s": resp.get("eval_duration", 0) / 1e9,
        "prompt_count": resp.get("prompt_eval_count", 0),
        "prompt_s": resp.get("prompt_eval_duration", 0) / 1e9,
        "done_reason": resp.get("done_reason"),
        "thinking_chars": len(msg.get("thinking") or ""),
        "answer": answer,
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{p['id']:<16} {'PASS' if passed else 'fail'} {wall:6.1f}s {rec['eval_count']:5d} tok", flush=True)
