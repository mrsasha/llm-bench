"""Context window and speed at depth, through Ollama's native API.

For each model:
  1. native context length and capabilities (tools, thinking) from /api/show
  2. prompt-processing and generation tok/s with ~1K, ~8K and ~24K-token prompts
     of real Python source (what a coding agent sends), at num_ctx 32K
  3. memory cost per context token, and from it the largest context that fits in
     the GPU memory budget (--gpu-budget-gb, Metal's working-set limit)

Measuring (3) depends on the runner. llama.cpp allocates the KV cache for the
whole num_ctx at load time, so the cost is the size difference between an 8K
and a 32K load. Ollama's MLX runner allocates as the context fills, so the cost
is the slope of /api/ps size over the 1K/8K/24K-token prompts.

Each prompt starts with a random nonce so Ollama's prompt cache can't skip work.
Writes one JSON file per model.
"""

import argparse
import json
import random
import sysconfig
import urllib.request
from pathlib import Path

BASE = "http://localhost:11434"
BENCH_CTX = 32768
PROMPT_TOKENS = [1000, 8000, 24000]
CHARS_PER_TOKEN = 3.2  # rough for Python source; actual counts come back from Ollama

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu-budget-gb", type=float, default=11.84,
                help="GPU-usable unified memory (llmfit system reports it)")
args = ap.parse_args()


def post(path, body, timeout=3600):
    body = {"stream": False, **body}
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def loaded():
    with urllib.request.urlopen(BASE + "/api/ps", timeout=30) as r:
        models = json.loads(r.read())["models"]
    names = {args.model, args.model + ":latest"}  # Modelfile-created models get :latest
    return next(m for m in models if names & {m["name"], m["model"]})


def unload():
    post("/api/generate", {"model": args.model, "keep_alive": 0})


def load(num_ctx):
    unload()
    post("/api/generate", {"model": args.model, "prompt": "hi", "options": {"num_ctx": num_ctx, "num_predict": 1}, **opts})
    return loaded()


def python_source(n_chars, seed):
    files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))
    random.Random(seed).shuffle(files)
    buf, size = [], 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        buf.append(f"# file: {f.name}\n{text}")
        size += len(text)
        if size >= n_chars:
            break
    return "\n".join(buf)[:n_chars]


show = post("/api/show", {"model": args.model})
info = show.get("model_info", {})
native_ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
caps = show.get("capabilities", [])
opts = {"think": False} if "thinking" in caps else {}

small = load(8192)
big = load(BENCH_CTX)
result = {
    "model": args.model, "native_ctx": native_ctx, "capabilities": caps,
    "runner": big.get("runner"), "size_8k_gb": round(small["size"] / 1e9, 2),
    "size_32k_gb": round(big["size"] / 1e9, 2), "gpu_pct": round(100 * big["size_vram"] / big["size"]),
    "speed": [],
}
print(f"runner {result['runner']}: {result['size_8k_gb']} GB @8K, {result['size_32k_gb']} GB @32K, {result['gpu_pct']}% GPU", flush=True)

for n in PROMPT_TOKENS:
    nonce = random.randrange(10**9)
    prompt = (f"[session {nonce}]\n" + python_source(int(n * CHARS_PER_TOKEN), seed=nonce)
              + "\n\nSummarize what the code above does in five bullet points.")
    r = post("/api/generate", {
        "model": args.model, "prompt": prompt,
        "options": {"num_ctx": BENCH_CTX, "num_predict": 256, "temperature": 0}, **opts,
    })
    pp = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
    tg = r["eval_count"] / (r["eval_duration"] / 1e9)
    result["speed"].append({
        "prompt_tokens": r["prompt_eval_count"], "prompt_tps": round(pp, 1),
        "gen_tokens": r["eval_count"], "gen_tps": round(tg, 1),
        "size_after_gb": round(loaded()["size"] / 1e9, 3),
    })
    print(f"prompt {r['prompt_eval_count']:>6} tok: pp {pp:7.1f} tok/s, gen {tg:5.1f} tok/s", flush=True)
after = loaded()
result["size_filled_gb"] = round(after["size"] / 1e9, 2)

prealloc = big["size"] - small["size"]
if prealloc > 50e6:  # llama.cpp style: KV sized by num_ctx at load
    per_tok = prealloc / (BENCH_CTX - 8192)
    weights = small["size"] - 8192 * per_tok
else:  # MLX style: KV grows with the filled context, allocated in chunks, so fit a
    # line through (filled tokens, size) for the loaded model and each prompt size
    pts = [(1, big["size"])] + [(s["prompt_tokens"] + s["gen_tokens"], s["size_after_gb"] * 1e9) for s in result["speed"]]
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    per_tok = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    weights = big["size"]
if per_tok < 1024:
    # neither the load size nor the filled size moved (seen with a llama.cpp GGUF
    # whose reported size ignores num_ctx), so there is nothing to extrapolate from
    result["kv_kb_per_token"] = result["est_max_ctx"] = None
    print("KV cache size not measurable; est. max ctx not reported", flush=True)
else:
    result["kv_kb_per_token"] = round(per_tok / 1024, 1)
    fit = int((args.gpu_budget_gb * 1e9 - weights) / per_tok)
    result["est_max_ctx"] = min(fit, native_ctx) if native_ctx else fit
    print(f"~{per_tok / 1024:.0f} KB/token -> est. max ctx {result['est_max_ctx']} in {args.gpu_budget_gb} GB", flush=True)

unload()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
