"""Sample memory use during a benchmark run and write one CSV row per interval.

    python mem_monitor.py --out results/mem-quality.csv [--interval 5] [--watch-pid PID]

Runs until killed, or until --watch-pid exits; run_all.sh and run_coding.sh
start it in the background, kill it on exit, and pass their own pid so it also
stops if they are killed without running their exit trap. Rows are appended,
so a resumed run keeps its earlier samples.

Columns:
  models          models loaded in Ollama (/api/ps), ";"-separated
  ollama_size_gb  Ollama's own size for the loaded models (weights + KV + buffers)
  ollama_vram_gb  the part of that Ollama placed on the GPU
  gpu_used_gb     VRAM in use on the whole GPU (nvidia-smi; blank without it, e.g. Apple)
  ram_avail_gb    system RAM still available (psutil)
  ollama_rss_gb   resident memory of Ollama's processes: ollama* (server) and
                  llama-server* (the llama.cpp runner newer Ollama versions spawn)

ollama_size/vram are Ollama's estimate; gpu_used and ollama_rss are what the OS
reports. On Apple unified memory the model shows up in ollama_rss, not gpu_used.

peaks(csv) summarizes a file per model for the analyze scripts as memory *used*:
how far VRAM rose and free RAM fell compared with the idle samples just before
the model loaded. That takes out the desktop and other apps, and on Apple the
RAM figure alone covers the model, since the GPU shares system memory.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

PS_URL = "http://127.0.0.1:11434/api/ps"  # "localhost" costs ~2 s on Windows trying ::1 first
OLLAMA_PROCS = ("ollama", "llama-server")  # name prefixes: server, llama.cpp runner
FIELDS = ["time", "models", "ollama_size_gb", "ollama_vram_gb", "gpu_used_gb", "ram_avail_gb", "ollama_rss_gb"]


def gb(n):
    return round(n / 1e9, 2)


def ollama_ps():
    try:
        with urllib.request.urlopen(PS_URL, timeout=10) as r:
            models = json.loads(r.read())["models"]
    except Exception:
        return "", "", ""  # Ollama down or restarting; keep sampling the rest
    names = ";".join(m["name"].removesuffix(":latest") for m in models)
    return names, gb(sum(m["size"] for m in models)), gb(sum(m["size_vram"] for m in models))


def gpu_used():
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return gb(sum(int(l) for l in out.split()) * 2**20)  # MiB, summed over GPUs
    except Exception:
        return ""


def ram():
    if psutil is None:
        return "", ""
    rss = 0
    # only ask the ollama processes for memory_info; asking every process is slow on Windows
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower().startswith(OLLAMA_PROCS):
            try:
                rss += p.memory_info().rss
            except psutil.Error:
                pass  # a runner exiting between the scan and the query
    return gb(psutil.virtual_memory().available), gb(rss)


def alive(pid):
    if psutil is not None:
        return psutil.pid_exists(pid)
    if os.name == "nt":
        return True  # os.kill(pid, 0) on Windows would send CTRL_C_EVENT, not probe
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def sample():
    models, size, vram = ollama_ps()
    avail, rss = ram()
    return {
        "time": datetime.now().isoformat(timespec="seconds"), "models": models,
        "ollama_size_gb": size, "ollama_vram_gb": vram, "gpu_used_gb": gpu_used(),
        "ram_avail_gb": avail, "ollama_rss_gb": rss,
    }


BASELINE_WINDOW_S = 60  # idle samples this recent set the baseline
LOAD_WINDOW_S = 30  # "at load" = the first this many seconds after each load


def peaks(path):
    """model -> {"vram": peak VRAM used, "ram": peak system RAM used,
    "ram_load": system RAM used right after loading, "size": peak ollama_size_gb},
    in GB, over the rows where that model was loaded.

    ram_load and ram differ because llama.cpp (Ollama's runner) keeps a prompt
    cache in system RAM, up to 8 GiB by default, which fills as prompts go by.

    "Used" is relative to the idle rows (nothing loaded) in the minute before:
    the lowest gpu_used_gb and the highest ram_avail_gb among them. Lowest and
    highest because Ollama drops a model from /api/ps a few seconds before its
    runner has released the memory. Values are None when the column was blank
    or no idle row came first."""
    acc = defaultdict(lambda: {"vram": None, "ram": None, "ram_load": None, "size": None})
    path = Path(path)
    if not path.exists():
        return {}
    idle = []  # (time, gpu_used, ram_avail) of recent idle rows
    loaded_at = {}  # model -> time it appeared, while it stays loaded
    num = lambda v: float(v) if v else None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = datetime.fromisoformat(row["time"])
            gpu, avail = num(row["gpu_used_gb"]), num(row["ram_avail_gb"])
            models = list(filter(None, row["models"].split(";")))
            loaded_at = {m: loaded_at.get(m, t) for m in models}
            if not models:
                if row["ollama_size_gb"] != "":  # blank means Ollama didn't answer, not idle
                    idle = [i for i in idle if (t - i[0]).total_seconds() <= BASELINE_WINDOW_S]
                    idle.append((t, gpu, avail))
                continue
            gpu_base = min((i[1] for i in idle if i[1] is not None), default=None)
            ram_base = max((i[2] for i in idle if i[2] is not None), default=None)
            used = {
                "vram": gpu - gpu_base if None not in (gpu, gpu_base) else None,
                "ram": ram_base - avail if None not in (avail, ram_base) else None,
                "size": num(row["ollama_size_gb"]),
            }
            for m in models:
                a = acc[m]
                just_loaded = (t - loaded_at[m]).total_seconds() <= LOAD_WINDOW_S
                for key, v in (*used.items(), ("ram_load", used["ram"] if just_loaded else None)):
                    if v is not None:
                        a[key] = v if a[key] is None else max(a[key], v)
    return dict(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5, help="seconds between samples")
    ap.add_argument("--watch-pid", type=int,
                    help="exit when this process is gone (the run script; under Git Bash pass its Windows pid)")
    args = ap.parse_args()
    if psutil is None:
        print("mem_monitor: psutil not installed (pip install psutil); RAM columns stay blank", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
            f.flush()
        # If the run script is killed without its exit trap firing, don't keep
        # sampling forever.
        while args.watch_pid is None or alive(args.watch_pid):
            w.writerow(sample())
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
