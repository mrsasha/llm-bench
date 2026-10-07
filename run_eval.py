"""Run lm-eval against an Ollama model through its OpenAI-compatible chat endpoint.

Thinking models put their reasoning in a separate `reasoning` field, and Ollama
applies stop sequences to the reasoning too. The stock task configs stop on
"Q:", which truncates the reasoning and leaves an empty answer, so stop
sequences are cleared here. --gen_kwargs on the CLI can't express an empty list.
"""

import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
ap.add_argument("--limit", type=int, required=True, help="samples per (sub)task")
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--max-gen-toks", type=int, default=8192)
ap.add_argument("--concurrent", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

results = simple_evaluate(
    model="local-chat-completions",
    model_args={
        "model": args.model,
        "base_url": OLLAMA_URL,
        "num_concurrent": args.concurrent,
        "max_retries": 3,
        "timeout": 3600,
        "tokenized_requests": False,
    },
    tasks=args.tasks.split(","),
    limit=args.limit,
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    gen_kwargs={
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_gen_toks": args.max_gen_toks,
        "until": [],
    },
    log_samples=True,
    # sqlite cache of finished requests, so an interrupted run resumes instead of restarting
    use_cache=str(Path(__file__).parent / "cache" / args.model),
    random_seed=42,
    numpy_random_seed=42,
    torch_random_seed=42,
    fewshot_random_seed=42,
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2, default=handle_non_serializable), encoding="utf-8")
for task, metrics in results["results"].items():
    print(task, {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
