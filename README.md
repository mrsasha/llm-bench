# llm-bench

Speed and quality benchmarks for local LLMs on a consumer GPU, run through
[Ollama](https://ollama.com) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Tools like [llmfit](https://github.com/AlexsJones/llmfit) and
[llm-checker](https://www.npmjs.com/package/llm-checker) tell you which models
*fit* your hardware and how fast they should run. They don't tell you how *good*
a model is, and they score community fine-tunes the same as the base model they
were built from. This project measures both on real hardware.

The first question it answers: **are the [Empero](https://empero.org)
"Qwen3.8 distill" models actually better than the Qwen models they are built on?**
[`empero-ai/Qwen3.8-9B-Distill-GGUF`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF)
is Qwen3.5-9B fine-tuned on ~70k reasoning traces from Qwen3.8 (2.4T A95B).
Empero reports these numbers on its model card:

| Task (Empero's own eval) | Qwen3.5-9B | Qwen3.8-9B-Distill |
|---|---|---|
| MMLU (CoT, 57 subjects) | 0.546 | 0.751 |
| GSM8K CoT | 0.885 | 0.870 |

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12 GB |
| CPU | Intel Core Ultra 7 265KF, 20 cores |
| RAM | 32 GB |
| OS | Windows 10 Pro |
| Runtime | Ollama 0.35.1, lm-eval 0.4.14.dev0 |

## Models

| Ollama name | Model | Quant | Notes |
|---|---|---|---|
| `bench-empero9b-q4` | Empero Qwen3.8-9B-Distill | Q4_K_M | Same quant as the baseline |
| `bench-empero9b-q5` | Empero Qwen3.8-9B-Distill | Q5_K_M | Speed only |
| `bench-qwen35-9b` | Qwen3.5 9B (`qwen3.5:latest`) | Q4_K_M | Baseline Empero was trained from |
| `bench-gemma4-12b` | Gemma 4 12B (`gemma4:12b`) | Q4_K_M | What llmfit and llm-checker usually recommend |
| `bench-qwen38-27b` | Qwen3.8 27B (`qwen3.8:27b-q4_K_M`) | Q4_K_M | Smallest official Qwen3.8; ~half runs on CPU |

## Results

### Speed

Generation speed from `llmfit bench --provider ollama` (3 runs; 2 runs for the 27B), 300 output tokens.

| Model | tokens/s | Time to first token | Runs on |
|---|---:|---:|---|
| Qwen3.5 9B Q4_K_M | 100.2 | 45 ms | 100% GPU |
| Empero 9B Q4_K_M | 98.3 | 49 ms | 100% GPU |
| Empero 9B Q5_K_M | 88.4 | 47 ms | 100% GPU |
| Gemma 4 12B Q4_K_M | 68.3 | 56 ms | 100% GPU |
| Qwen3.8 27B Q4_K_M | 7.1 | 962 ms | 52% GPU / 48% CPU |

Empero's distill has the same speed as the Qwen3.5 model it was built from.

### Quality

Ranges are 95% margins. GSM8K and MMLU use lm-eval's `flexible-extract` filter.
IFEval is prompt-level strict accuracy.

| Model | GSM8K CoT (n=50) | IFEval (n=50) | MMLU CoT (n=57, 1 per subject) |
|---|---|---|---|
| Empero 9B Q4_K_M | 0.94 ± 0.07 | 0.80 ± 0.11 | 0.84 ± 0.09 |
| Qwen3.5 9B Q4_K_M | *pending* | *pending* | *pending* |
| Gemma 4 12B Q4_K_M | *pending* | *pending* | *pending* |
| Qwen3.8 27B Q4_K_M | *pending (smaller subset)* | | |

Early observation: 6 of Empero's 50 IFEval prompts came back with an empty
answer, most likely because the model used the whole 8,192-token budget on
thinking. Those 6 are 6 of its 10 IFEval misses.

Don't read too much into a single column until the baselines are in. With
50–57 questions per test, differences under ~10 points are within noise.

## Method

- Every model runs from an Ollama Modelfile in [`modelfiles/`](modelfiles/) with
  `num_ctx 16384` and `seed 42`.
- The Qwen-family models use Empero's recommended sampling: `temperature 0.6`,
  `top_p 0.95`, `top_k 20`. Gemma uses Google's defaults: `temperature 1.0`, `top_p 0.95`, `top_k 64`.
- Thinking is on (each model's default), with up to 8,192 generated tokens per answer.
- lm-eval talks to Ollama's OpenAI-compatible endpoint (`local-chat-completions`)
  with the chat template applied and few-shot examples sent as multi-turn.
- The 27B model runs on a smaller subset: 15 GSM8K, 15 IFEval, and 15 MMLU
  subjects. [`analyze.py`](analyze.py) also scores every model on exactly those
  questions, so the comparison is like for like.

### Pitfalls this setup works around

- **Stop sequences truncate thinking.** lm-eval's GSM8K and MMLU configs stop
  generation at `Q:`. Ollama applies stop sequences to the reasoning as well as the
  answer, so thinking models are cut off mid-thought and return an empty answer.
  [`run_eval.py`](run_eval.py) clears the stop sequences. The `--gen_kwargs` CLI flag
  can't do this: it turns `until=[]` into the literal string `"[]"`.
- **The tasks force greedy decoding.** Their `do_sample: false` becomes
  `temperature 0`, which Qwen advises against for thinking mode because it can cause
  endless repetition. The runner sets sampling explicitly.
- **No parallel requests.** Ollama logs `model architecture does not currently
  support parallel requests` for the `qwen35` architecture, so runs are sequential.
- **Interrupted runs.** Results are only written at the end of a run, so the runner
  keeps an lm-eval request cache in `cache/` and a rerun picks up where it stopped.
  `run_all.sh` also skips any stage whose `results/quality-*.json` already exists;
  delete that file to rerun the stage.
- **Ollama going away.** `run_all.sh` refuses to start if Ollama isn't reachable.
  If Ollama stops mid-run (an update or memory pressure), it waits up to 10 minutes
  before the next stage instead of letting every remaining stage fail.

## Reproduce

Prerequisites: Ollama, Python 3, and lm-evaluation-harness with the IFEval
dependencies (`pip install langdetect immutabledict nltk`). `llmfit` is optional,
for the speed numbers.

```bash
# 1. Pull the models
ollama pull hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M
ollama pull hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q5_K_M
ollama pull qwen3.5:latest
ollama pull gemma4:12b
ollama pull qwen3.8:27b-q4_K_M

# 2. Create the benchmark variants
for f in modelfiles/*; do ollama create "$(basename "$f")" -f "$f"; done

# 3. Speed
llmfit bench bench-empero9b-q4 --provider ollama --runs 3

# 4. Quality (several hours; safe to interrupt and rerun)
./run_all.sh                # resume: skips stages that already have results
./run_all.sh --fresh        # delete results, logs and cache, then start over (asks first)
./run_all.sh --fresh --yes  # same, no prompt

# 5. Summary tables
python analyze.py
```

To run one model and task by hand:

```bash
python run_eval.py --model bench-empero9b-q4 --tasks gsm8k_cot --limit 50 \
  --out results/quality-bench-empero9b-q4-full.json
```

## Layout

```
modelfiles/   Ollama Modelfiles that pin context length, sampling and seed
run_eval.py   lm-eval runner for Ollama (clears stop sequences, caches requests)
run_all.sh    The full sequential run
analyze.py    Builds the result tables from results/quality-*.json
results/      speed-*.json from llmfit, quality-*.json from lm-eval (with per-sample logs)
```

## Caveats

- Small samples. Treat these numbers as a sanity check of the published claims,
  not as a leaderboard.
- Single machine, single seed, Q4 quantization. Empero's own numbers come from an
  unpublished harness and likely a different precision.
- The 27B model doesn't fit in 12 GB of VRAM, so its speed reflects CPU offload,
  not the model itself.

## License

[MIT](LICENSE)
