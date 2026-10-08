# llm-bench

Speed and quality benchmarks for local LLMs on consumer hardware, run through
[Ollama](https://ollama.com) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Two runs so far: a general-quality comparison on an RTX 5070 PC (below), and a
[coding-focused run on an Apple M4 Mac](#apple-m4-coding-models-for-a-16-gb-mac).

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

## Apple M4: coding models for a 16 GB Mac

Which local model is the best compromise between quality and speed for
programming, and how much context does each one actually get?

| | |
|---|---|
| Machine | Apple M4 (10 cores), 16 GB unified memory, ~11.8 GB usable by the GPU |
| OS | macOS 27.0.1 |
| Runtime | Ollama 0.40.0 (Homebrew; flash attention on, `q8_0` KV cache), Python 3.14 |

### Results

50 problems per model: 25 from [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus)
and 25 from [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus), the same
problems for every model. ± is the 95% margin.

| Model | Coding pass@1 | Median time / problem | Gen tok/s (short → 16–20K prompt) | Prompt tok/s | Wait for a 16K prompt | Usable ctx (11.8 GB) | Native ctx | Memory | Thinking |
|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 0.84 ±0.10 | 26 s | 9 → 5 | 22 | 11.9 min | ~25K | 32K | 9.5 GB @8K, 12.8 GB @32K | no |
| `qwen2.5-coder:7b` | 0.80 ±0.11 | 10 s | 14 → 10 | 87 | 3.1 min | 32K | 32K | 5.5 GB | no |
| `gemma4:12b` | 0.78 ±0.11 | 138 s | 14 → 13 | 69 | 3.9 min | ~27K | 256K | 7.8 GB | yes |
| `qwen3.5:9b` | 0.78 ±0.11 | 71 s | 24 → 20 | 140 | 1.9 min | ~51K | 256K | 7.7 GB | yes |
| Empero 9B Q4_K_M | 0.66 ±0.13 | 39 s | 18 → 13 | 122 | 2.2 min | not measured | 256K | 5.9 GB | yes |
| `qwen3.5:4b` | 0.58 ±0.14 | 39 s | 26 → 24 | 212 | 1.3 min | ~104K | 256K | 4.1 GB | yes |

All six support tool calling. Answers that used the full 8,192-token budget
without finishing count as failures: 9 for `qwen3.5:4b`, 4 each for
`gemma4:12b` and `qwen3.5:9b`, 1 for Empero.

- **The top four are tied on quality.** With 50 problems the margin is about
  ±0.11, so 0.78–0.84 is not a real difference. Speed and context decide.
- **`qwen2.5-coder:7b`** is the fastest strong model (no thinking, 10 s per
  problem), but its 32K maximum context is tight for an agent tool that sends a
  large system prompt plus files.
- **`qwen3.5:9b`** has the fastest prompt processing of the strong models and
  about 51K usable context, so it suits agent-style use with large contexts.
  Thinking makes each answer slower.
- **`qwen2.5-coder:14b`** scored highest but doesn't fit at 32K context (10%
  runs on the CPU), so a 16K prompt takes about 12 minutes. Its speeds were
  measured at 32K; capped at ~24K it would be somewhat faster.
- **Empero's distill scores below its Qwen3.5 base on coding** (0.66 vs 0.78),
  within noise, but in the same direction as the open question above.
- `rnj-1:8b` was a candidate but doesn't load in Ollama 0.40.0: llama.cpp
  aborts with `GGML_ASSERT(hparams.is_swa_any())`.
- Models of 24B and up (Devstral Small 2, Qwen3-Coder 30B, Qwen3.6/3.8 27B) were
  left out: at 15–19 GB they don't fit in 16 GB.

### Method

- [`code_bench.py`](code_bench.py) sends each problem to Ollama's native
  `/api/chat`, asks for the solution in a ```` ```python ```` block, and runs it
  against the EvalPlus tests in a subprocess (60 s timeout). lm-eval's code
  tasks aren't used because they prefill the assistant turn, which doesn't work
  well with thinking models behind Ollama. Every reference solution passes the
  harness.
- Sampling is each model's own default (Ollama library parameters; the Empero
  Modelfile above). `num_ctx 16384`, up to 8,192 generated tokens, thinking on
  where the model has it.
- **Median time per problem** is wall time from request to final answer,
  including thinking, so it already reflects how much each model thinks.
- [`ctx_bench.py`](ctx_bench.py) measures prompt processing and generation with
  ~1K, ~8K and ~24K-token prompts of real Python source at `num_ctx 32768`, with
  a random prefix so Ollama's prompt cache can't skip work. **Wait for a 16K
  prompt** is 16,000 tokens divided by prompt tok/s: what an agent pays before
  the first token of a fresh session.
- **Usable ctx** is the context that fits in the ~11.8 GB the GPU can use. It
  is extrapolated from the measured memory cost per token: the size difference
  between 8K and 32K loads for llama.cpp (which allocates the KV cache upfront),
  or the growth of the loaded size across the three prompts for Ollama's MLX
  runner (which allocates as the context fills). It assumes nothing else is
  using memory; with a browser and an editor open, expect less.
  For Empero neither number moved, so it isn't reported; from the
  architecture (8 full-attention layers, `q8_0` KV) it needs ~17 KB per token,
  so 256K plausibly fits.
- The `qwen3.5` library models run on Ollama's MLX runner; Empero and
  `qwen2.5-coder` run on llama.cpp. Some of the speed difference may come from
  the runner rather than the model.

### Reproduce

```bash
brew install ollama && brew services start ollama
for m in qwen3.5:9b qwen3.5:4b gemma4:12b qwen2.5-coder:7b qwen2.5-coder:14b \
         hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M; do ollama pull "$m"; done
ollama create bench-empero9b-q4 -f modelfiles/bench-empero9b-q4

./run_coding.sh                  # results/m4; waits for models still being pulled, resumes
python3 analyze_coding.py        # the table above
```

`code_bench.py` runs code the model wrote on your machine (in a temp directory,
with a timeout). A full run took about 8.5 hours on the M4, most of it on the
thinking models.

## Layout

```
modelfiles/        Ollama Modelfiles that pin context length, sampling and seed
run_eval.py        lm-eval runner for Ollama (clears stop sequences, caches requests)
run_all.sh         The full sequential run
analyze.py         Builds the result tables from results/quality-*.json
code_bench.py      HumanEval+/MBPP+ through Ollama; runs the tests, records timing
ctx_bench.py       Prompt/generation speed at depth and usable context per model
run_coding.sh      The coding run (ctx_bench + code_bench per model)
analyze_coding.py  Builds the coding table from results/m4/
results/           speed-*.json from llmfit, quality-*.json from lm-eval (with per-sample logs)
results/m4/        ctx-*.json and code-*.jsonl (every answer, pass/fail, timing) from the M4 run
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
