# llm-bench

Speed and quality benchmarks for local LLMs on consumer hardware, run through
[Ollama](https://ollama.com) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Three runs so far: a general-quality comparison on an RTX 5070 PC (below), a
[coding-focused run on an Apple M4 Mac](#apple-m4-coding-models-for-a-16-gb-mac),
and [the same coding run on the PC](#rtx-5070-pc-the-same-coding-run), with a
[PC vs Mac comparison](#pc-vs-mac).

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
| Runtime | Ollama 0.35.1 (Empero quality run), 0.40.0 (everything else); lm-eval 0.4.14.dev0 |

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

| Model | GSM8K CoT (n=50) | IFEval (n=50) | MMLU CoT (n=57, 1 per subject) | Empty answers |
|---|---|---|---|---|
| Empero 9B Q4_K_M | 0.94 ± 0.07 | 0.80 ± 0.11 | 0.84 ± 0.09 | 6/157 |
| Qwen3.5 9B Q4_K_M | 0.94 ± 0.07 | 0.80 ± 0.11 | 0.89 ± 0.08 | 8/157 |
| Gemma 4 12B Q4_K_M | 0.90 ± 0.08 | 0.82 ± 0.11 | 0.91 ± 0.07 | 12/157 |
| Qwen3.8 27B Q4_K_M (n=15 each) | 0.93 ± 0.13 | 0.87 ± 0.17 | 0.87 ± 0.17 | 0/45 |

The same 15 + 15 + 15 questions the 27B ran, for every model:

| Model | GSM8K CoT | IFEval | MMLU CoT |
|---|---|---|---|
| Empero 9B Q4_K_M | 0.93 | 0.80 | 0.80 |
| Qwen3.5 9B Q4_K_M | 0.93 | 0.93 | 0.87 |
| Gemma 4 12B Q4_K_M | 0.73 | 0.73 | 1.00 |
| Qwen3.8 27B Q4_K_M | 0.93 | 0.87 | 0.87 |

- **Empero's distill is not better than its base here.** It ties Qwen3.5 9B on
  GSM8K and IFEval and is 5 points lower on MMLU, within noise. Empero's
  +20-point MMLU gain doesn't reproduce: the base already scores 0.89 in this
  setup, against 0.546 in Empero's eval.
- **All four are close.** With 50–57 questions per test, differences under ~10
  points are noise; on the 15-question subset, under ~20.
- **The 27B isn't ahead** on the shared questions, and at 7 tokens/s (half of
  it on the CPU) it isn't worth it on a 12 GB card.
- **Empty answers** count as wrong. They're most likely prompts where the
  model spent the whole 8,192-token budget thinking. Gemma had the most (12).

This run predates the memory log, so it has no memory numbers.

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
dependencies (`pip install langdetect immutabledict nltk`). The coding run also
needs `datasets` and `numpy` (the EvalPlus tests import numpy; `code_bench.py`
refuses to start without it). `llmfit` is optional,
for the speed numbers. `psutil` (`pip install psutil`) is optional, for the
system-RAM columns of the memory log.

Both run scripts sample memory every 5 s while they run
([`mem_monitor.py`](mem_monitor.py)): the models Ollama has loaded and its size
estimate for them, VRAM in use on the whole GPU (`nvidia-smi`; blank on Apple),
available system RAM, and the RAM of the Ollama processes. The samples go to
`results/mem-quality.csv` and `<results-dir>/mem.csv`. The analyze scripts
report memory *used* per model: how far VRAM rose and free RAM fell compared
with the idle samples just before the model loaded, so the desktop and other
apps don't count. On a PC that's VRAM plus system RAM; on Apple, where the GPU
shares system memory, RAM alone covers it. RAM is shown at load and at peak:
Ollama's llama.cpp runner keeps a prompt cache in system RAM (up to 8 GiB by
default) that fills as prompts go by, so the peak is mostly that cache.

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
  within noise. On the PC it's the other way round (0.70 vs 0.62), so neither
  is better.
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
python3 analyze_coding.py results/m4 results/rtx5070-q8   # machines side by side
```

For comparable numbers on another machine, start Ollama with the same settings
(`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) and pass a results
dir, e.g. on Windows with Git Bash:
`PYTHON=python GPU_BUDGET_GB=12.0 ./run_coding.sh results/rtx5070-q8 <models>`.

`code_bench.py` runs code the model wrote on your machine (in a temp directory,
with a timeout). A full run took about 8.5 hours on the M4 and about 2 hours on
the RTX 5070, most of it on the thinking models.

## RTX 5070 PC: the same coding run

The [coding benchmark](#apple-m4-coding-models-for-a-16-gb-mac) on the PC from
[Hardware](#hardware): same models, same 50 problems, same scripts. Ollama
0.40.0, llama.cpp for every model, with the Mac's settings: flash attention on
and a `q8_0` KV cache (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`).
The GPU has 12.8 GB; the idle Windows desktop takes about 1.3 GB of it.

| Model | Coding pass@1 | Median time / problem | Gen tok/s (short → 17–25K prompt) | Prompt tok/s | Wait for a 16K prompt | Usable ctx (12 GB) | Native ctx | VRAM used | RAM used (load / peak) | Thinking |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 0.80 ±0.11 | 5 s | 28 → 9 | 1,502 | 11 s | ~26K | 32K | 10.5 GB | 3.5 / 5.7 GB | no |
| `qwen2.5-coder:7b` | 0.74 ±0.12 | 3 s | 112 → 83 | 4,236 | 4 s | 32K | 32K | 5.8 GB | 1.0 / 1.3 GB | no |
| Empero 9B Q4_K_M | 0.70 ±0.13 | 6 s | 98 → 91 | 3,900 | 4 s | 256K | 256K | 6.1 GB | 2.0 / 7.1 GB | yes |
| `gemma4:12b` | 0.70 ±0.13 | 38 s | 65 → 61 | 2,603 | 6 s | not measured | 256K | 8.6 GB | 2.6 / 9.6 GB | yes |
| `qwen3.5:4b` | 0.70 ±0.13 | 35 s | 137 → 117 | 5,640 | 3 s | 256K | 256K | 4.3 GB | 1.2 / 13.2 GB | yes |
| `qwen3.5:9b` | 0.62 ±0.13 | 28 s | 88 → 84 | 3,821 | 4 s | 256K | 256K | 7.3 GB | 1.8 / 7.0 GB | yes |

Answers that hit the 8,192-token limit: 9 for `qwen3.5:9b`, 8 for
`gemma4:12b`, 6 for `qwen3.5:4b`, 1 for Empero.

- **Again, quality is a tie.** 0.62–0.80 is within the ±0.12 margin.
- **Everything that fits in VRAM is fast.** A 16K-token prompt takes seconds,
  and the non-thinking `qwen2.5-coder:7b` answers in 3 s.
- **`qwen2.5-coder:14b` scored highest but doesn't quite fit at 32K context**
  (12.8 GB, 81% on the GPU), so generation falls to 9 tokens/s with a long
  prompt. At ~26K context it would fit.
- **The 9B thinking models get the full 256K context** in 12 GB with the
  `q8_0` cache, as does the 4B.
- **VRAM used** is how far GPU memory rose above the idle desktop. **RAM used**
  is how far free system RAM fell: at load, and at peak. The peak is mostly
  the prompt cache Ollama's llama.cpp runner keeps in system RAM (up to
  8 GiB), which fills as prompts go by, so a 4B model on the GPU can still
  take 13 GB of RAM over a long session.
- **Usable ctx** uses a 12 GB budget (`GPU_BUDGET_GB=12.0`). With the
  desktop's 1.3 GB, about 11.5 GB is really free, so expect a few percent less.
  For `gemma4:12b` the loaded size didn't grow from 8K to 32K context
  (sliding-window attention), so there's no per-token cost to extrapolate from.

### `f16` vs `q8_0` KV cache

The PC first ran with Ollama's defaults (`f16` KV cache, flash attention
`auto`); those results are in [`results/rtx5070/`](results/rtx5070/). The
`q8_0` cache halves the memory per token of context and didn't measurably
change quality: scores moved by up to 0.08 in both directions, with no pattern.
It mattered most for `qwen2.5-coder:14b`:

| `qwen2.5-coder:14b` | `f16` | `q8_0` |
|---|---|---|
| Ollama size @32K | 15.8 GB | 12.8 GB |
| On the GPU | 66% | 81% |
| Gen tok/s (short → long prompt) | 20 → 6 | 28 → 9 |
| Usable ctx (12 GB) | ~15K | ~26K |

## PC vs Mac

The coding run on both machines, both with flash attention and a `q8_0` KV
cache. Each cell is **Mac / PC**.

| Model | Runner | Coding pass@1 | Median time / problem | Gen tok/s | Prompt tok/s | Wait for a 16K prompt | Ollama size @32K |
|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | llama.cpp / llama.cpp | 0.84 / 0.80 | 26 s / 5 s | 9 / 28 | 22 / 1,502 | 11.9 min / 11 s | 12.8 / 12.8 GB |
| `qwen2.5-coder:7b` | llama.cpp / llama.cpp | 0.80 / 0.74 | 10 s / 3 s | 14 / 112 | 87 / 4,236 | 3.1 min / 4 s | 5.5 / 5.7 GB |
| `gemma4:12b` | MLX / llama.cpp | 0.78 / 0.70 | 138 s / 38 s | 14 / 65 | 69 / 2,603 | 3.9 min / 6 s | 7.8 / 7.9 GB |
| `qwen3.5:9b` | MLX / llama.cpp | 0.78 / 0.62 | 71 s / 28 s | 24 / 88 | 140 / 3,821 | 1.9 min / 4 s | 7.7 / 6.2 GB |
| Empero 9B Q4_K_M | llama.cpp / llama.cpp | 0.66 / 0.70 | 39 s / 6 s | 18 / 98 | 122 / 3,900 | 2.2 min / 4 s | 5.9 / 5.9 GB |
| `qwen3.5:4b` | MLX / llama.cpp | 0.58 / 0.70 | 39 s / 35 s | 26 / 137 | 212 / 5,640 | 1.3 min / 3 s | 4.1 / 3.8 GB |

Generated by `python analyze_coding.py results/m4 results/rtx5070-q8`.

- **Scores agree within noise.** The largest gap is 0.16 (`qwen3.5:9b`); the
  margin on a difference between two 50-problem runs is about ±0.17. The
  ranking shuffles; the conclusion that the six are close doesn't.
- **The PC generates 3–8× faster and reads prompts 27–68× faster.** On the Mac
  a fresh 16K-token agent session waits minutes before the first token; on the
  PC, seconds.
- **Memory now matches** for the models both machines run on llama.cpp
  (`qwen2.5-coder`, Empero). For the MLX models the Mac size is at load: MLX
  allocates as the context fills.
- **Time per problem gains less than raw speed** for the `qwen3.5` models: on
  the PC (llama.cpp) they generated 2–4× more tokens per problem than on the
  Mac (MLX), median 2,373 vs 1,091 for the 9B and 4,493 vs 1,030 for the 4B.
  The other models generated about the same on both.
- Measured memory (the memory log) exists only for the PC; the M4 run
  predates it.

## Layout

```
modelfiles/        Ollama Modelfiles that pin context length, sampling and seed
run_eval.py        lm-eval runner for Ollama (clears stop sequences, caches requests)
run_all.sh         The full sequential run
analyze.py         Builds the result tables from results/quality-*.json
code_bench.py      HumanEval+/MBPP+ through Ollama; runs the tests, records timing
ctx_bench.py       Prompt/generation speed at depth and usable context per model
run_coding.sh      The coding run (ctx_bench + code_bench per model)
analyze_coding.py  Builds the coding table from results/m4/, or compares several results dirs
mem_monitor.py     Samples VRAM, system RAM and Ollama's memory during a run (CSV)
results/           speed-*.json from llmfit, quality-*.json from lm-eval (with per-sample logs)
results/m4/        ctx-*.json and code-*.jsonl (every answer, pass/fail, timing) from the M4 run
results/rtx5070-q8/ The same for the RTX 5070 PC (q8_0 KV cache), plus mem.csv (the memory log)
results/rtx5070/   The PC run with the default f16 KV cache
```

## Caveats

- Small samples. Treat these numbers as a sanity check of the published claims,
  not as a leaderboard.
- Single seed, Q4 quantization. Each quality number comes from one machine.
  Empero's own numbers come from an unpublished harness and likely a different
  precision.
- The 27B model doesn't fit in 12 GB of VRAM, so its speed reflects CPU offload,
  not the model itself.

## License

[MIT](LICENSE)
