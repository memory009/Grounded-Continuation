# Grounded Continuation: Runtime Verifier — Code Release

<video src="./grounded-continuation-demo.mp4" controls width="100%">
  Your browser does not support the video tag. [Download the demo video](./assets/grounded-continuation-demo.mp4).
</video>

Code accompanying the paper
**"Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations"**.

The repository contains the symbolic engine, the LLM Interpreter pipeline and
one runner per experiment in the paper, together with the recorded GPT-4o
Interpreter outputs of the end-to-end runs. Result files are not included;
every runner regenerates its outputs under `experiments/results/`.

## Layout

| Path | Contents |
|---|---|
| `symbolic_engine.py` | Reference engine: dependency map, standing labels, `Affected` and `Affected*` queries. Deterministic, no API key. |
| `pipeline.py` | LLM Interpreter pipeline and the shared LLM call. |
| `benchmark_adapter.py` | ReviseQA adapter: scenario loading, engine state under the benchmark's edits, QA prompt, closed-form scoring. |
| `experiments/scripts/` | One runner per experiment (table below). |
| `experiments/cache/` | Recorded GPT-4o Interpreter outputs, read by the end-to-end arms (table below). |

## Environment

Tested with Python 3.11, PyTorch 2.10 (CUDA 12.8) and vLLM 0.19.1 on a single
NVIDIA A100 40 GB. Any GPU that holds a 14B model in bfloat16 is sufficient.

### 1. Python packages

```bash
conda create -n gc python=3.11 -y
conda activate gc
pip install -r requirements.txt
pip install vllm==0.19.1          # only needed to serve the open-weight models
```

`requirements.txt` covers every runner. `torch` and `transformers` are used
only by the BEAM dense retriever.

### 2. API keys

```bash
export OPENAI_API_KEY=...         # GPT-4o (Interpreter, QA), GPT-4o-mini (QA)
export ANTHROPIC_API_KEY=...      # Claude Sonnet 4, classification experiment only
export VLLM_API_KEY=dummy         # any value; local vLLM endpoints do not check it
```

Which keys a run needs depends on the models chosen. The engine, the latency
benchmark and `run_recon_oracle.py` need none. With the recorded Interpreter
outputs in `experiments/cache/`, the end-to-end arms need no OpenAI key when

the QA model is served locally.

### 3. Serving the open-weight models

Open-weight models are served with vLLM behind an OpenAI-compatible endpoint,
one model per GPU:

```bash
# QA models and open Interpreters
vllm serve Qwen/Qwen2.5-7B-Instruct   --port 8000 --max-model-len 16384 --dtype bfloat16 --gpu-memory-utilization 0.90
vllm serve Qwen/Qwen2.5-14B-Instruct  --port 8002 --max-model-len 16384 --dtype bfloat16 --gpu-memory-utilization 0.90
vllm serve unsloth/gemma-3-12b-it     --port 8003 --max-model-len 16384 --dtype bfloat16 --gpu-memory-utilization 0.90
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8004 --max-model-len 16384 --dtype bfloat16 --gpu-memory-utilization 0.90
# BEAM judge
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.90
```

Runners take the endpoint as `--base-url http://localhost:<port>/v1/chat/completions`
together with `--model <name>` (`--qa-base-url` / `--qa-model` and
`--judge-base-url` / `--judge-model` for BEAM). For OpenAI models pass
`--base-url https://api.openai.com/v1/chat/completions --api-key-env OPENAI_API_KEY`.
The BEAM retriever `BAAI/bge-small-en-v1.5` is downloaded from Hugging Face on
first use and runs on CPU.

### 4. Data

Third-party benchmarks are not included. Place them under `data/` as released
by their authors:

```
data/reviseqa/reviseqa_data/nl/verified/                          # ReviseQA, 930 verified scenarios (github.com/ChadiHelwe/reviseqa)
data/memoryagentbench/Conflict_Resolution-00000-of-00001.parquet  # MemoryAgentBench FactConsolidation (Hugging Face: ai-hyz/MemoryAgentBench)
data/beam/100K-00000-of-00001.parquet                             # BEAM 100K split (Hugging Face: Mohammadta/BEAM)
data/recon/{medical,finance}/case_*/                              # RECON case skeletons and questions
```

The data paths can also be set through `RQ_DATA_DIR`, `CR_DATA` (or `--data`),
`BEAM_DATA`, and `--data` for the RECON runners.

## Quick start (no GPU, no API key)

```bash
python symbolic_engine.py                                   # Phase 2 dependency map, retraction queries, Affected* answers
python experiments/scripts/run_retraction_latency.py        # latency scaling
python experiments/scripts/run_recon_oracle.py              # RECON: one-step Affected vs Affected* (needs data/recon)
```

## Recorded Interpreter outputs

The end-to-end arms read decisions that the GPT-4o Interpreter recorded, so
they run with the QA model only. Deleting a file re-runs the Interpreter with
the runner in the last column, at the cost of the GPT-4o calls.

| File under `experiments/cache/` | Read by | Produced by |
|---|---|---|
| `reviseqa_interp/interp_shard{0..3}.json` (per-edit ops) | `run_reviseqa_multimodel.py --arms e2e_replay` | `run_reviseqa_interp_full.py --shard-index {0..3}` |
| `memagentbench_cr/e2e_ingest_{6k,32k}.json` (supersession decisions) | `run_memagentbench_cr_e2e.py --phase qa`, `run_cr_e2e_dep.py` | `run_memagentbench_cr_e2e.py --phase ingest` |
| `beam_ingest/gpt-4o_final_ua/conv_{1..20}.json` (extracted statements and decisions) | `run_beam.py qa --ingest-tag gpt-4o_final_ua` | `run_beam.py ingest --ingest-tag gpt-4o_final_ua` |

## Running the experiments

All runners are launched from the repository root and write under `experiments/results/`.

### ReviseQA

```bash
python experiments/scripts/run_reviseqa_multimodel.py --tag qwen2.5-7b \
    --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1/chat/completions \
    --arms two_arm transcript_rag e2e_replay
```

`two_arm` produces LLM-only and verifier (native), `transcript_rag` produces RAG
(TF-IDF, top-22), and `e2e_replay` produces verifier (end-to-end) from the
recorded Interpreter ops.

### MemoryAgentBench FactConsolidation

The engine ingests the stream one observation per fact and retracts a record
when a later fact carries the same key. The verifier arm selects the active
records linked to the entities in the question and registers them as the
question's dependency set. `--check-only` verifies the engine state and
selection without calling a model.

```bash
V="sh_6k mh_6k sh_32k mh_32k sh_64k mh_64k sh_262k mh_262k"
M="--model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1/chat/completions --tag qwen2.5-7b"
python experiments/scripts/run_memagentbench_cr.py $M --variants $V         # LLM-only, RAG, verifier + TF-IDF
python experiments/scripts/run_cr_engine_native.py $M --variants $V         # verifier (native)
# end-to-end, over the recorded GPT-4o ingestion of the 6K and 32K streams
python experiments/scripts/run_memagentbench_cr_e2e.py --phase qa --variants sh_6k mh_6k sh_32k mh_32k \
    --qa-model Qwen/Qwen2.5-7B-Instruct --qa-base-url http://localhost:8000/v1/chat/completions --tag qwen2.5-7b
python experiments/scripts/run_cr_e2e_dep.py $M --variants sh_6k mh_6k sh_32k mh_32k
```

For GPT-4o / GPT-4o-mini pass `--model gpt-4o --base-url https://api.openai.com/v1/chat/completions --api-key-env OPENAI_API_KEY`.

### BEAM

`run_beam.py` has four phases (`ingest`, `qa`, `judge`, `report`), all resumable.
The dense retriever (bge-small-en-v1.5) needs `torch` and `transformers`.

```bash
C="--convs $(seq -s ' ' 1 20)"
# 1) GPT-4o Interpreter ingestion; recorded in experiments/cache/beam_ingest/gpt-4o_final_ua, so this step can be skipped
python experiments/scripts/run_beam.py ingest $C --ingest-tag gpt-4o_final_ua \
    --max-facts 20 --value-change-mode annotate --decision-batching none --speaker-rule user-authority

# 2) QA, per model (example: Qwen2.5-7B served on :8000)
QA="--qa-model Qwen/Qwen2.5-7B-Instruct --qa-base-url http://localhost:8000/v1/chat/completions"
COMMON="$C --ingest-tag gpt-4o_final_ua --top-k 100 --retriever dense --annotate changed"
python experiments/scripts/run_beam.py qa $COMMON $QA --tag qwen2.5-7b \
    --arms full --ctx-char-budget 48000           # LLM-only; 440000 for GPT-4o(-mini)
python experiments/scripts/run_beam.py qa $COMMON $QA --tag qwen2.5-7b \
    --arms tr verifier_src                        # RAG and verifier

# 3) judge (Qwen2.5-32B-Instruct-AWQ on :8001) and per-ability report
python experiments/scripts/run_beam.py judge --tag qwen2.5-7b \
    --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ --judge-base-url http://localhost:8001/v1/chat/completions
python experiments/scripts/run_beam.py report --tag qwen2.5-7b
```

For GPT-4o / GPT-4o-mini add `--api-key-env OPENAI_API_KEY` and point
`--qa-base-url` at `https://api.openai.com/v1/chat/completions`. The RAG and
verifier arms are repeated three times under different `--tag` values.

### Interpreter classification

```bash
python experiments/scripts/run_experiments.py --runs 5 --save cls_claude.json          # Claude Sonnet 4
python experiments/scripts/run_experiments.py --backend openai --model gpt-4o \
    --base-url https://api.openai.com/v1/chat/completions --save cls_gpt4o.json
python experiments/scripts/run_experiments.py --backend openai --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1/chat/completions --save cls_qwen7b.json
```

The Phase 2 and Phase 3 scenarios, ground truth and the three prompt conditions
are embedded in the script.

### RECON

```bash
python experiments/scripts/run_recon_oracle.py                   # one-step Affected vs Affected*
python experiments/scripts/run_recon_llm_oracle.py --mode chain  # GPT-4o, isolated chain
python experiments/scripts/run_recon_llm_oracle.py --mode full   # GPT-4o, full skeleton
```
