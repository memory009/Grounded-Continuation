# Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations

Qisong He, Jinwei Hu, Xinmiao Huang, Changshun Wu, Yi Dong and Xiaowei Huang
School of Computer Science & Informatics, University of Liverpool

Paper: https://arxiv.org/abs/2605.14175

A conversation establishes things as it goes, and later turns can retract
them. An LLM that answers from the raw transcript, or from a retrieval window
over it, keeps using premises that the conversation has already given up. This
repository contains the runtime verifier from the paper: an LLM Interpreter
maps each utterance to one of eight epistemic operations, and a symbolic
engine maintains a dependency map that records what every commitment rests on.
At each turn the engine checks whether the next output still traces back to
commitments in good standing, and answers retraction queries in time linear in
the map's size.

**What you get**

- **Accuracy where premises get superseded.** On ReviseQA and the
  fact-consolidation split of MemoryAgentBench, the verifier leads a
  budget-matched transcript-retrieval baseline across five QA models and lifts
  MemoryAgentBench single-hop accuracy from 0.46–0.95 to 0.93–0.98. With the
  verifier, a 7B QA model overtakes unaided GPT-4o.
- **Per-query cost independent of conversation length.** Prompts stay near
  0.8k tokens where full context reaches 114k, and a retraction query runs in
  under a microsecond at 2000 turns.
- **Guarantees on the dependency map.** The active set is conflict-free at
  every turn, retracting a premise removes exactly its dependents, and both the
  grounding check and the retraction query are linear time.

## Layout

| Path | Contents |
|---|---|
| `symbolic_engine.py` | Reference engine: dependency map, standing labels, `Verify` walk, `Affected*` retraction query. No API key needed. |
| `pipeline.py` | LLM Interpreter: classifies each utterance into an operation and applies it to the engine. |
| `benchmark_adapter.py` | Loaders, prompt construction and scoring for ReviseQA, MemoryAgentBench and RECON. |
| `verify_experiment.py` | Direct verifier test harness over the 50-item test set in `experiments/e2_verify/`. |
| `experiments/scripts/`, `experiments/e5_robustness/` | One runner per experiment in the paper. |
| `experiments/prompts/` | The three classification prompt conditions (Minimal, Definitions, State-augmented). |
| `symbolic_engine.jsx` | Browser demo of the engine with step-through and a counterfactual panel. |

Runners are launched from the repository root and write under
`experiments/results/` (git-ignored).

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Closed models are called through `OPENAI_API_KEY` (GPT-4o, GPT-4o-mini) and
`ANTHROPIC_API_KEY` (Claude, classification experiment only). Open-weight QA
models (Qwen2.5-7B/14B-Instruct, Gemma-3-12B-it, Llama-3.1-8B-Instruct) are
served with vLLM behind an OpenAI-compatible endpoint. One 40 GB GPU is enough
for every model used.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
# runners then take --base-url http://localhost:8000/v1/chat/completions
```

### Datasets

The benchmarks are not redistributed here. Download them from their authors
and place them under `datasets_cache/`:

| Benchmark | Source | Location |
|---|---|---|
| ReviseQA | https://github.com/ChadiHelwe/reviseqa | `datasets_cache/reviseqa/` (so that `datasets_cache/reviseqa/reviseqa_data/nl/verified/ex_*.json` exist) |
| MemoryAgentBench | https://huggingface.co/datasets/ai-hyz/MemoryAgentBench, file `data/Conflict_Resolution-00000-of-00001.parquet` | `datasets_cache/memoryagentbench/data/` |
| RECON | https://anonymous.4open.science/r/RECON-Bench (link given in the RECON paper) | `datasets_cache/recon/<domain>/case_*/skeleton.json` and `questions.json` |

```bash
git clone https://github.com/ChadiHelwe/reviseqa datasets_cache/reviseqa
huggingface-cli download ai-hyz/MemoryAgentBench --repo-type dataset \
    --include "data/Conflict_Resolution-*.parquet" --local-dir datasets_cache/memoryagentbench
```

## Quick start (no GPU, no API key)

```bash
python symbolic_engine.py                                              # Phase 2 scenario: dependency map, retraction queries, Affected* answers
python experiments/scripts/e4_retraction_latency.py --output e4.json   # retraction latency from 13 to 2000 turns
```

## Experiments

Each runner produces every arm of the corresponding table from the same QA
model, the same prompt shape and temperature 0; the `--arms` flag selects
which. Baselines and the verifier therefore come from one command.

**ReviseQA.** `two_arm` produces the LLM-only baseline and the verifier with
benchmark-native updates, `transcript_rag` the budget-matched TF-IDF baseline.
Add `--max-scenarios 10` for a smoke run.

```bash
python experiments/scripts/run_reviseqa_multimodel.py \
    --tag qwen2.5-7b --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1/chat/completions \
    --arms two_arm transcript_rag
```

For the end-to-end arm a GPT-4o Interpreter extracts every update from raw
text. The Interpreter pass is sharded and cached, then replayed for any QA
model at no further Interpreter cost:

```bash
for i in 0 1 2 3; do
  python experiments/scripts/run_reviseqa_interp_full.py --num-shards 4 --shard-index $i
done
python experiments/scripts/run_reviseqa_multimodel.py --tag qwen2.5-7b \
    --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1/chat/completions \
    --arms e2e_replay
```

**MemoryAgentBench fact consolidation.** By default one run produces the
long-context baseline (`llm_full`, raw fact stream tail-truncated to the QA
budget), the TF-IDF top-50 baseline over the full stream (`tr`) and the
verifier (`verifier`) for all eight variants. Add `--variants sh_6k
--max-questions 10` for a smoke run.

```bash
python experiments/scripts/run_memagentbench_cr.py \
    --tag qwen2.5-7b --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1/chat/completions
```

For the end-to-end arm the Interpreter decides, fact by fact, which active
fact the new one supersedes; ingestion is cached and reused by the QA phase:

```bash
python experiments/scripts/run_memagentbench_cr_e2e.py --phase ingest --lengths 6k 32k
python experiments/scripts/run_memagentbench_cr_e2e.py --phase qa \
    --variants sh_6k mh_6k sh_32k mh_32k \
    --qa-model Qwen/Qwen2.5-7B-Instruct --qa-base-url http://localhost:8000/v1/chat/completions
```

For a closed QA model pass `--model gpt-4o --base-url
https://api.openai.com/v1/chat/completions --api-key-env OPENAI_API_KEY`. The
end-to-end arms need `OPENAI_API_KEY` for the Interpreter.

The remaining experiments in the paper (Interpreter classification, the direct
verifier test, RECON soundness, dependency-extraction ablations, the noise
diagnostic, cost accounting) each have their own runner under
`experiments/`; see the usage block at the top of each file.

## Using the verifier on your own conversation

Run the Interpreter and the engine end-to-end over a transcript given as a
JSON list of `{"speaker": ..., "text": ...}` objects, then query which
commitments a retraction would take down:

```bash
python pipeline.py --conversation my_conversation.json --all-queries \
    --backend openai --model gpt-4o --base-url https://api.openai.com/v1/chat/completions
```

The engine can also be driven directly, without any LLM:

```python
from symbolic_engine import EpistemicEngine

e = EpistemicEngine()
e.observe("o1", "Payment errors started at 02:15", turn="T1", speaker="Carol")
e.hypothesize("h1", "The Redis connection pool is exhausted", turn="T2", speaker="Bob", explains=["o1"])
e.hypothesize("h2", "Auth failures are caused by the Redis outage", turn="T3", speaker="Alice", depends_on=["h1"])

e.get_affected_closure("h1")            # ['h2']: what would lose support if h1 fell
e.undermine("h1", "Redis metrics are healthy", turn="T4", speaker="Carol")
e.retract_assumption("h1")              # h2 is flagged as depending on a retracted premise
print(e.get_state_summary())
```

## License

The code is released under the MIT License (full text in `LICENSE`). You may
use, copy, modify and redistribute it, including in commercial products, as
long as the copyright and permission notice stays with the code. It comes
without warranty. The authored 50-item test set in `experiments/e2_verify/`
is released under CC BY 4.0: reuse it freely with attribution to the paper.
