#!/usr/bin/env python3
"""BEAM (100K split) three-arm runner: full context vs transcript-RAG vs verifier.

Data: Mohammadta/BEAM on Hugging Face (ICLR 2026, CC BY-SA 4.0), split 100K =
20 synthetic user-assistant conversations (188-392 messages, ~100-225K tokens),
20 probing questions each: two per ability for ten abilities. Eight abilities
are general long-conversation memory (information extraction, multi-session
reasoning, temporal reasoning, event ordering, summarization, abstention,
instruction following, preference following); two are supersession
(knowledge_update, contradiction_resolution). Each question carries a rubric
of atomic criteria; BEAM's official judge scores every criterion 0 / 0.5 / 1
with one unified prompt and averages.

Phases (all resumable, results under experiments/results/beam/):
  ingest   GPT-4o Interpreter, end-to-end: extract atomic facts from every
           message (user and assistant), then stream the facts through the
           engine's standing rule with an LLM supersession decision over the
           TF-IDF top-5 active candidates (same design as the MemAB-FC
           end-to-end arm). Superseded facts keep their record; only their
           standing changes.
  qa       answer every probing question with one QA model under three arms
           and the SAME prompt (BEAM's official RAG answer prompt):
             full      whole transcript, tail-truncated to a character budget
             tr        transcript-RAG: TF-IDF over user-assistant turn pairs,
                       chunks added in similarity order until the context is
                       budget-matched to the verifier's context for that question
             verifier  TF-IDF top-k over ACTIVE facts; a fact that replaced an
                       earlier one is rendered with the replaced value marked
  judge    BEAM's unified judge prompt, one call per rubric criterion
  report   per-ability table per arm, context sizes, knowledge-update capture

Usage:
  python experiments/scripts/run_beam.py ingest --convs 1 5
  python experiments/scripts/run_beam.py qa --convs 1 5 --tag qwen2.5-7b \\
      --qa-model Qwen/Qwen2.5-7B-Instruct --qa-base-url http://localhost:8000/v1/chat/completions \\
      --ctx-char-budget 100000
  python experiments/scripts/run_beam.py qa --convs 1 5 --tag gpt-4o --qa-model gpt-4o \\
      --qa-base-url https://api.openai.com/v1/chat/completions --api-key-env OPENAI_API_KEY
  python experiments/scripts/run_beam.py judge --tag qwen2.5-7b \\
      --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ --judge-base-url http://localhost:8001/v1/chat/completions
  python experiments/scripts/run_beam.py report --tag qwen2.5-7b
"""
import argparse, ast, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
PROJECT_DIR = os.environ.get("BEAM_PROJECT_DIR", ".")
DATA = os.environ.get("BEAM_DATA", os.path.join(PROJECT_DIR, "data", "beam",
                                               "100K-00000-of-00001.parquet"))
OUT_DIR = os.path.join(PROJECT_DIR, "experiments/results/beam")
# Interpreter extraction and supersession decisions (shipped for gpt-4o_final_ua)
INGEST_DIR = os.path.join(PROJECT_DIR, "experiments/cache/beam_ingest")

CATEGORIES = ["information_extraction", "multi_session_reasoning", "temporal_reasoning",
              "event_ordering", "summarization", "abstention", "instruction_following",
              "preference_following", "knowledge_update", "contradiction_resolution"]
SUPERSESSION_CATS = {"knowledge_update", "contradiction_resolution"}

# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
EXTRACT_SYSTEM = (
    "You extract atomic facts from one message of a long conversation between a "
    "user and an AI assistant. Keep facts that could be asked about later: the "
    "user's situation, numbers, dates, deadlines, counts, prices, decisions, plans, "
    "preferences, standing instructions to the assistant, people and relationships, "
    "and the concrete content of the assistant's advice or plans (key "
    "recommendations, schedules, values). Each fact is one self-contained sentence "
    "with the subject named explicitly ('The user ...', 'The assistant ...'); "
    "resolve pronouns; keep numbers, dates and names verbatim. Skip greetings, "
    "filler and generic explanations that carry no specific information about this "
    "user or project. Return at most {maxf} facts. Reply ONLY with JSON: "
    '{{"facts": ["...", "..."]}}.'
)

SUPERSEDE_SYSTEM = (
    "You maintain a store of facts about a user and their conversation with an "
    "assistant. A NEW fact supersedes an OLD fact if both state a value for the "
    "SAME attribute of the SAME subject (the same deadline, count, price, duration, "
    "choice, status or plan detail), so the new value replaces the old one. A fact "
    "that adds detail on a different attribute, or restates the same value, does "
    "not supersede. Given the NEW fact and numbered CANDIDATE facts from the store, "
    "decide which candidate (if any) states an earlier value for the same attribute, "
    "and whether the NEW fact gives a DIFFERENT value from it (a restatement of the "
    "same value, in any wording, is not a change). Reply ONLY with JSON: "
    '{"supersedes": <candidate number or null>, "value_changed": <true or false>}.'
)

# BEAM's official answer prompt (src/prompts.py, answer_generation_for_rag), used
# verbatim for all three arms so the arms differ only in the CONTEXT.
ANSWER_PROMPT = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below.

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation

RESPONSE:
"""

# BEAM's official judge prompt (src/prompts.py, unified_llm_judge_base_prompt).
JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_ULOCK = threading.Lock()
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "failed": 0}


def _record_usage(u):
    with _ULOCK:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += (u or {}).get("prompt_tokens", 0) or 0
        USAGE["completion_tokens"] += (u or {}).get("completion_tokens", 0) or 0
        USAGE["cached_tokens"] += ((u or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0


def chat(messages, url, key, model, max_tokens=400, json_mode=False, timeout=300):
    """OpenAI-compatible chat call with retry; returns content or None."""
    import requests
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    backoff = 2
    quota_waits = 0
    attempt = 0
    while attempt < 6:
        attempt += 1
        try:
            r = requests.post(url, timeout=timeout,
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {key}"}, json=body)
            d = r.json()
            if r.status_code == 429 and "insufficient_quota" not in str(d):
                time.sleep(backoff); backoff = min(backoff * 2, 32); continue
            if "error" in d:
                msg = str(d["error"])[:200]
                print(f"  [chat] API error (attempt {attempt}): {msg}", flush=True)
                if "insufficient_quota" in msg or "credit_balance_exhausted" in msg:
                    # credit top-ups propagate unevenly for a while: wait, do not burn retries
                    quota_waits += 1
                    if quota_waits > 15:
                        raise SystemExit("API quota exhausted for >15 min: " + msg)
                    print(f"  [chat] quota error, waiting 60s ({quota_waits}/15)", flush=True)
                    attempt -= 1
                    time.sleep(60); continue
                if json_mode and "response_format" in msg:
                    body.pop("response_format", None)
            if "choices" in d and d["choices"]:
                _record_usage(d.get("usage"))
                return d["choices"][0]["message"].get("content")
        except Exception as e:
            print(f"  [chat] error attempt {attempt}: {e}", flush=True)
        time.sleep(backoff); backoff = min(backoff * 2, 32)
    with _ULOCK:
        USAGE["failed"] += 1
    return None


def parse_json(text):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def tfidf_rank(texts, query):
    """Indices of texts sorted by TF-IDF cosine to query (desc, stable)."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(lowercase=True)
        mat = vec.fit_transform(texts + [query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    except ValueError:
        sims = np.zeros(len(texts))
    return sorted(range(len(texts)), key=lambda i: (-float(sims[i]), i))


def est_tokens(s):
    return len(s) // 4


import hashlib


class EmbCache:
    """Text -> bge-small vector, persisted per conversation so every QA run reuses it."""

    def __init__(self, path=None):
        self.path, self.d, self.dirty = path, {}, False
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.d = {k: z[k] for k in z.files}

    @staticmethod
    def key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text):
        return self.d.get(self.key(text))

    def put(self, text, vec):
        self.d[self.key(text)] = vec; self.dirty = True

    def save(self):
        """Several QA runs may embed the same conversation concurrently: write to a
        per-process temp file and never fail the run over a cache write."""
        if self.path and self.dirty:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, **self.d); os.replace(tmp, self.path)
            except OSError as e:
                print(f"  [emb-cache] save skipped: {e}", flush=True)
            self.dirty = False


class CandidateIndex:
    """Candidates for the supersession decision: the union of the top-n active
    facts under three views of the same text, dense bge-small-en-v1.5 (CLS,
    normalised), word TF-IDF (1-2 grams, sublinear, English stop words) and
    plain unigram TF-IDF. No single lexical or dense ranker put the true
    predecessor in its top-5 on natural conversation, the union of top-8s did."""

    def __init__(self, per_view=8, cap=20, cache=None):
        self.per_view, self.cap = per_view, cap
        self._tok = self._mdl = None
        self.emb = {}          # fid -> np.array
        self.cache = cache or EmbCache()

    _load_lock = threading.Lock()

    def _load(self):
        # transformers' lazy imports are not thread-safe: load under a lock, and callers
        # load eagerly from the main thread before any worker pool starts
        with CandidateIndex._load_lock:
            if self._mdl is None:
                import torch
                from transformers import AutoTokenizer, AutoModel
                self._torch = torch
                torch.set_num_threads(max(1, int(os.environ.get("BEAM_TORCH_THREADS", "8"))))
                self._tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
                self._mdl = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    def embed(self, fid, text):
        if fid not in self.emb:
            v = self.cache.get(text)
            if v is None:
                self._load()
                with self._torch.no_grad():
                    t = self._tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
                    h = self._mdl(**t).last_hidden_state[:, 0]
                    v = self._torch.nn.functional.normalize(h, dim=-1)[0].numpy()
                self.cache.put(text, v)
            self.emb[fid] = v
        return self.emb[fid]

    def candidates(self, store, cand_fids, new_fid, new_text):
        if not cand_fids:
            return []
        texts = [store[a]["text"] for a in cand_fids]
        picked = []
        # dense view
        q = self.embed(new_fid, new_text)
        M = np.stack([self.embed(a, store[a]["text"]) for a in cand_fids])
        sims = M @ q
        for i in np.argsort(-sims)[: self.per_view]:
            picked.append(cand_fids[int(i)])
        # lexical views
        for vec in (TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
                    TfidfVectorizer(lowercase=True)):
            try:
                mat = vec.fit_transform(texts + [new_text])
                sc = cosine_similarity(mat[-1], mat[:-1]).ravel()
            except ValueError:
                continue
            for i in sorted(range(len(texts)), key=lambda i: (-float(sc[i]), i))[: self.per_view]:
                if cand_fids[i] not in picked:
                    picked.append(cand_fids[i])
        return sorted(picked)[: self.cap] if len(picked) <= self.cap else sorted(picked[: self.cap])


def save_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def merge_save_results(path, results):
    """Write a QA/judge results file without clobbering fields another process added
    since we loaded it: per (question, arm), union the on-disk fields with ours."""
    try:
        disk = json.load(open(path)) if os.path.exists(path) else {}
    except Exception:
        disk = {}
    merged = dict(disk)
    for k, rec in results.items():
        if k.startswith("_"):
            merged[k] = rec; continue
        d = merged.get(k)
        if not d:
            merged[k] = rec; continue
        arms = dict(d.get("arms", {}))
        for a, x in rec.get("arms", {}).items():
            arms[a] = {**arms.get(a, {}), **x}
        merged[k] = {**d, **{kk: vv for kk, vv in rec.items() if kk != "arms"}, "arms": arms}
    save_json(path, merged)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_beam(conv_ids=None):
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA).to_pylist()
    convs = []
    for r in rows:
        cid = int(r["conversation_id"])
        if conv_ids and cid not in conv_ids:
            continue
        msgs = []
        for session in r["chat"]:
            for m in session:
                msgs.append({"id": int(m["id"]), "role": m["role"],
                             "content": m["content"] or "",
                             "time_anchor": (m.get("time_anchor") if m.get("time_anchor")
                                             not in (None, "None") else None)})
        msgs.sort(key=lambda m: m["id"])
        pq_raw = r["probing_questions"]
        try:
            qs = json.loads(pq_raw)
        except Exception:
            qs = ast.literal_eval(pq_raw)
        convs.append({"conv_id": cid, "category": r["conversation_seed"]["category"],
                      "title": r["conversation_seed"]["title"], "messages": msgs,
                      "questions": qs})
    return convs


def gold_of(item):
    for k in ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance"):
        if item.get(k):
            return item[k]
    return ""


def _flat_ints(v):
    """BEAM's source_chat_ids mix ints, strings and nested lists; keep the ints."""
    out = []
    for x in (v if isinstance(v, list) else [v]):
        if isinstance(x, list):
            out.extend(_flat_ints(x))
        else:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
    return out


def source_ids(item):
    sc = item.get("source_chat_ids")
    out = {}
    if isinstance(sc, dict):
        for k, v in sc.items():
            out[k] = _flat_ints(v)
    elif isinstance(sc, list):
        out["all"] = _flat_ints(sc)
    return out


# --------------------------------------------------------------------------
# phase: ingest
# --------------------------------------------------------------------------
INGEST_TAG = "gpt-4o"


def ingest_path(cid):
    return os.path.join(INGEST_DIR, INGEST_TAG, f"conv_{cid}.json")


def extract_facts(conv, args):
    """Parallel per-message extraction; returns {msg_id: [facts]}."""
    path = ingest_path(conv["conv_id"])
    state = json.load(open(path)) if os.path.exists(path) else {}
    extracted = {int(k): v for k, v in state.get("extracted", {}).items()}
    todo = [m for m in conv["messages"] if m["id"] not in extracted]
    print(f"[conv {conv['conv_id']}] extraction: {len(extracted)} cached, {len(todo)} to do", flush=True)
    lock = threading.Lock()

    def work(m):
        content = m["content"]
        if len(content) > args.max_msg_chars:
            content = content[:args.max_msg_chars] + " [...]"
        head = f"MESSAGE {m['id']} ({m['role']}" + (f", dated {m['time_anchor']}" if m["time_anchor"] else "") + "):\n"
        out = chat([{"role": "system", "content": EXTRACT_SYSTEM.format(maxf=args.max_facts)},
                    {"role": "user", "content": head + content}],
                   args.interp_url, args.interp_key, args.interp_model,
                   max_tokens=900, json_mode=True)
        if out is None:
            print(f"  [conv {conv['conv_id']}] msg {m['id']}: extraction call failed, not cached", flush=True)
            return m["id"]
        d = parse_json(out) or {}
        facts = d.get("facts") if isinstance(d, dict) else None
        if not isinstance(facts, list):
            facts = []
        facts = [str(f).strip() for f in facts if str(f).strip()][:args.max_facts]
        with lock:
            extracted[m["id"]] = facts
            if len(extracted) % 20 == 0:
                state["extracted"] = {str(k): v for k, v in extracted.items()}
                save_json(path, state)
        return m["id"]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, m) for m in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 25 == 0:
                print(f"  [conv {conv['conv_id']}] extracted {i}/{len(todo)} messages", flush=True)
    state["extracted"] = {str(k): v for k, v in extracted.items()}
    save_json(path, state)
    missing = [m["id"] for m in conv["messages"] if m["id"] not in extracted]
    if missing:
        raise SystemExit(f"[conv {conv['conv_id']}] {len(missing)} messages not extracted "
                         f"(API failures); rerun to resume")
    return extracted, state


def build_store(conv, extracted, state, args):
    """Standing pass. Facts are applied message by message: the facts of one message
    are decided concurrently against the store as it stood BEFORE the message (they
    never supersede each other), then applied in order; if two of them name the same
    predecessor, the first wins. Decisions are cached per fact and resumable."""
    path = ingest_path(conv["conv_id"])
    decisions = {int(k): (v if isinstance(v, dict) else {"sup": v, "cands": None})
                 for k, v in state.get("decisions", {}).items()}
    store = []            # list of fact dicts, fid = index
    active = []           # fids
    emb_cache = EmbCache(os.path.join(OUT_DIR, "emb_cache", f"conv_{conv['conv_id']}.npz"))
    index = CandidateIndex(per_view=args.cand_k, cap=args.cand_cap, cache=emb_cache)
    n_calls = 0
    lock = threading.Lock()

    def decide(fid, text, top):
        """One supersession call; returns (sup or None, value_changed)."""
        cand_txt = "\n".join(f"{a}. {store[a]['text']}" for a in top)
        out = chat([{"role": "system", "content": SUPERSEDE_SYSTEM},
                    {"role": "user", "content": f"CANDIDATES:\n{cand_txt}\n\nNEW FACT:\n{text}"}],
                   args.interp_url, args.interp_key, args.interp_model,
                   max_tokens=40, json_mode=True)
        d = parse_json(out) or {}
        sup = d.get("supersedes") if isinstance(d, dict) else None
        try:
            sup = int(sup) if sup is not None else None
        except Exception:
            sup = None
        if sup is not None and sup not in top:
            sup = None
        vc = bool(d.get("value_changed", True)) if isinstance(d, dict) else True
        if args.value_change_mode == "require" and not vc:
            sup = None
        return sup, vc

    VALUE_RE = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
    # user-grounded: values stated in user turns strictly between two statements' messages
    user_vals = [(m["id"], value_tokens(m["content"])) for m in conv["messages"] if m["role"] == "user"]

    def grounded(newv, lo, hi):
        return any(v in vals for mid, vals in user_vals if lo < mid < hi for v in newv)

    confirmed = set()
    if args.standing_variant == "user-confirm":
        # replay only: confirmation looks ahead, so every decision must already be cached
        roles = [m["role"] for m in conv["messages"] for _ in extracted.get(m["id"], [])]
        if len(decisions) < len(roles):
            raise SystemExit(f"[conv {conv['conv_id']}] user-confirm needs all {len(roles)} decisions cached, "
                             f"found {len(decisions)}")
        kids = {}
        for f, d in decisions.items():
            if d.get("sup") is not None and d.get("value_changed") is False:
                kids.setdefault(d["sup"], []).append(f)
        # a statement is confirmed when a later user statement restates its value, directly or
        # through a chain of same-value restatements
        for f in sorted(kids, reverse=True):
            if any(roles[g] == "user" or g in confirmed for g in kids[f]):
                confirmed.add(f)

    def apply(fid):
        sup = decisions[fid]["sup"]
        if (sup is not None and args.supersession_filter == "value-slot"
                and VALUE_RE.search(store[sup]["text"]) and not VALUE_RE.search(store[fid]["text"])):
            # a statement that carries no value cannot replace one that does (a restatement
            # that drops the number is not an update)
            store[fid]["blocked_supersession"] = sup
            sup = None
        if (sup is not None and args.speaker_rule == "user-authority"
                and store[sup]["role"] == "user" and store[fid]["role"] == "assistant"):
            # the assistant restating or contradicting a user commitment does not retract it:
            # commitments belong to the speaker who made them
            store[fid]["blocked_supersession"] = sup
            sup = None
        vc = decisions[fid].get("value_changed")
        if sup is not None and args.standing_variant == "restatement-stands" and vc is False:
            # restating the same value is not a revision: the earlier statement stands
            store[fid]["restates"] = sup
            sup = None
        if (sup is not None and args.standing_variant == "user-confirm"
                and store[fid]["role"] == "assistant" and vc is not False and fid not in confirmed):
            # an assistant revision takes effect only once a later user statement restates it
            store[fid]["unconfirmed_supersession"] = sup
            sup = None
        d_sup = decisions[fid]["sup"]
        if (args.standing_variant == "user-grounded" and store[fid]["role"] == "assistant"
                and d_sup is not None and d_sup < len(store)):
            base = store[d_sup]
            if base["status"] == "unconfirmed" and vc is False:
                # restating an unadmitted value carries no more weight than the original
                store[fid]["status"] = "unconfirmed"; store[fid]["restates_unconfirmed"] = d_sup
                return
            if vc is not False and not ADVICE_RE.search(base["text"]):
                newv = value_tokens(store[fid]["text"]) - value_tokens(base["text"])
                if newv and not grounded(newv, base["msg_id"], store[fid]["msg_id"]):
                    # the assistant changes a recorded value that no user turn in between states:
                    # drift rather than an update. The earlier statement stands, this one is not admitted.
                    store[fid]["status"] = "unconfirmed"; store[fid]["ungrounded_supersession"] = d_sup
                    return
        if sup is not None and sup < len(store) and store[sup]["status"] == "active":
            store[sup]["status"] = "superseded"
            store[sup]["superseded_by"] = fid
            store[fid]["supersedes"] = sup
            store[fid]["value_changed"] = decisions[fid].get("value_changed")
            active.remove(sup)
        active.append(fid)

    for m in conv["messages"]:
        texts = extracted.get(m["id"], [])
        if not texts:
            continue
        base_fid = len(store)
        for text in texts:
            store.append({"fid": len(store), "msg_id": m["id"], "role": m["role"],
                          "time_anchor": m["time_anchor"], "text": text,
                          "status": "active", "superseded_by": None, "supersedes": None})
        if args.decision_batching == "none":
            # original sequential rule: each fact sees the live active set (minus its own message)
            for i, text in enumerate(texts):
                fid = base_fid + i
                if fid not in decisions:
                    cands = [a for a in active if store[a]["msg_id"] != m["id"]]
                    top = index.candidates(store, cands, fid, text) if cands else []
                    if top:
                        sup, vc = decide(fid, text, top); n_calls += 1
                    else:
                        sup, vc = None, None
                    decisions[fid] = {"sup": sup, "cands": top, "value_changed": vc}
                    if n_calls % 25 == 0:
                        state["decisions"] = {str(k): v for k, v in decisions.items()}
                        save_json(path, state)
                apply(fid)
            continue
        cand_fids = list(active)                       # store as it stood before this message
        todo = []
        for i, text in enumerate(texts):
            fid = base_fid + i
            if fid in decisions:
                continue
            top = index.candidates(store, cand_fids, fid, text) if cand_fids else []
            if not top:
                decisions[fid] = {"sup": None, "cands": top, "value_changed": None}
                continue
            todo.append((fid, text, top))
        if todo:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(todo))) as ex:
                futs = {ex.submit(decide, fid, text, top): (fid, top) for fid, text, top in todo}
                for f in as_completed(futs):
                    fid, top = futs[f]
                    sup, vc = f.result()
                    decisions[fid] = {"sup": sup, "cands": top, "value_changed": vc}
            n_calls += len(todo)
        # apply this message's decisions in fact order
        for i in range(len(texts)):
            apply(base_fid + i)
        if n_calls and n_calls % 100 < len(todo):
            state["decisions"] = {str(k): v for k, v in decisions.items()}
            save_json(path, state)
    emb_cache.save()
    state["decisions"] = {str(k): v for k, v in decisions.items()}
    state["store"] = store
    state["summary"] = {"n_messages": len(conv["messages"]), "n_facts": len(store),
                        "n_active": len(active),
                        "n_superseded": sum(1 for r in store if r["status"] == "superseded"),
                        "n_unconfirmed": sum(1 for r in store if r["status"] == "unconfirmed"),
                        "decision_calls_this_run": n_calls, "usage": dict(USAGE),
                        "interp_model": args.interp_model,
                        "value_change_mode": args.value_change_mode,
                        "max_facts": args.max_facts, "decision_batching": args.decision_batching,
                        "speaker_rule": args.speaker_rule, "supersession_filter": args.supersession_filter,
                        "standing_variant": args.standing_variant}
    save_json(path, state)
    print(f"[conv {conv['conv_id']}] store: {len(store)} facts, {len(active)} active, "
          f"{state['summary']['n_superseded']} superseded ({n_calls} decision calls)", flush=True)
    return store


def phase_ingest(args):
    os.makedirs(os.path.join(INGEST_DIR, INGEST_TAG), exist_ok=True)
    convs = load_beam(args.convs)
    ready = []
    for conv in convs:
        extracted, state = extract_facts(conv, args)
        if args.extract_only:
            continue
        if args.redo_decisions:
            state.pop("decisions", None); state.pop("store", None); state.pop("summary", None)
        ready.append((conv, extracted, state))
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(ready)))) as ex:
        futs = [ex.submit(build_store, conv, extracted, state, args) for conv, extracted, state in ready]
        for f in futs:
            f.result()
    print("usage:", USAGE, flush=True)


# --------------------------------------------------------------------------
# phase: qa
# --------------------------------------------------------------------------
class Ranker:
    """QA-time ranking of a text pool against a question, one mode for BOTH arms:
    tfidf  unigram TF-IDF cosine (the paper's transcript-RAG retriever)
    dense  bge-small-en-v1.5 cosine (BEAM's own RAG retriever)
    fusion reciprocal-rank fusion of the two (RRF, k=60)"""

    def __init__(self, mode="tfidf", cache=None):
        self.mode = mode
        self.index = CandidateIndex(cache=cache) if mode in ("dense", "fusion") else None
        if self.index is not None:
            self.index._load()          # eager, main thread
        self._pool_emb = {}

    def _dense_order(self, key, texts, query):
        if key not in self._pool_emb:
            self._pool_emb[key] = np.stack([self.index.embed((key, i), t) for i, t in enumerate(texts)])
        q = self.index.embed(("q", query), query)
        sims = self._pool_emb[key] @ q
        return sorted(range(len(texts)), key=lambda i: (-float(sims[i]), i))

    def order(self, key, texts, query):
        if not texts:
            return []
        if self.mode == "tfidf":
            return tfidf_rank(texts, query)
        dense = self._dense_order(key, texts, query)
        if self.mode == "dense":
            return dense
        lex = tfidf_rank(texts, query)
        score = {}
        for r, i in enumerate(dense):
            score[i] = score.get(i, 0.0) + 1.0 / (60 + r)
        for r, i in enumerate(lex):
            score[i] = score.get(i, 0.0) + 1.0 / (60 + r)
        return sorted(range(len(texts)), key=lambda i: (-score[i], i))


ANNOTATE = "changed"
RETRACTION_FILTER = "flag"

VALUE_TOKEN_RE = re.compile(
    r"(?P<time>\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)"
    r"|(?P<date>\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b)"
    r"|(?P<iso>\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?P<year>\b(?:19|20)\d{2}\b)"
    r"|(?P<money>\$\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?P<pct>\d+(?:\.\d+)?\s*%)"
    r"|(?P<unit>\d[\d,]*(?:\.\d+)?\s*(?:words|miles|hours|minutes|weeks|days|months|years|cm|kg|lbs|pages|books|problems|sessions|sources|courses|scenes))"
    r"|(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<wday>\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)", re.I)
NAME_STOP = {"The", "A", "An", "I", "It", "This", "That", "These", "Those", "They", "He", "She", "We", "You", "User", "Assistant"}
ADVICE_RE = re.compile(r"^The assistant\b|\b(?:should|is advised|is encouraged|is recommended|suggests|recommends|advises|can|could|may|might)\b")


def value_tokens(text):
    """Values in a statement, normalised: times (3 PM = 3:00 pm), dates without the year, years,
    money, percentages, numbers with a unit, weekdays, and bare numbers of two or more digits
    (a bare single digit is too common to identify a value)."""
    out = set()
    for m in VALUE_TOKEN_RE.finditer(text):
        kind, v = m.lastgroup, re.sub(r"\s+", " ", m.group(0).strip().lower())
        if kind == "time":
            h, rest = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", v).groups()[0], re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", v).groups()[1:]
            v = f"{int(h)}:{rest[0] or '00'} {rest[1]}"
        elif kind == "num" and len(v.replace(",", "")) < 2:
            continue
        elif kind in ("money", "pct", "unit", "num"):
            v = v.replace(" ", "").replace(",", "").lstrip("$")     # $60,000 = 60,000 = 60000
        out.add(v)
    return out


def name_tokens(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return {w[:-2] if w.endswith("'s") else w for w in words[1:] if w[0].isupper() and w not in NAME_STOP}


def is_retraction(old, new):
    """Is `new` (which supersedes `old`) rendered as a retraction? flag: the Interpreter's
    value_changed flag; value: the flag and a value or name in `new` that `old` lacks."""
    if new.get("value_changed") is False:
        return False
    if RETRACTION_FILTER == "value":
        return bool(value_tokens(new["text"]) - value_tokens(old["text"])) or bool(name_tokens(new["text"]) - name_tokens(old["text"]))
    return True


def render_fact(rec, store):
    when = f", {rec['time_anchor']}" if rec.get("time_anchor") else ""
    line = f"[fact {rec['fid']}, msg {rec['msg_id']}, {rec['role']}{when}] {rec['text']}"
    if rec.get("supersedes") is not None and ANNOTATE != "none":
        old = store[rec["supersedes"]]
        if ANNOTATE == "all" or is_retraction(old, rec):
            line += f" (replaces fact {old['fid']} from msg {old['msg_id']}: {old['text']})"
    return line


FACT_ROLES = ("user", "assistant")


def verifier_context(store, question, k, ranker=None):
    active = [r for r in store if r["status"] == "active" and r["role"] in FACT_ROLES]
    ranker = ranker or Ranker("tfidf")
    order = ranker.order(("facts", id(store)), [r["text"] for r in active], question)
    picked = sorted((active[i] for i in order[:k]), key=lambda r: r["fid"])
    return "\n".join(render_fact(r, store) for r in picked)


def facts_context(store, question, k, ranker, active_only, annotate):
    """MemAB-style pair: the same extracted facts, the same retriever and k;
    active_only=False is transcript-RAG over ALL facts (stale and current mixed),
    active_only=True is the verifier. Plain lines when annotate is False."""
    pool = [r for r in store if (r["status"] == "active" or not active_only)]
    order = ranker.order(("facts-all" if not active_only else "facts", id(store)), [r["text"] for r in pool], question)
    picked = sorted((pool[i] for i in order[:k]), key=lambda r: r["fid"])
    if annotate:
        return "\n".join(render_fact(r, store) for r in picked)
    return "\n".join(f"[fact {r['fid']}, msg {r['msg_id']}, {r['role']}" + (f", {r['time_anchor']}" if r.get('time_anchor') else "") + f"] {r['text']}" for r in picked)


def facts_std_context(store, question, k, ranker):
    """Soft standing: retrieve over ALL facts (like tr_facts) and let the engine's
    verdict travel with each fact instead of filtering: a superseded fact is marked
    with what replaced it, a superseding fact with what it replaced. The QA model
    sees every statement plus the engine's opinion, so an engine error is recoverable."""
    pool = list(store)
    order = ranker.order(("facts-all", id(store)), [r["text"] for r in pool], question)
    picked = sorted((pool[i] for i in order[:k]), key=lambda r: r["fid"])
    lines = []
    for r in picked:
        head = f"[fact {r['fid']}, msg {r['msg_id']}, {r['role']}" + (f", {r['time_anchor']}" if r.get('time_anchor') else "") + "]"
        if r["status"] == "superseded":
            new = store[r["superseded_by"]]
            lines.append(f"{head} {r['text']}  [SUPERSEDED by msg {new['msg_id']}: {new['text']}]")
        else:
            lines.append(f"{head} {r['text']}")
    return "\n".join(lines)


def verifier_context_roles(store, question, k, ranker, roles):
    """verifier rendering restricted to commitments made by the given speakers."""
    active = [r for r in store if r["status"] == "active" and r["role"] in roles]
    order = ranker.order(("facts-" + "-".join(roles), id(store)), [r["text"] for r in active], question)
    picked = sorted((active[i] for i in order[:k]), key=lambda r: r["fid"])
    return "\n".join(render_fact(r, store) for r in picked)


def verifier_src_context(store, messages, question, k, ranker, char_budget):
    """Same state as `verifier`, rendered as source text: rank the ACTIVE facts, add the
    original message each fact came from (in rank order) until the budget is used, then
    list the statements in those messages that were later superseded, each with the
    statement that replaced it. The QA model sees the raw wording of current statements
    and an explicit retraction list; superseded turns that carry no active fact are never
    included."""
    active = [r for r in store if r["status"] == "active"]
    order = ranker.order(("facts", id(store)), [r["text"] for r in active], question)
    msg_by_id = {m["id"]: m for m in messages}
    picked, used = [], 0
    for i in order[:k]:
        mid = active[i]["msg_id"]
        if mid in picked or mid not in msg_by_id:
            continue
        m = msg_by_id[mid]
        text = f"[msg {mid} {m['role']}] {m['content']}"
        if used + len(text) > char_budget:
            if picked:
                continue
            text = text[:char_budget]                    # at least one message, truncated
        picked.append(mid); used += len(text)
        if used >= char_budget:
            break
    body = "\n\n".join(f"[msg {mid} {msg_by_id[mid]['role']}] {msg_by_id[mid]['content']}"
                        if len(f"[msg {mid} {msg_by_id[mid]['role']}] {msg_by_id[mid]['content']}") <= char_budget
                        else f"[msg {mid} {msg_by_id[mid]['role']}] {msg_by_id[mid]['content']}"[:char_budget]
                        for mid in sorted(picked))
    notes = []
    pset = set(picked)
    for r in store:
        if r["status"] == "superseded" and r["msg_id"] in pset and r["superseded_by"] is not None:
            new = store[r["superseded_by"]]
            if not is_retraction(r, new):
                continue                                 # restatement, same value
            notes.append(f"- [msg {r['msg_id']}] {r['text']}  ->  later replaced by [msg {new['msg_id']}]: {new['text']}")
    if notes:
        body += ("\n\nSTATEMENTS ABOVE THAT WERE LATER REPLACED (do not use them; the current "
                 "statement follows the arrow):\n" + "\n".join(notes[:30]))
    return body


def retraction_notes(store, msg_ids):
    """Engine standing rendered as a retraction list for the given messages: every
    statement from those messages that was later replaced by a different value."""
    notes = []
    for r in store:
        if r["status"] == "superseded" and r["msg_id"] in msg_ids and r["superseded_by"] is not None:
            new = store[r["superseded_by"]]
            if not is_retraction(r, new):
                continue
            notes.append(f"- [msg {r['msg_id']}] {r['text']}  ->  later replaced by [msg {new['msg_id']}]: {new['text']}")
    return notes


def tr_std_context(pairs, pair_msg_ids, store, question, char_budget, ranker):
    """transcript-RAG, identical retrieval and text, plus the engine's retraction list
    for the retrieved pairs: the literal 'RAG minus what was retracted' comparison."""
    order = ranker.order(("pairs", id(pairs)), pairs, question)
    picked, used = [], 0
    for i in order:
        if picked and used + len(pairs[i]) > char_budget:
            continue
        picked.append(i); used += len(pairs[i])
        if used >= char_budget:
            break
    body = "\n\n".join(pairs[i] for i in sorted(picked))
    ids = {m for i in picked for m in pair_msg_ids[i]}
    notes = retraction_notes(store, ids)
    if notes:
        body += ("\n\nSTATEMENTS ABOVE THAT WERE LATER REPLACED (do not use them; the current "
                 "statement follows the arrow):\n" + "\n".join(notes[:30]))
    return body


def turn_pairs(messages, with_ids=False):
    pairs, ids, i = [], [], 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            a = messages[i + 1]
            pairs.append(f"[msg {m['id']} user] {m['content']}\n[msg {a['id']} assistant] {a['content']}")
            ids.append((m["id"], a["id"])); i += 2
        else:
            pairs.append(f"[msg {m['id']} {m['role']}] {m['content']}")
            ids.append((m["id"],)); i += 1
    return (pairs, ids) if with_ids else pairs


def tr_context(pairs, question, char_budget, ranker=None):
    ranker = ranker or Ranker("tfidf")
    order = ranker.order(("pairs", id(pairs)), pairs, question)
    picked, used = [], 0
    for i in order:
        if picked and used + len(pairs[i]) > char_budget:
            continue
        picked.append(i); used += len(pairs[i])
        if used >= char_budget:
            break
    return "\n\n".join(pairs[i] for i in sorted(picked))


def full_context(messages, char_budget):
    lines = [f"[msg {m['id']} {m['role']}] {m['content']}" for m in messages]
    text = "\n".join(lines)
    if len(text) > char_budget:
        text = "[... earlier conversation truncated ...]\n" + text[-char_budget:]
    return text


def phase_qa(args):
    out_path = os.path.join(OUT_DIR, f"qa_{args.tag}.json")
    results = json.load(open(out_path)) if (os.path.exists(out_path) and not args.force_restart) else {}
    convs = load_beam(args.convs)
    for conv in convs:
        cid = conv["conv_id"]
        ip = ingest_path(cid)
        have_store = os.path.exists(ip) and "store" in json.load(open(ip))
        arms_here = list(args.arms)
        if not have_store:
            arms_here = [a for a in arms_here if a in ("full", "beamrag")]
            if not arms_here:
                print(f"[conv {cid}] no ingest store yet, skipping", flush=True); continue
            print(f"[conv {cid}] no ingest store yet: running only {arms_here}", flush=True)
        store = json.load(open(ip))["store"] if have_store else []
        budget_store = store
        if args.budget_ingest_tag and have_store:
            budget_store = json.load(open(os.path.join(INGEST_DIR, args.budget_ingest_tag, f"conv_{cid}.json")))["store"]
        pairs, pair_ids = turn_pairs(conv["messages"], with_ids=True)
        fullctx = full_context(conv["messages"], args.ctx_char_budget)
        emb_cache = EmbCache(os.path.join(OUT_DIR, "emb_cache", f"conv_{cid}.npz"))
        ranker = Ranker(args.retriever, cache=emb_cache)
        # embed the pools once, before the thread pool starts
        if args.retriever != "tfidf" and store:
            ranker.order(("facts", id(store)), [r["text"] for r in store if r["status"] == "active"], "warm up")
            if budget_store is not store:
                ranker.order(("facts", id(budget_store)), [r["text"] for r in budget_store if r["status"] == "active"], "warm up")
            ranker.order(("pairs", id(pairs)), pairs, "warm up")
            emb_cache.save()
        beam_ranker = None
        if "beamrag" in args.arms:
            # BEAM's own RAG baseline: bge-small over user-assistant turn pairs, top-5, no budget matching
            beam_ranker = ranker if args.retriever == "dense" else Ranker("dense", cache=emb_cache)
            beam_ranker.order(("pairs", id(pairs)), pairs, "warm up")
            emb_cache.save()
        jobs = []
        for cat in CATEGORIES:
            for qi, item in enumerate(conv["questions"].get(cat, [])):
                key = f"{cid}|{cat}|{qi}"
                rec = results.get(key) or {"conv": cid, "cat": cat, "idx": qi,
                                           "question": item["question"], "gold": gold_of(item),
                                           "rubric": item.get("rubric", []),
                                           "source_ids": source_ids(item), "arms": {}}
                results[key] = rec
                for arm in arms_here:
                    if arm in rec["arms"] and rec["arms"][arm].get("resp") is not None:
                        continue
                    jobs.append((key, arm, item["question"]))
        print(f"[conv {cid}] {len(jobs)} (question, arm) jobs", flush=True)
        lock = threading.Lock()

        def run(job):
            key, arm, q = job
            if arm == "verifier":
                ctx = verifier_context(store, q, args.top_k, ranker)
            elif arm == "facts_std":
                ctx = facts_std_context(store, q, args.top_k, ranker)
            elif arm == "tr_facts":
                ctx = facts_context(store, q, args.top_k, ranker, active_only=False, annotate=False)
            elif arm == "verifier_plain":
                ctx = facts_context(store, q, args.top_k, ranker, active_only=True, annotate=False)
            elif arm == "verifier_user":
                ctx = verifier_context_roles(store, q, args.top_k, ranker, ("user",))
            elif arm == "tr_std":
                vctx = verifier_context(budget_store, q, args.top_k, ranker)
                ctx = tr_std_context(pairs, pair_ids, store, q, max(len(vctx), 500), ranker)
            elif arm == "verifier_src":
                vctx = verifier_context(budget_store, q, args.top_k, ranker)
                ctx = verifier_src_context(store, conv["messages"], q, args.top_k, ranker, max(len(vctx), 500))
            elif arm == "beamrag":
                order = beam_ranker.order(("pairs", id(pairs)), pairs, q)
                ctx = "\n\n".join(pairs[i] for i in sorted(order[:args.beam_rag_k]))
            elif arm == "tr":
                vctx = verifier_context(budget_store, q, args.top_k, ranker)
                ctx = tr_context(pairs, q, max(len(vctx), 500), ranker)
            else:
                ctx = fullctx
            prompt = ANSWER_PROMPT.replace("<context>", ctx).replace("<question>", q)
            resp = chat([{"role": "user", "content": prompt}], args.qa_base_url, args.qa_key,
                        args.qa_model, max_tokens=args.max_answer_tokens)
            with lock:
                results[key]["arms"][arm] = {"resp": resp, "ctx_chars": len(ctx),
                                             "ctx_tokens_est": est_tokens(ctx)}
            return key

        primer = next((j for j in jobs if j[1] == "full"), None)
        if primer is not None:
            run(primer); jobs.remove(primer)      # populates the provider's prompt cache for this transcript
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if i % 10 == 0:
                    with lock:
                        merge_save_results(out_path, results)
                    print(f"  [conv {cid}] {i}/{len(jobs)}", flush=True)
        merge_save_results(out_path, results)
    results["_config"] = {"qa_model": args.qa_model, "top_k": args.top_k, "retriever": args.retriever, "annotate": ANNOTATE,
                          "retraction_filter": RETRACTION_FILTER,
                          "ingest_tag": INGEST_TAG, "budget_ingest_tag": args.budget_ingest_tag,
                          "ctx_char_budget": args.ctx_char_budget, "arms": args.arms,
                          "answer_prompt": "BEAM answer_generation_for_rag (all arms)",
                          "usage": dict(USAGE)}
    merge_save_results(out_path, results)
    print("usage:", USAGE, flush=True)


# --------------------------------------------------------------------------
# phase: judge
# --------------------------------------------------------------------------
def judge_one(resp, rubric, args):
    scores, reasons = [], []
    for item in rubric:
        prompt = JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", resp or "")
        out = chat([{"role": "user", "content": prompt}], args.judge_base_url, args.judge_key,
                   args.judge_model, max_tokens=300)
        d = parse_json(out) or {}
        s = d.get("score") if isinstance(d, dict) else None
        try:
            s = float(s)
        except Exception:
            m = re.search(r'"score"\s*:\s*([01](?:\.[05])?)', out or "")
            s = float(m.group(1)) if m else 0.0
        s = min(max(s, 0.0), 1.0)
        scores.append(s); reasons.append((d.get("reason") if isinstance(d, dict) else out) or "")
    return {"scores": scores, "mean": (sum(scores) / len(scores)) if scores else 0.0,
            "reasons": reasons, "judge": args.judge_model}


def phase_judge(args):
    out_path = os.path.join(OUT_DIR, f"qa_{args.tag}.json")
    results = json.load(open(out_path))
    jobs = [(k, arm) for k, rec in results.items() if not k.startswith("_")
            for arm, a in rec["arms"].items()
            if a.get("resp") is not None and (args.rejudge or args.judge_field not in a)]
    print(f"{len(jobs)} (question, arm) to judge with {args.judge_model}", flush=True)
    lock = threading.Lock()

    def run(job):
        k, arm = job
        j = judge_one(results[k]["arms"][arm]["resp"], results[k]["rubric"], args)
        with lock:
            results[k]["arms"][arm][args.judge_field] = j
        return k

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 20 == 0:
                with lock:
                    merge_save_results(out_path, results)
                print(f"  judged {i}/{len(jobs)}", flush=True)
    merge_save_results(out_path, results)
    print("usage:", USAGE, flush=True)


# --------------------------------------------------------------------------
# phase: report
# --------------------------------------------------------------------------
def ku_capture(convs):
    """For each knowledge_update item: did a fact from the updated message replace
    a fact from the original message(s)? Is the original superseded at all?"""
    rows = []
    for conv in convs:
        ip = ingest_path(conv["conv_id"])
        if not os.path.exists(ip):
            continue
        store = json.load(open(ip)).get("store", [])
        by_msg = {}
        for r in store:
            by_msg.setdefault(r["msg_id"], []).append(r)
        for item in conv["questions"].get("knowledge_update", []):
            sc = source_ids(item)
            orig, upd = sc.get("original_info", []), sc.get("updated_info", [])
            upd_facts = [r for m in upd for r in by_msg.get(m, [])]
            orig_facts = [r for m in orig for r in by_msg.get(m, [])]
            chain = any(r["supersedes"] is not None and store[r["supersedes"]]["msg_id"] in orig
                        for r in upd_facts)
            rows.append({"conv": conv["conv_id"], "q": item["question"][:70],
                         "upd_extracted": len(upd_facts) > 0,
                         "upd_active": any(r["status"] == "active" for r in upd_facts),
                         "orig_superseded": any(r["status"] == "superseded" for r in orig_facts),
                         "chain_upd_replaces_orig": chain, "n_upd_msgs": len(upd)})
    return rows


def phase_report(args):
    out_path = os.path.join(OUT_DIR, f"qa_{args.tag}.json")
    results = json.load(open(out_path))
    present = {a for k, rec in results.items() if not k.startswith("_") for a in rec["arms"]}
    ORDER = ("full", "tr", "tr_std", "beamrag", "tr_facts", "facts_std", "verifier_plain", "verifier", "verifier_user", "verifier_src")
    arms = [a for a in ORDER if a in present] + sorted(present - set(ORDER))
    table = {c: {a: [] for a in arms} for c in CATEGORIES}
    ctx = {a: [] for a in arms}
    for k, rec in results.items():
        if k.startswith("_"):
            continue
        for a, r in rec["arms"].items():
            if args.judge_field in r:
                table[rec["cat"]][a].append(r[args.judge_field]["mean"])
            ctx[a].append(r.get("ctx_tokens_est", 0))
    print(f"\n== BEAM 100K, tag={args.tag}, QA model={results.get('_config', {}).get('qa_model')}, judge field={args.judge_field}")
    print(f"{'ability':<26}" + "".join(f"{a:>12}" for a in arms) + "     n")
    gen, sup = {a: [] for a in arms}, {a: [] for a in arms}
    for c in CATEGORIES:
        n = len(table[c][arms[0]])
        print(f"{c:<26}" + "".join(f"{(np.mean(table[c][a]) if table[c][a] else float('nan')):>12.3f}" for a in arms) + f"  {n:>4}")
        for a in arms:
            (sup if c in SUPERSESSION_CATS else gen)[a].extend(table[c][a])
    print("-" * (26 + 12 * len(arms) + 6))
    print(f"{'general (8 abilities)':<26}" + "".join(f"{np.mean(gen[a]) if gen[a] else float('nan'):>12.3f}" for a in arms) + f"  {len(gen[arms[0]]):>4}")
    print(f"{'supersession (KU + CR)':<26}" + "".join(f"{np.mean(sup[a]) if sup[a] else float('nan'):>12.3f}" for a in arms) + f"  {len(sup[arms[0]]):>4}")
    print(f"{'mean context tokens':<26}" + "".join(f"{np.mean(ctx[a]) if ctx[a] else 0:>12.0f}" for a in arms))
    convs = load_beam(args.convs) if args.convs else load_beam(sorted({rec['conv'] for k, rec in results.items() if not k.startswith('_')}))
    rows = ku_capture(convs)
    if rows:
        print(f"\n== knowledge-update capture over {len(rows)} items")
        for key in ("upd_extracted", "upd_active", "orig_superseded", "chain_upd_replaces_orig"):
            print(f"  {key:<26} {sum(1 for r in rows if r[key])}/{len(rows)}")
        for r in rows:
            print("  ", {k: v for k, v in r.items() if k not in ('n_upd_msgs',)})
    for conv in convs:
        ip = ingest_path(conv["conv_id"])
        if os.path.exists(ip):
            print(f"\n[conv {conv['conv_id']}] ingest summary:", json.load(open(ip)).get("summary"))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["ingest", "qa", "judge", "report"])
    ap.add_argument("--convs", nargs="+", type=int, default=None, help="conversation ids (default: all 20)")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force-restart", action="store_true")
    # ingest
    ap.add_argument("--interp-model", default="gpt-4o")
    ap.add_argument("--interp-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--interp-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--ingest-tag", default="gpt-4o",
                    help="name of the ingest cache (which Interpreter produced the store)")
    ap.add_argument("--max-facts", type=int, default=10, help="facts per message cap")
    ap.add_argument("--max-msg-chars", type=int, default=12000)
    ap.add_argument("--cand-k", type=int, default=8, help="top-n per retrieval view (dense, bigram TF-IDF, unigram TF-IDF) for the supersession decision")
    ap.add_argument("--cand-cap", type=int, default=20, help="max candidates shown to the supersession decision")
    ap.add_argument("--extract-only", action="store_true", help="run the extraction pass only (no standing pass)")
    ap.add_argument("--supersession-filter", choices=["none", "value-slot"], default="none",
                    help="value-slot: a new statement without any number/date cannot supersede one that has one")
    ap.add_argument("--speaker-rule", choices=["none", "user-authority"], default="none",
                    help="user-authority: an assistant statement never supersedes a user commitment")
    ap.add_argument("--standing-variant", choices=["none", "user-confirm", "restatement-stands", "user-grounded"], default="none",
                    help="replay variants of the standing rule on cached decisions: user-confirm = an assistant "
                         "revision retires a statement only if a later user statement restates the new value; "
                         "restatement-stands = a same-value restatement retires nothing; user-grounded = an "
                         "assistant statement that changes a recorded value no user turn in between states is not "
                         "admitted (status unconfirmed) and retires nothing")
    ap.add_argument("--decision-batching", choices=["none", "message"], default="none",
                    help="none = original sequential standing pass (default); message = decide a message's facts concurrently")
    ap.add_argument("--value-change-mode", choices=["ignore", "annotate", "require"], default="annotate",
                    help="ignore: original rule, flag not used; annotate: original rule, value_changed stored and used "
                         "only for the (replaces ...) annotation; require: supersede only when the value changed (ablation)")
    ap.add_argument("--require-value-change", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--redo-decisions", action="store_true", help="discard cached supersession decisions and rebuild the store")
    # qa
    ap.add_argument("--arms", nargs="+", default=["full", "tr", "verifier"], choices=["full", "tr", "verifier", "beamrag", "verifier_src", "tr_std", "verifier_user", "tr_facts", "verifier_plain", "facts_std"])
    ap.add_argument("--beam-rag-k", type=int, default=5, help="turn pairs for the beamrag arm (BEAM's official RAG uses 5)")
    ap.add_argument("--qa-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--qa-base-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--top-k", type=int, default=30, help="active facts retrieved for the verifier arm")
    ap.add_argument("--budget-ingest-tag", default=None,
                    help="compute the tr / tr_std / verifier_src character budget from this store instead of "
                         "--ingest-tag (keeps budgets fixed when comparing standing-rule variants)")
    ap.add_argument("--retraction-filter", choices=["flag", "value"], default="flag",
                    help="which superseded statements the retraction list and the (replaces ...) annotation show: "
                         "flag = the Interpreter's value_changed flag (default); value = the flag and a number, date, "
                         "time or name in the new statement that the old one lacks")
    ap.add_argument("--annotate", choices=["all", "changed", "none"], default="changed",
                    help="which superseding facts carry the (replaces ...) annotation; legacy stores without the flag annotate all")
    ap.add_argument("--retriever", choices=["tfidf", "dense", "fusion"], default="tfidf",
                    help="QA-time ranker used identically by the tr and verifier arms")
    ap.add_argument("--ctx-char-budget", type=int, default=100000,
                    help="full-context tail-truncation budget in characters (~25K tokens for a 32K window; use 480000 for GPT-4o)")
    ap.add_argument("--max-answer-tokens", type=int, default=400)
    # judge
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--judge-base-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--judge-key-env", default="VLLM_API_KEY")
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--judge-field", default="judge",
                    help="where judge results are stored/read (use e.g. judge_gpt4o for a second judge)")
    args = ap.parse_args()
    args.interp_key = os.environ.get(args.interp_key_env, "dummy")
    args.qa_key = os.environ.get(args.api_key_env, "dummy")
    args.judge_key = os.environ.get(args.judge_key_env, "dummy")
    if args.require_value_change:
        args.value_change_mode = "require"
    global INGEST_TAG, ANNOTATE, RETRACTION_FILTER
    INGEST_TAG = args.ingest_tag
    ANNOTATE = args.annotate
    RETRACTION_FILTER = args.retraction_filter
    os.makedirs(OUT_DIR, exist_ok=True)
    {"ingest": phase_ingest, "qa": phase_qa, "judge": phase_judge, "report": phase_report}[args.phase](args)


if __name__ == "__main__":
    main()
