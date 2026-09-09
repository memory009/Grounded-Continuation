#!/usr/bin/env python3
"""
Benchmark Adapter — Grounded Continuation runtime verifier (ICLR 2027 submission).

Connects the ECM pipeline to external benchmarks. LoCoMo evaluation runs
in one of two modes:

  official:    Prompts aligned with the published LoCoMo QA protocol
               (CONV_START_PROMPT, `DATE:/CONVERSATION:` blocks, turns
               rendered as `Speaker said, "text"`, cat-2 temporal suffix,
               cat-5 multiple-choice preprocessing with stable option
               ordering). Token-budget-aware context truncation. Used
               when reporting comparable numbers alongside LoCoMo
               baselines. Produces `baseline_*` and, when --with-engine,
               `hybrid_*` answer/score fields under `official_*` names.

  diagnostic:  ECM-oriented prompts that carry the engine state and
               dependency map alongside the raw dialogue. Not a comparable
               LoCoMo baseline; used to probe whether epistemic structure
               aids long-horizon QA. Produces `diagnostic_hybrid_*` and
               (with --run-baseline) `diagnostic_baseline_*` fields.

Scope of fidelity to the published LoCoMo protocol (see the
datasets_cache/locomo/task_eval/ reference implementation):
  - normalize_answer, f1_score, f1 (multi-answer), and exact_match
    match evaluation.py:80-145 for categories 1-4.
  - cat-5 scoring follows evaluation.py:216-221 — a correct answer is
    one whose text contains "not mentioned" or "no information
    available"; this is applied regardless of the `answer` field, so
    the 2 cat-5 items with a real answer (like "No") are scored
    correct only if the model output selects the "Not mentioned"
    option after MCQ post-processing.
  - cat-2 suffix and cat-5 MCQ preprocessing mirror
    gpt_utils.py:244-259 (with stable hash-based option ordering in
    place of the non-reproducible `random.random()` used upstream).
  - The official-mode context layout follows hf_llm_utils.py:181-222:
    sessions iterated forward, within-session turns prepended newest-
    first, session headers prepended after each session's inner loop.
    The resulting dialogue has the newest kept session at the top.
  - Cat-5 items without an `answer` field are not scored by this
    module; they are counted under `skipped_cat5_missing_answer`.

Usage:
    # Official LoCoMo-protocol eval with engine-augmented hybrid answer
    python benchmark_adapter.py --benchmark locomo \\
        --data /path/to/locomo10.json \\
        --locomo-mode official --with-engine --run-baseline \\
        --backend openai --model Qwen/Qwen2.5-7B-Instruct

    # Diagnostic ECM hybrid vs plain LLM baseline
    python benchmark_adapter.py --benchmark locomo \\
        --data /path/to/locomo10.json \\
        --locomo-mode diagnostic --run-baseline \\
        --backend openai --model Qwen/Qwen2.5-7B-Instruct
"""

import os, sys, json, re, string, hashlib, argparse, time
from collections import Counter
from typing import Optional
import pipeline
from pipeline import EpistemicPipeline

from nltk.stem import PorterStemmer
_PS = PorterStemmer()


# ============================================================
# LoCoMo-official QA metrics
# ------------------------------------------------------------
# Per-category scoring dispatch re-implements
# datasets_cache/locomo/task_eval/evaluation.py:189-221. For cats 1-4
# the scoring reproduces evaluation.py's Porter-stemmed token F1 and
# bag-of-words exact match. For cat-5 we apply the same substring
# check (evaluation.py:216-221) to the model output, even when the
# item has a real `answer`, because that is how the published
# evaluation scores every cat-5 item.
# ============================================================

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def _exact_match(prediction: str, ground_truth: str) -> int:
    p = _normalize_answer(prediction)
    g = _normalize_answer(ground_truth)
    return int(set(p.split()) == set(g.split()))


def _f1_single(prediction: str, ground_truth: str) -> float:
    pt = [_PS.stem(w) for w in _normalize_answer(prediction).split()]
    gt = [_PS.stem(w) for w in _normalize_answer(ground_truth).split()]
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(pt)
    recall = n / len(gt)
    return 2 * precision * recall / (precision + recall)


def _f1_multi(prediction: str, ground_truth: str) -> float:
    """Cat-1 multi-answer F1: split both by comma, mean of per-gt max."""
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    gts = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not gts:
        return 0.0
    return sum(max(_f1_single(p, g) for p in preds) for g in gts) / len(gts)


def _adversarial_correct(prediction: str) -> int:
    lo = prediction.lower()
    return int("no information available" in lo or "not mentioned" in lo)


def locomo_score(prediction: Optional[str], ground_truth: str,
                 category: Optional[int]) -> dict:
    """LoCoMo per-category F1/EM. See evaluation.py:189-224 for dispatch."""
    if prediction is None:
        return {"f1": 0.0, "em": 0}
    if category == 1:
        return {"f1": _f1_multi(prediction, ground_truth),
                "em": _exact_match(prediction, ground_truth)}
    if category in (2, 3, 4):
        gt = ground_truth.split(";")[0].strip() if category == 3 else ground_truth
        return {"f1": _f1_single(prediction, gt),
                "em": _exact_match(prediction, gt)}
    if category == 5:
        # evaluation.py:216-221 scores every cat-5 by substring test on
        # the model output. After MCQ post-processing the prediction is
        # either the real answer text or "Not mentioned in the
        # conversation"; only the latter is counted correct.
        c = _adversarial_correct(prediction)
        return {"f1": float(c), "em": c}
    return {"f1": _f1_single(prediction, ground_truth),
            "em": _exact_match(prediction, ground_truth)}


# ============================================================
# LoCoMo-official prompt constants + question preprocessing
# ------------------------------------------------------------
# Re-implemented from datasets_cache/locomo/task_eval/gpt_utils.py
# (CONV_START_PROMPT @ line 51, `Speaker said, "text"` turn format @
# line 93, cat-2 temporal suffix @ line 244, cat-5 MCQ template @ line
# 246, cat-5 post-processing @ line 128). The "wriiten" typo in the
# CONV_START_PROMPT is preserved for fidelity to the published prompt.
# ============================================================

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. The "
    "conversation takes place over multiple days and the date of each "
    "conversation is wriiten at the beginning of the conversation.\n\n"
)
CAT2_TEMPORAL_SUFFIX = (
    " Use DATE of CONVERSATION to answer with an approximate date."
)
CAT5_MCQ_TEMPLATE = "{question} Select the correct answer: (a) {a} (b) {b}. "
CAT5_NOT_MENTIONED = "Not mentioned in the conversation"
OFFICIAL_ANSWER_INSTRUCTION = (
    "Write the answer in a short phrase or a few words. "
    "Use the exact words from the conversation whenever possible. "
    "If the conversation does not contain the answer, respond with "
    '"No information available".'
)


def _stable_mcq_order(conv_id: str, q_idx: int) -> bool:
    """Deterministic 50/50 split for cat-5 option ordering.

    The official implementation uses random.random() < 0.5 which is not
    reproducible across runs. We replace it with a hash over
    (conv_id, q_idx) so the smoke and full runs are identical and so
    two backends can be compared turn-for-turn. Returns True iff the
    real answer should be option (a).
    """
    h = hashlib.sha256(f"{conv_id}:{q_idx}".encode()).hexdigest()
    return int(h[:8], 16) % 2 == 0


def _preprocess_question_official(qa: dict, conv_id: str, q_idx: int):
    """Return (llm_question, mcq_key, skip_reason).

    mcq_key is a dict mapping 'a'/'b' -> option text when cat-5 MCQ is
    constructed, else None. skip_reason is non-None when the item cannot
    be scored in official mode (cat-5 without `answer`); callers must
    record the item under skipped counts and not include it in F1/EM.
    """
    cat = qa.get("category")
    q = qa["question"]
    if cat == 2:
        return q + CAT2_TEMPORAL_SUFFIX, None, None
    if cat == 5:
        if qa.get("answer") in (None, ""):
            # True adversarial (no real answer field). The published protocol
            # scores these by checking for "not mentioned" in the raw
            # output; but that behavior does not match the MCQ pipeline
            # used for the 2 answerable cat-5 items, so we report them
            # as a separate bucket rather than silently conflate.
            return None, None, "cat5_missing_answer"
        actual = str(qa["answer"])
        real_is_a = _stable_mcq_order(conv_id, q_idx)
        if real_is_a:
            question = CAT5_MCQ_TEMPLATE.format(
                question=q, a=actual, b=CAT5_NOT_MENTIONED)
            key = {"a": actual, "b": CAT5_NOT_MENTIONED}
        else:
            question = CAT5_MCQ_TEMPLATE.format(
                question=q, a=CAT5_NOT_MENTIONED, b=actual)
            key = {"a": CAT5_NOT_MENTIONED, "b": actual}
        return question, key, None
    return q, None, None


def _postprocess_answer_official(raw_pred: Optional[str],
                                  mcq_key: Optional[dict]) -> Optional[str]:
    """Map `a`/`(a)` tokens back to option text for cat-5 MCQ.

    Mirrors get_cat_5_answer in gpt_utils.py:128 with a small extension
    to also recognize labels inside longer free-form answers (e.g. the
    model writes '(a) Not mentioned in the conversation').
    """
    if raw_pred is None or mcq_key is None:
        return raw_pred
    p = raw_pred.strip().lower()
    if not p:
        return raw_pred
    # Exact single char or parenthesized token
    if p in ("a", "b"):
        return mcq_key[p]
    if p in ("(a)", "(b)"):
        return mcq_key[p[1]]
    # Leading tag within a longer answer
    if p.startswith("(a)"):
        return mcq_key["a"]
    if p.startswith("(b)"):
        return mcq_key["b"]
    if p.startswith("a.") or p.startswith("a:") or p.startswith("a "):
        return mcq_key["a"]
    if p.startswith("b.") or p.startswith("b:") or p.startswith("b "):
        return mcq_key["b"]
    # Check if an option's text appears verbatim in the answer
    for k in ("a", "b"):
        if mcq_key[k].lower() in p:
            return mcq_key[k]
    return raw_pred


# ============================================================
# Context-window safeguard (conservative, tokenizer-free)
# ------------------------------------------------------------
# We do not import a model-specific tokenizer; instead we use a
# conservative chars-per-token estimate (3.0 is safe for English/CJK
# mixes with the Qwen2.5 BPE vocabulary). When a truncation actually
# fires, callers print a warning so the result file remains honest.
# ============================================================

_APPROX_CHARS_PER_TOKEN = 3.0


def _approx_tokens(s: str) -> int:
    return max(1, int(len(s) / _APPROX_CHARS_PER_TOKEN) + 1)


def _build_official_conversation(turns_by_session, question_text: str,
                                 model_ctx_tokens: int,
                                 reserve_tokens: int = 600):
    """Reproduce the LoCoMo-official context layout from
    datasets_cache/locomo/task_eval/hf_llm_utils.py:181-222 (same logic
    in gpt_utils.py/claude_utils.py/gemini_utils.py).

    Sessions iterated forward; within each session, turns iterated
    newest-first and each prepended to the running context with the
    official `{speaker} said, "{text}"\\n\\n` format. After each
    session's inner loop, the `\\nDATE: {date_time}\\nCONVERSATION:\\n`
    header is prepended to the context. The final output therefore has
    the newest kept session at the TOP and session 1 at the BOTTOM —
    this is the byte-order the published LoCoMo baselines see.

    Truncation: as soon as one turn would overflow the budget the loop
    terminates, committing only the turns (newest-first) that already
    fit in the current session. Later sessions that had not yet been
    visited are dropped entirely.

    Args:
        turns_by_session: iterable of (sess_idx:int, date_time:str, turns:list[dict]).
        question_text:    the final question string (already preprocessed).
        model_ctx_tokens: model context window (e.g., 16384 for Qwen-7B).
        reserve_tokens:   leave this many tokens free for the instruction
                          block, engine state (if appended), and answer.

    Returns:
        (context_text, truncated_flag, used_tokens_est, sessions_kept)
    """
    budget = max(0, model_ctx_tokens
                 - _approx_tokens(question_text) - reserve_tokens)
    if budget <= 0:
        return "", True, 0, 0

    query_conv = ""
    used = 0
    truncated = False
    stopped_inside_session = False
    sessions_kept = 0

    for sess_idx, date_time, turns in turns_by_session:
        header = f"\nDATE: {date_time}\nCONVERSATION:\n"
        header_t = _approx_tokens(header)
        had_any_turn = False
        for t in reversed(turns):
            turn_str = f'{t["speaker"]} said, "{t["text"]}"\n\n'
            tt = _approx_tokens(turn_str)
            # Reserve header_t now so an accepted turn can never push the
            # final session (turns + header) past budget.
            if used + header_t + tt > budget:
                stopped_inside_session = True
                truncated = True
                break
            query_conv = turn_str + query_conv
            used += tt
            had_any_turn = True
        if had_any_turn:
            query_conv = header + query_conv
            used += header_t
            sessions_kept += 1
        if stopped_inside_session:
            break

    return query_conv, truncated, used, sessions_kept


# ============================================================
# Conversation Loaders
# ============================================================

# Category labels inferred from the scoring logic in
# datasets_cache/locomo/task_eval/evaluation.py:
#   1 -> multi-hop   (multi-answer F1 over comma-separated sub-answers)
#   2 -> temporal    (official prompt appends "Use DATE ..." suffix)
#   3 -> open-domain (scoring trims answer to substring before ';')
#   4 -> single-hop  (no special handling)
#   5 -> adversarial (either MCQ with real-answer decoy, or "not mentioned"
#                     check when the item has no answer field)
# The raw integer `category` is the authoritative key used by
# `locomo_score` and the official preprocessing; the string `type` is
# a human-friendly label only. When the mapping is uncertain for a
# given int, we emit `cat_<N>` instead of guessing.
_CATEGORY_TYPE = {
    1: "multi_hop", 2: "temporal", 3: "open_domain",
    4: "single_hop", 5: "adversarial",
}


def load_locomo(data_path: str) -> list[dict]:
    """Load LoCoMo (snap-research/locomo, data/locomo10.json).

    Per-conversation output:
      {
        "id":                "conv-26",
        "turns":             pristine flat list [{speaker, text, session, dia_id}, ...],
        "turns_by_session":  [(sess_idx, date_time, [turn, ...]), ...] sorted ascending,
        "session_dates":     {sess_idx: date_time, ...},
        "speakers":          [speaker_a, speaker_b],
        "queries":           [{question, answer, adversarial_answer, category,
                               type, evidence}, ...],
      }

    Turns are not augmented with injected metadata. Session timestamps
    flow via `session_dates` and `turns_by_session` so callers can render
    them into an official `DATE:/CONVERSATION:` block (official mode) or
    a compact timeline prefix (diagnostic mode) without polluting the
    dialogue itself.
    """
    with open(data_path) as f:
        data = json.load(f)

    benchmarks = []
    for item in data:
        conv = item["conversation"]
        speaker_a = conv.get("speaker_a", "")
        speaker_b = conv.get("speaker_b", "")
        session_keys = sorted(
            [k for k in conv.keys()
             if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]))

        turns: list[dict] = []
        turns_by_session: list[tuple] = []
        session_dates: dict = {}
        for sk in session_keys:
            sess_idx = int(sk.split("_")[1])
            dt = conv.get(f"{sk}_date_time", "")
            if dt:
                session_dates[sess_idx] = dt
            session_turns: list[dict] = []
            for t in conv[sk]:
                td = {
                    "speaker": t["speaker"],
                    "text": t["text"],
                    "session": sess_idx,
                    "dia_id": t.get("dia_id"),
                }
                session_turns.append(td)
                turns.append(td)
            turns_by_session.append((sess_idx, dt, session_turns))

        queries = []
        for q in item.get("qa", []):
            cat = q.get("category")
            queries.append({
                "question": q["question"],
                "answer": q.get("answer"),  # may be None for pure adversarial
                "adversarial_answer": q.get("adversarial_answer"),
                "category": cat,
                "type": _CATEGORY_TYPE.get(cat, f"cat_{cat}"),
                "evidence": q.get("evidence", []),
            })

        benchmarks.append({
            "id": item.get("sample_id", "unknown"),
            "turns": turns,
            "turns_by_session": turns_by_session,
            "session_dates": session_dates,
            "speakers": [speaker_a, speaker_b],
            "queries": queries,
        })
    return benchmarks


def _locomo_truncate_to_n_turns(benchmarks: list[dict], n: int) -> None:
    """Pre-slice each LoCoMo conversation to at most `n` turns
    chronologically (oldest sessions first, cumulative count). In-place.

    Used by the tipping-point length-sweep experiment: every conversation
    ends up with exactly `n` (or its native length, whichever is smaller)
    cumulative turns, independent of per-conv token density, so the
    X-axis of the sweep is turn count rather than token budget.

    Mutates each bench dict:
      - `turns_by_session` shrunk and re-committed (tuples preserved)
      - `turns` (flat list) truncated to match
      - `session_dates` restricted to kept sessions
      - `n_turns_after_truncation` field added for downstream reporting

    Does not alter `queries` (we report per-bucket evidence-coverage
    separately instead of filtering).
    """
    for bench in benchmarks:
        tbs = bench.get("turns_by_session") or []
        new_tbs = []
        kept_sessions: set = set()
        remaining = n
        for sess_idx, date_time, session_turns in tbs:
            if remaining <= 0:
                break
            if len(session_turns) <= remaining:
                new_tbs.append((sess_idx, date_time, list(session_turns)))
                remaining -= len(session_turns)
                kept_sessions.add(sess_idx)
            else:
                new_tbs.append((sess_idx, date_time,
                                list(session_turns[:remaining])))
                kept_sessions.add(sess_idx)
                remaining = 0
        bench["turns_by_session"] = new_tbs
        # Re-derive the flat turn list so downstream fields stay consistent.
        bench["turns"] = [t for _, _, turns in new_tbs for t in turns]
        # Restrict session_dates to the kept sessions.
        if bench.get("session_dates"):
            bench["session_dates"] = {
                s: d for s, d in bench["session_dates"].items()
                if s in kept_sessions
            }
        bench["n_turns_after_truncation"] = sum(
            len(t) for _, _, t in new_tbs)


def load_mt_bench_101(data_path: str) -> list[dict]:
    """Load MT-Bench-101 (mtbench101/mt-bench-101; ACL 2024).

    Data source: `data/subjective/mtbench101.jsonl` from the official repo.
    Each JSONL line has exactly three top-level keys:
      {"task": <2-letter>, "id": <int>, "history": [{"user":<str>,"bot":<str>},...]}

    Returns one dict per dialogue:
      id:    f"{task}_{id}"   (task-unique)
      task:  2-letter task code
      turns: flattened [{speaker:"User"|"Assistant", text:str}, ...]
             (gold reference -- used for teacher-forced history and as
              the judge's ground-truth-history context; NOT the thing
              being evaluated. Per MT-Bench-101 protocol only the
              tested model's generation for the *last* round in each
              per-turn evaluation instance is scored.)
      raw_history: the original [{user,bot},...] preserved for the
                   evaluator which needs per-turn user/bot access.
      queries: empty (MT-Bench-101 does not use a separate QA list;
               the evaluator re-runs each assistant turn as its own
               test instance).
    """
    benchmarks = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            task = d["task"]
            sample_id = d["id"]
            raw_history = d["history"]
            turns: list[dict] = []
            for h in raw_history:
                turns.append({"speaker": "User", "text": h["user"]})
                turns.append({"speaker": "Assistant", "text": h["bot"]})
            benchmarks.append({
                "id": f"{task}_{sample_id}",
                "task": task,
                "sample_id": sample_id,
                "turns": turns,
                "raw_history": raw_history,
                "queries": [],
            })
    return benchmarks


# ============================================================
# MT-Bench-101 official evaluation protocol
# ------------------------------------------------------------
# Rubric text, judge intro, score-format instruction, task-skip/ref
# policy, and scoring aggregation ported verbatim from
# sefira/opencompass (forked from open-compass/opencompass):
#   opencompass/datasets/subjective/mtbench101.py (judge,
#     score_format, eval_*, unique_prompt, skip_first_tasks,
#     need_ref_tasks)
#   opencompass/summarizers/subjective/mtbench101.py
#     (post_process_mtbench101, get_final_results: per-(task,dialogue)
#      MIN then per-task mean).
# Judge model (gpt-4-1106-preview) and judge temperature (0.8) from
#   configs/eval_subjective_mtbench101.py.
# References: https://github.com/mtbench101/mt-bench-101 ; ACL 2024
# paper https://aclanthology.org/2024.acl-long.401/ .
# Any wording quirks (e.g. the "wriiten"-style rubric typos) are
# preserved so our judge prompts are byte-comparable to the published
# protocol.
# ============================================================

MTBENCH101_SKIP_FIRST_TASKS = {"FR", "CR", "AR", "SA", "SC", "CM"}
MTBENCH101_NEED_REF_TASKS = {"MR", "GR"}

MTBENCH101_JUDGE_INTRO = (
    "Please act as an impartial judge follow this instructions: In the following "
    "conversations, only the response of the 'assistant' in the last round of "
    "conversations is the output of the large language model (AI assistant) that "
    "needs to be evaluated.  Please act as an impartial judge and score this "
    "response on a scale of 1 to 10, where 1 indicates that the response "
    "completely fails to meet the criteria, and 10 indicates that the response "
    "perfectly meets all the evaluation criteria.    Note that only the response "
    "of the 'assistant' in the LAST ROUND of conversations is the output of the "
    "large language model (the AI assistant) that needs to be evaluated; the "
    "previous conversations is the groud truth history which do NOT need to be "
    "evaluated."
)

MTBENCH101_SCORE_FORMAT = (
    "\n\n Note that only the response of the 'assistant' in the LAST ROUND of "
    "conversations is the output of the large language model (the AI assistant) "
    "that needs to be evaluated!! You must provide your explanation. After "
    "providing your explanation, please show the score by strictly following "
    "this format: 'Rating: [[score]]', for example 'Rating: [[6]]'. The DIALGUE "
    "need to be judged is in this format: \n *** \n DIALGUE \n ***"
)

# Task-specific rubrics (verbatim from opencompass fork; 13 task codes).
MTBENCH101_RUBRICS = {
    "CM": (
        "The capacity of a large language model to recall and utilize previously "
        "mentioned information from earlier in the conversation is a critical "
        "indicator of its conversational memory abilities. This competency is "
        "essential for maintaining context and coherence throughout an extended "
        "dialogue. The performance of the AI assistant should be evaluated based "
        "on its ability to consistently reference and integrate past information "
        "into current responses. The evaluation criteria are as follows:\n"
        "\n1.Analyze whether the AI assistant appropriately recalls relevant "
        "details from earlier parts of the conversation when responding to "
        "'Human's inquiries or comments.\n"
        "2.Assess the AI assistant's ability to integrate the remembered "
        "information into its current responses in a way that is coherent and "
        "adds value to the dialogue.\n"
        "3.Examine the AI assistant's consistency in maintaining the context "
        "established by previous dialogue exchanges throughout the entire "
        "conversation.\n"
        "4.Evaluate the effectiveness of the AI assistant's memory recall in "
        "facilitating a smooth and logical progression of the conversation, "
        "avoiding repetitive or contradictory statements.\n"
        "Scoring Guidelines:\n"
        "\n1-3 points: The AI assistant demonstrates poor recall of previous "
        "conversation details, leading to inconsistent or contradictory "
        "responses, and fails to maintain the dialogue's context, resulting in "
        "a disjointed or unclear conversation flow.\n"
        "4-6 points: The AI assistant exhibits a moderate ability to remember "
        "past information, but its integration into the conversation is "
        "sporadic or partially effective, leading to a conversation that lacks "
        "full coherence or occasionally disregards established context.\n"
        "7-9 points: The AI assistant reliably recalls and utilizes earlier "
        "information, contributing to a coherent dialogue that respects the "
        "conversation's context, with minor lapses in memory that do not "
        "significantly disrupt the conversation flow.\n"
        "10 points: The AI assistant demonstrates exceptional memory recall, "
        "seamlessly weaving past details into current responses to enrich the "
        "dialogue and preserve context, ensuring a smooth and logical "
        "conversation that progresses naturally.\n"
        "When scoring, consider the significance of the AI assistant's memory "
        "recall to the overall quality of the conversation. If recalling past "
        "information was not necessary for a particular exchange, the AI "
        "assistant's failure to reference earlier dialogue should not impact "
        "the score negatively. However, if recalling previous information "
        "enhances the dialogue's clarity, relevance, and continuity, this "
        "should be regarded as a positive attribute of the language model's "
        "performance.\n"
        "\nPlease provide a rationale for your score, specifically addressing "
        "how the AI assistant's memory recall and the use of past information "
        "align with the evaluation criteria and contribute to the "
        "conversation's effectiveness."
    ),
    "SC": (
        "We aim to specifically evaluate the self-correction ability of the "
        "large language model (AI assistant) when the 'Human' identifies an "
        "error in the AI assistant's initial response. The criteria for "
        "evaluation are as follows:\n"
        "\n1. Upon 'Human' pointing out a potential mistake, the AI assistant "
        "must thoroughly assess its previous response and engage with 'Human's' "
        "observation.\n"
        "2. If the 'Human' has correctly identified an error, the AI assistant "
        "must acknowledge the mistake, correct it, and provide an updated and "
        "accurate response.\n"
        "\nScoring Guidelines:\n"
        "\n- 1-3 points: The AI assistant fails to recognize or adequately "
        "address the error identified by 'Human,' and does not make the "
        "necessary corrections to its response.\n"
        "- 4-6 points: The AI assistant recognizes the error identified by "
        "'Human' but may only partially correct the mistake or provide an "
        "incomplete updated response.\n"
        "- 7-9 points: The AI assistant correctly identifies and acknowledges "
        "the error, making a substantial correction to its response and "
        "effectively updating the answer.\n"
        "- 10 points: The AI assistant exhibits exceptional self-correction "
        "ability, promptly acknowledging the error and providing a "
        "comprehensive and precise updated response.\n"
        "\nIn addition to scoring, please provide a justification for your "
        "assessment, focusing on how effectively the AI assistant's reaction "
        "to 'Human's' identified error demonstrates its ability to "
        "self-correct and address the criteria outlined above."
    ),
    "SA": (
        "We aim to specifically evaluate the ability of the large language "
        "model (AI assistant) to maintain confidence in its original response "
        "when challenged by 'Human', provided that the challenge is incorrect. "
        "The criteria for evaluation are as follows:\n"
        "\n1. When 'Human' questions the content of the AI assistant's "
        "response, the AI assistant must engage with 'Human's' query and "
        "assess the validity of the challenge.\n"
        "2. If 'Human's challenge is based on incorrect information or "
        "assumptions, the AI assistant must confidently maintain its original "
        "answer and provide a clear and reasoned explanation to justify its "
        "stance.\n"
        "\nScoring Guidelines:\n"
        "\n- 1-3 points: The AI assistant either does not engage with 'Human's' "
        "challenge appropriately or changes its original response without "
        "proper justification, even when 'Human's' challenge is incorrect.\n"
        "- 4-6 points: The AI assistant engages with the challenge but may "
        "lack confidence in its response, providing a weak justification for "
        "maintaining its original answer.\n"
        "- 7-9 points: The AI assistant demonstrates the ability to "
        "appropriately assess the incorrect challenge and maintains its "
        "original response with a clear and well-supported justification.\n"
        "- 10 points: The AI assistant exhibits excellent ability to maintain "
        "confidence in its original response, providing a strong and "
        "convincing explanation that effectively addresses 'Human's' incorrect "
        "challenge.\n"
        "\nIn addition to scoring, please provide a justification for your "
        "assessment, focusing on how the AI assistant's reaction to the "
        "challenge reflects its understanding and confidence in its original "
        "response, and how well it meets the criteria outlined above."
    ),
    "AR": (
        "The AI assistant's understanding of references is essential for "
        "maintaining a coherent dialogue. The following criteria should be "
        "used to evaluate its performance:\n"
        "\n1. The AI assistant's response must demonstrate a correct "
        "understanding of referential information from questions asked by "
        "'Human,' which typically relate to content from the previous "
        "dialogue. Ideally, the AI should explicitly acknowledge or clarify "
        "these references in its reply.\n"
        "2. The response from the AI assistant should be consistent with the "
        "content of the 'Human's question in the current round, providing "
        "true and accurate information, free from misunderstandings or "
        "inaccuracies related to the references.\n"
        "\nScoring Guidelines:\n"
        "\n- 1-3 points: The AI assistant fails to recognize or correctly "
        "interpret the referential information, leading to responses that are "
        "either inaccurate or unrelated to the previous content.\n"
        "- 4-6 points: The AI assistant shows a partial understanding of "
        "references, but the response might include some inaccuracies or fail "
        "to fully utilize the referential information.\n"
        "- 7-9 points: The AI assistant's response indicates a good "
        "understanding of the references, with only slight inaccuracies or "
        "omissions in the connection to the previous dialogue.\n"
        "- 10 points: The AI assistant demonstrates excellent understanding "
        "and use of referential information, perfectly aligning its response "
        "with the previous content and the current question accurately and "
        "precisely.\n"
        "\nIn addition to the score, please provide an explanation that "
        "specifically addresses how the AI assistant's response demonstrates "
        "its ability or inability to understand and use referential "
        "information in accordance with the criteria above. "
    ),
    "TS": (
        "The AI assistant's ability to handle shifts in conversation topics "
        "is crucial for maintaining relevance and adaptability during a "
        "dialogue. This skill is particularly important when 'Human' "
        "introduces a new topic or changes the subject abruptly. The "
        "performance of the AI assistant should be evaluated on its capacity "
        "to smoothly transition between topics without being inappropriately "
        "influenced by previous dialogue content. The evaluation criteria are "
        "as follows:\n"
        "\n1. Identify whether the AI assistant can detect and acknowledge "
        "the change in topic introduced by 'Human' without reverting back to "
        "or becoming stuck on the previous subject.\n"
        "2. Evaluate the relevance of the AI assistant's responses to the "
        "new topic, ensuring they are not improperly influenced or colored by "
        "the preceding dialogue rounds.\n"
        "3. Assess the AI assistant's ability to provide coherent and "
        "contextually appropriate responses to the new subject, displaying an "
        "understanding of the conversation's evolving nature.\n"
        "4. Consider the AI assistant's proficiency in offering complete and "
        "insightful answers to the new topic, which demonstrate a clear break "
        "from past conversation threads.\n"
        "Scoring Guidelines:\n"
        "\n1-3 points: The AI assistant struggles with topic transitions, "
        "frequently reverting to or being influenced by the previous topic, "
        "resulting in irrelevant or confused responses to the new subject "
        "matter.\n"
        "4-6 points: The AI assistant shows a moderate ability to adapt to "
        "new topics, but occasionally exhibits lingering effects from earlier "
        "discussions, leading to partially relevant or less focused responses "
        "to the topic shifts.\n"
        "7-9 points: The AI assistant adapts to topic changes well, with "
        "minimal reference to or influence from prior topics, providing "
        "responses that are largely relevant and well-aligned with the new "
        "conversation direction.\n"
        "10 points: The AI assistant excels at adapting to topic shifts, "
        "seamlessly transitioning to and fully engaging with the new subject "
        "matter without any irrelevant carryover from previous dialogue "
        "content.\n"
        "When scoring, consider the smoothness of the AI assistant's "
        "transition between topics and its ability to engage with the new "
        "subject matter independently of the prior conversation. If a topic "
        "shift is not present or is so subtle that continuity with previous "
        "content is warranted, the AI assistant's ability to maintain "
        "coherence should not negatively affect the score. However, if a "
        "clear topic shift occurs and the AI assistant handles it deftly, "
        "providing relevant and insightful input on the new topic, this "
        "should be recognized as a positive aspect of its conversational "
        "capabilities.\n"
        "\nPlease provide a rationale for your score, specifically addressing "
        "the effectiveness of the AI assistant's topic transition and its "
        "relevance to the new subject matter in accordance with the "
        "evaluation criteria."
    ),
}


def _mtbench101_parse_score(judge_raw: Optional[str]) -> Optional[int]:
    """Mirror post_process_mtbench101: extract [[N]] -> int. Returns
    None if no match."""
    if not judge_raw:
        return None
    m = re.search(r"\[([0-9]+)\]", judge_raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _call_judge_openai(system_prompt: str, user_prompt: str,
                       api_key: str,
                       model: str = "gpt-4-turbo",
                       temperature: float = 0.8,
                       max_retries: int = 3,
                       judge_url: str = "https://api.openai.com/v1/chat/completions"
                       ) -> Optional[str]:
    """Dedicated judge call against api.openai.com — decoupled from the
    pipeline.call_llm dispatcher so the tested model (e.g. Qwen-7B on
    local vLLM) and the judge (GPT-4) can coexist in one process."""
    try:
        import requests
    except ImportError:
        return None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                judge_url,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": model,
                      "temperature": temperature,
                      "max_tokens": 1024,
                      "messages": [
                          {"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt},
                      ]},
                timeout=120)
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            print(f"  [judge] API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}")
        except Exception as e:
            print(f"  [judge] Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def _mtbench101_dep_map_stats(dep_map: dict) -> dict:
    """Per-turn/per-dialogue ECM-state snapshot: total entries, how
    many are auto-generated (`h_auto_*`), how many contain h->h edges,
    how many are degenerate (auto + single observation dep). Used to
    check whether the LoCoMo-style degenerate pattern recurs on
    MT-Bench-101."""
    total = len(dep_map or {})
    h_auto = 0
    h_to_h = 0
    degenerate = 0
    for h, deps in (dep_map or {}).items():
        deps_list = list(deps) if isinstance(deps, (list, set, tuple)) else [deps]
        is_auto = isinstance(h, str) and h.startswith("h_auto_")
        has_h_dep = any(isinstance(d, str) and d.startswith("h") for d in deps_list)
        if is_auto:
            h_auto += 1
        if has_h_dep:
            h_to_h += 1
        if is_auto and len(deps_list) == 1 and isinstance(deps_list[0], str) \
                and deps_list[0].startswith("o"):
            degenerate += 1
    return {
        "n_entries": total,
        "h_auto": h_auto,
        "h_to_h": h_to_h,
        "degenerate": degenerate,
    }


def _mtbench101_render_history_text(raw_history: list, up_to_turn_idx: int,
                                    include_current_bot: bool = False) -> str:
    """Render history as the OpenCompass-style text block used in the
    judge prompt_template. `up_to_turn_idx` is 0-indexed inclusive;
    include_current_bot=True renders the gold `bot` for that turn too
    (used only for teacher-forced context in the judge's ground-truth
    history when the *current* turn being scored is later).

    The last segment always ends with 'Assistant: '; callers append
    either the gold bot (for history up to a past turn) or the model's
    prediction (for the turn being evaluated).
    """
    parts = []
    for idx in range(up_to_turn_idx + 1):
        h = raw_history[idx]
        parts.append("\n\n Human: " + h["user"] + "\n\nAssistant: ")
        if idx < up_to_turn_idx or include_current_bot:
            parts.append(h["bot"])
    return "".join(parts)


def _mtbench101_build_tested_prompt(raw_history: list, turn_idx: int,
                                    engine_block: Optional[str] = None
                                    ) -> tuple[str, str]:
    """Construct (system, user) for the tested model's per-turn
    generation at turn `turn_idx`. Prior turns' gold bot responses are
    teacher-forced into the history verbatim; the current turn's gold
    bot is NEVER placed in the prompt (only the current user is).

    For hybrid_dep_only, `engine_block` is appended after history and
    before the final 'Assistant:' prefix so it serves as immediate
    context for the generation.
    """
    system = ("You are a helpful AI assistant responding to the user's "
              "latest message. Stay consistent with the prior turns.")
    conv_parts = []
    for idx in range(turn_idx):
        h = raw_history[idx]
        conv_parts.append(f"Human: {h['user']}\n\nAssistant: {h['bot']}\n\n")
    current = raw_history[turn_idx]
    conv_parts.append(f"Human: {current['user']}\n\n")
    history_text = "".join(conv_parts)
    if engine_block:
        user = (f"{history_text}"
                f"{engine_block}"
                f"Assistant: ")
    else:
        user = f"{history_text}Assistant: "
    return system, user


def _mtbench101_score_one(raw_history: list, turn_idx: int,
                          prediction: str, task: str,
                          judge_api_key: str,
                          judge_model: str,
                          judge_temperature: float
                          ) -> tuple[Optional[int], Optional[str]]:
    """Build the judge prompts and score a single turn. Returns
    (parsed_score, judge_raw)."""
    if task not in MTBENCH101_RUBRICS:
        # Task rubric not ported (outside the 5-task smoke set).
        return None, None
    history_text = _mtbench101_render_history_text(
        raw_history, up_to_turn_idx=turn_idx, include_current_bot=False)
    system_prompt = (MTBENCH101_JUDGE_INTRO
                     + MTBENCH101_RUBRICS[task]
                     + MTBENCH101_SCORE_FORMAT)
    if task in MTBENCH101_NEED_REF_TASKS:
        # Not used by the 5-task smoke (MR/GR). Kept here for completeness
        # so adding those rubrics later does not require restructuring.
        ref_answer = raw_history[turn_idx]["bot"]
        user_prompt = (
            f"The dialogue need to be judged is: \n *** \n "
            f"{history_text}{prediction} \n ***\n\n"
            f"                    The reference solution is: \n ### \n "
            f"{ref_answer} \n ###\n\n")
    else:
        user_prompt = (
            f"The dialogue need to be judged is: \n *** \n "
            f"{history_text}{prediction} \n ***")
    judge_raw = _call_judge_openai(
        system_prompt, user_prompt, judge_api_key,
        model=judge_model, temperature=judge_temperature)
    parsed = _mtbench101_parse_score(judge_raw)
    return parsed, judge_raw


def evaluate_mtbench101_official(
        pipe_factory,
        bench: dict,
        tested_api_key: str,
        judge_api_key: str,
        judge_model: str = "gpt-4-1106-preview",
        judge_temperature: float = 0.8,
        run_baseline: bool = True,
        run_hybrid_dep_only: bool = True,
        ) -> dict:
    """Run one MT-Bench-101 dialogue end-to-end under the OpenCompass
    per-turn single-score protocol.

    For each evaluated turn k (skip_first respected):
      1. Ingest turn-k user into a fresh-per-dialogue EpistemicPipeline
         (built by `pipe_factory()`), so the dep_map snapshot reflects
         state up through the current user message.
      2. Build the tested model's prompt using teacher-forced gold
         history for turns <k and the live user for turn k. The gold
         bot for turn k is NEVER placed in the prompt.
      3. Generate the tested model's prediction (baseline and/or
         hybrid_dep_only).
      4. Hand the prediction to the GPT-4 judge with the task rubric.
      5. Teacher-force the gold bot for turn k into the pipeline so
         later-turn state is faithful to the reference conversation.

    Returns a dict with per-turn records and dialogue-level aggregates.
    """
    from pipeline import call_llm
    raw_history = bench["raw_history"]
    task = bench["task"]
    dialogue_id = bench["id"]
    skip_first = task in MTBENCH101_SKIP_FIRST_TASKS

    pipe_baseline = None  # Baseline does not use ECM; no pipe needed.
    pipe_hybrid = pipe_factory() if run_hybrid_dep_only else None

    per_turn_records: list[dict] = []
    dep_map_snapshots: list[dict] = []

    for turn_idx, h in enumerate(raw_history):
        if skip_first and turn_idx == 0:
            # Teacher-force first turn (not evaluated).
            if pipe_hybrid is not None:
                pipe_hybrid.process_turn("User", h["user"])
                pipe_hybrid.process_turn("Assistant", h["bot"])
            per_turn_records.append({
                "turn_id": turn_idx + 1,
                "evaluated": False,
                "skip_reason": "skip_first_task_turn_1",
            })
            continue

        # Ingest the current user into hybrid pipe before evaluation.
        if pipe_hybrid is not None:
            pipe_hybrid.process_turn("User", h["user"])
            dep_map = pipe_hybrid.get_dependency_map()
        else:
            dep_map = {}
        dep_stats = _mtbench101_dep_map_stats(dep_map)
        dep_map_snapshots.append({
            "turn_id": turn_idx + 1,
            "dep_map_stats": dep_stats,
            "dep_map": dep_map,
        })

        record: dict = {
            "turn_id": turn_idx + 1,
            "evaluated": True,
            "user": h["user"],
            "gold_bot": h["bot"],
            "dep_map_stats": dep_stats,
        }

        # ---- BASELINE ----
        if run_baseline:
            sys_b, usr_b = _mtbench101_build_tested_prompt(
                raw_history, turn_idx, engine_block=None)
            base_raw = call_llm(sys_b, usr_b, tested_api_key, temperature=0)
            base_ans = (base_raw or "").strip()
            base_score, base_judge_raw = _mtbench101_score_one(
                raw_history, turn_idx, base_ans, task,
                judge_api_key, judge_model, judge_temperature)
            record["baseline"] = {
                "system_prompt": sys_b,
                "user_prompt": usr_b,
                "model_answer": base_ans,
                "judge_raw": base_judge_raw,
                "score": base_score,
            }

        # ---- HYBRID (dep-map-only) ----
        if run_hybrid_dep_only:
            engine_block, engine_stats = _engine_state_block_from_dep_map(
                dep_map, mode="full")
            sys_h, usr_h = _mtbench101_build_tested_prompt(
                raw_history, turn_idx, engine_block=engine_block)
            hyb_raw = call_llm(sys_h, usr_h, tested_api_key, temperature=0)
            hyb_ans = (hyb_raw or "").strip()
            hyb_score, hyb_judge_raw = _mtbench101_score_one(
                raw_history, turn_idx, hyb_ans, task,
                judge_api_key, judge_model, judge_temperature)
            record["hybrid_dep_only"] = {
                "system_prompt": sys_h,
                "user_prompt": usr_h,
                "engine_block": engine_block,
                "engine_block_stats": engine_stats,
                "model_answer": hyb_ans,
                "judge_raw": hyb_judge_raw,
                "score": hyb_score,
            }

        per_turn_records.append(record)

        # Teacher-force the gold bot for future-turn state fidelity.
        if pipe_hybrid is not None:
            pipe_hybrid.process_turn("Assistant", h["bot"])

    # Dialogue-level min score per path (MIN across evaluated turns).
    def _min_over_evaluated(path_key):
        vals = [r[path_key]["score"] for r in per_turn_records
                if r.get("evaluated") and r.get(path_key, {}).get("score") is not None]
        return min(vals) if vals else None

    final_dep_map = pipe_hybrid.get_dependency_map() if pipe_hybrid else {}
    final_dep_stats = _mtbench101_dep_map_stats(final_dep_map)

    return {
        "id": dialogue_id,
        "task": task,
        "sample_id": bench.get("sample_id"),
        "n_turns": len(raw_history),
        "skip_first": skip_first,
        "n_evaluated_turns": sum(1 for r in per_turn_records if r.get("evaluated")),
        "per_turn": per_turn_records,
        "dialogue_baseline_min": _min_over_evaluated("baseline") if run_baseline else None,
        "dialogue_hybrid_dep_only_min": _min_over_evaluated("hybrid_dep_only") if run_hybrid_dep_only else None,
        "final_dep_map": final_dep_map,
        "final_dep_map_stats": final_dep_stats,
        "dep_map_snapshots": dep_map_snapshots,
    }


def load_custom_json(data_path: str) -> list[dict]:
    """Load either a single simple-format conversation or a list of
    structured conversations."""
    with open(data_path) as f:
        data = json.load(f)
    if isinstance(data, list) and data and "speaker" in data[0]:
        return [{"id": "custom", "turns": data, "queries": []}]
    return data


# ============================================================
# ECM engine state block (diagnostic mode only)
# ============================================================

def _timeline_block(session_dates: Optional[dict]) -> str:
    if not session_dates:
        return ""
    lines = ["## Session timeline"]
    for sess in sorted(session_dates.keys()):
        lines.append(f"Session {sess}: {session_dates[sess]}")
    return "\n".join(lines) + "\n\n"


def _trim_dep_map(dep_map: dict, top_k: int = 10) -> tuple[dict, dict]:
    """Ablation-only trimmer for the dep map injected into the LLM prompt.

    Rules (minimum version, per ablation spec 2026-04-23):
      - Drop entries whose hypothesis id matches `h_auto_*` AND whose
        dep list has exactly one element that is an observation
        (starts with `o`); these are the degenerate single-anchor
        auto-hypotheses that dominate LoCoMo engine state.
      - Among surviving entries, prioritize those whose dep list
        contains at least one hypothesis (a dep starting with `h`);
        these are the interesting h->h chains.
      - Keep at most `top_k` entries (h->h first, then others), in the
        dict's original insertion order within each group.

    Returns (trimmed_dep_map, stats). Stats reports the counts needed
    to diagnose how aggressive the trimming was.
    """
    deps_by_hyp = {h: list(v) for h, v in dep_map.items()}
    total = len(deps_by_hyp)

    filtered: dict = {}
    n_dropped_degenerate = 0
    for h, deps in deps_by_hyp.items():
        is_auto = h.startswith("h_auto_")
        is_single_obs = (len(deps) == 1
                         and isinstance(deps[0], str)
                         and deps[0].startswith("o"))
        if is_auto and is_single_obs:
            n_dropped_degenerate += 1
            continue
        filtered[h] = deps

    h_to_h_ids = [h for h, deps in filtered.items()
                  if any(isinstance(d, str) and d.startswith("h")
                         for d in deps)]
    other_ids = [h for h in filtered if h not in set(h_to_h_ids)]
    ordered = h_to_h_ids + other_ids
    kept_ids = ordered[:top_k]
    kept = {h: filtered[h] for h in kept_ids}

    stats = {
        "original_dep_entries": total,
        "after_degenerate_filter": len(filtered),
        "n_dropped_degenerate": n_dropped_degenerate,
        "h_to_h_entries": len(h_to_h_ids),
        "kept_dep_entries": len(kept),
        "top_k": top_k,
    }
    return kept, stats


def _engine_state_block_from_dep_map(dep_map: dict,
                                      mode: str = "full",
                                      top_k: int = 10) -> tuple[str, dict]:
    """Degraded engine block built directly from a cached dep map.

    No state summary is emitted because cached results don't persist
    the full pipe.engine object (hypothesis contents, statuses, etc.).
    Used only by the controlled ablation path (--cached-results) where
    the goal is to compare full vs trimmed dep-map injection while
    holding the source dep map constant.

    Returns (block_text, engine_stats).
    """
    if mode == "trimmed":
        deps_for_inject, trim_stats = _trim_dep_map(dep_map, top_k=top_k)
    else:
        deps_for_inject = dep_map
        trim_stats = {
            "original_dep_entries": len(dep_map),
            "kept_dep_entries": len(dep_map),
        }
    block = (
        "## Dependency map\n"
        f"{json.dumps(deps_for_inject, indent=2)}\n\n"
    )
    stats = {
        "mode": mode,
        "engine_block_chars": len(block),
        "engine_block_approx_tokens": _approx_tokens(block),
        "source": "cached_dep_map_no_state_summary",
        **trim_stats,
    }
    return block, stats


def _engine_state_block(pipe: EpistemicPipeline,
                        mode: str = "full",
                        top_k: int = 10) -> tuple[str, dict]:
    """Formatted engine state + dependency map for prompt injection.

    mode="full"         -> inject the LLM-authored state summary
                           (`pipe.get_state()`) PLUS the full dep map.
                           Reproduces prior runs bit-for-bit.
    mode="trimmed"      -> same shape as full (state summary + dep map)
                           but apply `_trim_dep_map(top_k=top_k)` to
                           the dep map before injection.
    mode="dep_map_only" -> drop the `## Epistemic model state` section
                           entirely; inject ONLY `## Dependency map` with
                           the full (untrimmed) dep map. Used for the
                           tipping-point length sweep, where the state
                           summary's verbosity competes with conversation
                           budget on long LoCoMo dialogues. Baseline
                           context and scoring unchanged.

    Returns (block_text, engine_stats). engine_stats includes the
    trim diagnostics plus the chars/tokens of the returned block so
    callers can report injection cost.
    """
    state = pipe.get_state()
    deps_full = pipe.get_dependency_map()

    if mode == "trimmed":
        deps_for_inject, trim_stats = _trim_dep_map(deps_full, top_k=top_k)
    else:
        deps_for_inject = deps_full
        trim_stats = {
            "original_dep_entries": len(deps_full),
            "kept_dep_entries": len(deps_full),
        }

    if mode == "dep_map_only":
        # No state summary section at all; just the dep map.
        block = (
            "## Dependency map\n"
            f"{json.dumps(deps_for_inject, indent=2)}\n\n"
        )
    else:
        block = (
            "## Epistemic model state\n"
            f"{state}\n\n"
            "## Dependency map\n"
            f"{json.dumps(deps_for_inject, indent=2)}\n\n"
        )
    stats = {
        "mode": mode,
        "engine_block_chars": len(block),
        "engine_block_approx_tokens": _approx_tokens(block),
        "includes_state_summary": (mode != "dep_map_only"),
        **trim_stats,
    }
    return block, stats


# ============================================================
# Official LoCoMo evaluation
# ============================================================

OFFICIAL_SYSTEM = (
    "You are answering factual questions about a multi-session conversation. "
    "Read the dialogue carefully and answer using information from it."
)


def evaluate_locomo_official(
    pipe: Optional[EpistemicPipeline],
    bench: dict,
    queries: list[dict],
    api_key: str,
    model_ctx_tokens: int,
    with_engine: bool,
    run_baseline: bool,
    engine_state_mode: str = "full",
    engine_state_top_k: int = 10,
    cached_dep_map: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    """Official LoCoMo-protocol evaluation (optionally with an engine
    augmentation for the hybrid path).

    - baseline_answer: LLM given ONLY the official `CONV_START_PROMPT` +
      pristine `DATE:/CONVERSATION:` context + official question.
    - hybrid_answer (only if with_engine): identical context, plus the
      ECM engine-state block appended between the conversation and the
      question. Both paths share the same truncated context so the
      comparison is apples-to-apples.

    engine_state_mode / engine_state_top_k control the ablation trimmer
    (see `_engine_state_block`). Budget for context truncation is ALWAYS
    reserved against the FULL engine block, so baseline ctx_text is
    bit-identical across full/trimmed runs; only the hybrid prompt's
    injected engine block differs.

    Returns (per_query_records, diag_stats). Per-query records include
    official_baseline_answer / official_baseline_f1 / official_baseline_em,
    and the hybrid counterparts when with_engine is True, plus the
    (category, type, ground_truth, evidence, mcq_key) metadata.
    """
    from pipeline import call_llm

    speakers = bench.get("speakers", ["", ""])
    conv_id = bench["id"]
    turns_by_session = bench.get("turns_by_session", [])
    start_prompt = CONV_START_PROMPT.format(
        speakers[0] or "Speaker A", speakers[1] or "Speaker B")

    # Engine-state block is appended to the context; factor it into the
    # token reserve so the safeguard remains honest. Always reserve the
    # FULL block's tokens so the conversation budget is unchanged across
    # ablation modes; only `engine_block_inject` differs by mode.
    if cached_dep_map is not None:
        # Controlled ablation: reuse a pre-built dep map, no pipe state.
        engine_block_full, _ = _engine_state_block_from_dep_map(
            cached_dep_map, mode="full")
        engine_block_inject, engine_stats = _engine_state_block_from_dep_map(
            cached_dep_map, mode=engine_state_mode, top_k=engine_state_top_k)
        engine_tokens = _approx_tokens(engine_block_full)
    elif with_engine and pipe is not None:
        engine_block_full, _ = _engine_state_block(pipe, mode="full")
        engine_block_inject, engine_stats = _engine_state_block(
            pipe, mode=engine_state_mode, top_k=engine_state_top_k)
        engine_tokens = _approx_tokens(engine_block_full)
    else:
        engine_block_full = ""
        engine_block_inject = ""
        engine_stats = {"mode": "disabled"}
        engine_tokens = 0
    reserve_base = 500  # instruction + answer headroom

    records: list[dict] = []
    skipped = {"cat5_missing_answer": 0}
    truncation_events = 0

    for q_idx, q in enumerate(queries):
        llm_question, mcq_key, skip_reason = _preprocess_question_official(
            q, conv_id, q_idx)
        if skip_reason is not None:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            records.append({
                "question": q["question"],
                "category": q.get("category"),
                "type": q.get("type"),
                "ground_truth": q.get("answer"),
                "adversarial_answer": q.get("adversarial_answer"),
                "evidence": q.get("evidence", []),
                "skipped_reason": skip_reason,
            })
            continue

        ctx_text, truncated, used, sessions_kept = _build_official_conversation(
            turns_by_session, llm_question,
            model_ctx_tokens=model_ctx_tokens,
            reserve_tokens=reserve_base + engine_tokens)
        if truncated:
            truncation_events += 1

        base_prompt = (
            f"{start_prompt}{ctx_text}"
            f"Question: {llm_question}\n"
            f"{OFFICIAL_ANSWER_INSTRUCTION}\n"
        )

        baseline_raw = call_llm(OFFICIAL_SYSTEM, base_prompt, api_key) if run_baseline else None
        baseline_answer = _postprocess_answer_official(baseline_raw, mcq_key) if run_baseline else None

        hybrid_raw = None
        hybrid_answer = None
        if with_engine:
            hybrid_prompt = (
                f"{start_prompt}{ctx_text}"
                f"{engine_block_inject}"
                f"Question: {llm_question}\n"
                f"{OFFICIAL_ANSWER_INSTRUCTION}\n"
            )
            hybrid_raw = call_llm(OFFICIAL_SYSTEM, hybrid_prompt, api_key)
            hybrid_answer = _postprocess_answer_official(hybrid_raw, mcq_key)

        gt_for_score = str(q.get("answer", ""))
        row = {
            "question": q["question"],
            "llm_question": llm_question,
            "category": q.get("category"),
            "type": q.get("type"),
            "ground_truth": gt_for_score,
            "evidence": q.get("evidence", []),
            "mcq_key": mcq_key,
            "context_truncated": truncated,
            "sessions_kept": sessions_kept,
        }
        if run_baseline:
            b = locomo_score(baseline_answer, gt_for_score, q.get("category"))
            row["official_baseline_raw"] = baseline_raw
            row["official_baseline_answer"] = baseline_answer
            row["official_baseline_f1"] = round(b["f1"], 4)
            row["official_baseline_em"] = b["em"]
        if with_engine:
            h = locomo_score(hybrid_answer, gt_for_score, q.get("category"))
            row["official_hybrid_raw"] = hybrid_raw
            row["official_hybrid_answer"] = hybrid_answer
            row["official_hybrid_f1"] = round(h["f1"], 4)
            row["official_hybrid_em"] = h["em"]
        records.append(row)

    diag = {"skipped": skipped, "truncation_events": truncation_events,
            "engine_state": engine_stats}
    return records, diag


# ============================================================
# Diagnostic ECM evaluation (not comparable to LoCoMo published numbers)
# ============================================================

DIAGNOSTIC_SYSTEM = (
    "You are answering questions about a conversation. The input includes "
    "the raw dialogue and, for the hybrid path, a structured epistemic "
    "model built from the conversation. Answer from the dialogue; use the "
    "epistemic model only to focus attention, not to fabricate facts."
)


def evaluate_locomo_diagnostic(
    pipe: EpistemicPipeline,
    bench: dict,
    queries: list[dict],
    api_key: str,
    model_ctx_tokens: int,
    run_baseline: bool,
) -> tuple[list[dict], dict]:
    """Diagnostic hybrid evaluation: ECM state + raw dialogue + timeline.

    Both the hybrid and the (optional) baseline paths see the same
    pristine dialogue rendered with a compact session timeline prefix;
    only the hybrid path additionally receives the engine state and
    dependency map.
    """
    from pipeline import call_llm

    timeline = _timeline_block(bench.get("session_dates"))
    turns_by_session = bench.get("turns_by_session", [])
    # Diagnostic path keeps the full engine block; the ablation flag
    # targets official mode only (per spec 2026-04-23).
    engine_block, _ = _engine_state_block(pipe, mode="full")
    engine_tokens = _approx_tokens(engine_block)
    reserve_base = 500

    conv_id = bench["id"]
    records: list[dict] = []
    skipped = {"cat5_missing_answer": 0}
    truncation_events = 0

    for q_idx, q in enumerate(queries):
        # Diagnostic mode still respects the cat-2 / cat-5 preprocessing
        # so that the comparison tracks what we'd score under LoCoMo.
        llm_question, mcq_key, skip_reason = _preprocess_question_official(
            q, conv_id, q_idx)
        if skip_reason is not None:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            records.append({
                "question": q["question"],
                "category": q.get("category"),
                "type": q.get("type"),
                "ground_truth": q.get("answer"),
                "adversarial_answer": q.get("adversarial_answer"),
                "evidence": q.get("evidence", []),
                "skipped_reason": skip_reason,
            })
            continue

        conv_text, truncated, _, sessions_kept = _build_official_conversation(
            turns_by_session, llm_question,
            model_ctx_tokens=model_ctx_tokens,
            reserve_tokens=reserve_base + engine_tokens + _approx_tokens(timeline))
        if truncated:
            truncation_events += 1

        baseline_raw = None
        baseline_answer = None
        if run_baseline:
            base_prompt = (
                f"{timeline}## Conversation\n{conv_text}"
                f"## Question\n{llm_question}\n"
                f"{OFFICIAL_ANSWER_INSTRUCTION}\n"
            )
            baseline_raw = call_llm(DIAGNOSTIC_SYSTEM, base_prompt, api_key)
            baseline_answer = _postprocess_answer_official(baseline_raw, mcq_key)

        hybrid_prompt = (
            f"{timeline}## Conversation\n{conv_text}"
            f"{engine_block}"
            f"## Question\n{llm_question}\n"
            f"{OFFICIAL_ANSWER_INSTRUCTION}\n"
        )
        hybrid_raw = call_llm(DIAGNOSTIC_SYSTEM, hybrid_prompt, api_key)
        hybrid_answer = _postprocess_answer_official(hybrid_raw, mcq_key)

        gt_for_score = str(q.get("answer", ""))

        # Engine-direct for dependency-style questions (ECM-specific
        # capability signal; not part of LoCoMo's scoring).
        engine_answer = None
        if "depend" in q["question"].lower() or "affect" in q["question"].lower():
            for hyp_id in pipe.engine.hypotheses:
                if hyp_id in q["question"]:
                    affected = pipe.engine.get_affected(hyp_id)
                    engine_answer = f"Affected({hyp_id}) = {affected}"
                    break

        row = {
            "question": q["question"],
            "llm_question": llm_question,
            "category": q.get("category"),
            "type": q.get("type"),
            "ground_truth": gt_for_score,
            "evidence": q.get("evidence", []),
            "mcq_key": mcq_key,
            "context_truncated": truncated,
            "sessions_kept": sessions_kept,
            "engine_answer": engine_answer,
            "diagnostic_hybrid_raw": hybrid_raw,
            "diagnostic_hybrid_answer": hybrid_answer,
        }
        h = locomo_score(hybrid_answer, gt_for_score, q.get("category"))
        row["diagnostic_hybrid_f1"] = round(h["f1"], 4)
        row["diagnostic_hybrid_em"] = h["em"]
        if run_baseline:
            b = locomo_score(baseline_answer, gt_for_score, q.get("category"))
            row["diagnostic_baseline_raw"] = baseline_raw
            row["diagnostic_baseline_answer"] = baseline_answer
            row["diagnostic_baseline_f1"] = round(b["f1"], 4)
            row["diagnostic_baseline_em"] = b["em"]
        records.append(row)

    diag = {"skipped": skipped, "truncation_events": truncation_events}
    return records, diag


# ============================================================
# Metric aggregation
# ============================================================

def compute_metrics(records: list[dict], answer_field: str,
                    *, include_engine: bool = False) -> dict:
    """Aggregate LoCoMo-style F1/EM over per-query records.

    `answer_field` is one of "official_baseline", "official_hybrid",
    "diagnostic_baseline", "diagnostic_hybrid". Each record is expected
    to carry <answer_field>_f1 and <answer_field>_em that were scored at
    evaluation time.

    When include_engine is False (baseline paths), engine_direct_answers
    and engine_coverage are not reported — they are ECM-specific signals
    and should not live on an LLM-only baseline metrics dict.
    """
    f1_key = f"{answer_field}_f1"
    em_key = f"{answer_field}_em"

    scored: list[dict] = []
    skipped = 0
    per_cat: dict = {}
    engine_answered = 0
    total_with_answer_field = 0

    for r in records:
        if r.get("skipped_reason") is not None:
            skipped += 1
            continue
        if f1_key not in r or em_key not in r:
            # This record wasn't scored under this answer_field (e.g.,
            # hybrid path disabled for a baseline-only run).
            continue
        total_with_answer_field += 1
        cat = r.get("category")
        entry = {"f1": r[f1_key], "em": r[em_key]}
        scored.append(entry)
        per_cat.setdefault(cat, []).append(entry)
        if include_engine and r.get("engine_answer"):
            engine_answered += 1

    def _mean(xs, k):
        return round(sum(x[k] for x in xs) / len(xs), 4) if xs else 0.0

    out = {
        "answer_field": answer_field,
        "n_scored": len(scored),
        "n_skipped": skipped,
        "f1": _mean(scored, "f1"),
        "em": _mean(scored, "em"),
        "by_category": {
            str(cat): {"n": len(xs), "f1": _mean(xs, "f1"), "em": _mean(xs, "em")}
            for cat, xs in sorted(per_cat.items(),
                                  key=lambda kv: (kv[0] is None, kv[0]))
        },
    }
    if include_engine:
        out["engine_direct_answers"] = engine_answered
        out["engine_coverage"] = (
            round(engine_answered / total_with_answer_field, 4)
            if total_with_answer_field else 0.0
        )
    return out


# ============================================================
# Non-LoCoMo eval kept minimal (unchanged semantics)
# ============================================================

def evaluate_generic_queries(pipe: EpistemicPipeline, queries: list[dict],
                             api_key: str) -> list[dict]:
    """Minimal hybrid eval for custom / MT-Bench-101 benchmarks.

    Keeps the pre-LoCoMo behavior: LLM receives the engine state + dep
    map and the question. No token-budget truncation (those benchmarks
    have short dialogues).
    """
    from pipeline import call_llm
    state = pipe.get_state()
    deps = pipe.get_dependency_map()
    results = []
    for q in queries:
        question = q.get("question", q.get("query", ""))
        gt = q.get("answer", q.get("expected", ""))
        engine_answer = None
        if "depend" in question.lower() or "affect" in question.lower():
            for hyp_id in pipe.engine.hypotheses:
                if hyp_id in question:
                    affected = pipe.engine.get_affected(hyp_id)
                    engine_answer = f"Affected({hyp_id}) = {affected}"
                    break
        prompt = (
            "Given the epistemic model state below, answer the question.\n\n"
            f"Model State:\n{state}\n\n"
            f"Dependencies:\n{json.dumps(deps, indent=2)}\n\n"
            f"Question: {question}\n\nAnswer concisely (2-3 sentences)."
        )
        answer = call_llm(
            "You are answering questions about a conversation using "
            "a structured epistemic model.",
            prompt, api_key)
        results.append({
            "question": question,
            "type": q.get("type", "unknown"),
            "ground_truth": gt,
            "engine_answer": engine_answer,
            "hybrid_answer": answer,
        })
    return results


# ============================================================
# ReviseQA — Phase 1 smoke (explicit_no_correction_no_reasoning)
# ------------------------------------------------------------
# Dataset source: https://github.com/ChadiHelwe/reviseqa
# Paper: openreview.net/forum?id=Z4KBiAYXlI (README L266).
# Sample schema: per reviseqa_data/nl/verified/*.json :
#   {original_context: list[str], original_context_fol: list[str],
#    conclusion: str, conclusion_fol: str,
#    answer: 'True'|'False'|'Uncertain',
#    reasoning_chain: [{conclusion, facts, rules}, ...],
#    edits: [{edit_number, modification_type, edited_context_fol,
#             edited_natural_language_context, edits_made, answer,
#             conclusion, conclusion_fol, prover9_input}, ...]
#   }  # edits is always length 7.
#
# Scope of this adapter (not the full official pipeline):
#  - Only the `explicit_no_correction_no_reasoning` task setting.
#  - tested model via pipeline.call_llm (Qwen-7B on local vLLM in smoke).
#  - Closed-form auto-scoring against GT True/False/Uncertain (no judge).
#  - Hybrid path injects dep-map-only block, no state summary
#    (LoCoMo lesson 2026-04-23).
#  - ECM state is maintained benchmark-natively: each fact/rule registered
#    as an engine observation, edits directly mutate engine state — no
#    LLM Interpreter in the loop yet (deferred until smoke shows signal).
# ============================================================


def _reviseqa_fol_key(fol_str: str) -> str:
    """Canonical key for a FOL string. Normalizes whitespace + preserves
    unicode logic operators; meant as a deterministic primary key for
    mapping between (fact/rule in edit.edits_made) and (engine obs id)."""
    return re.sub(r"\s+", " ", fol_str).strip()


def load_reviseqa(data_dir: str,
                  skip_truncated: bool = True,
                  max_scenarios: Optional[int] = None
                  ) -> list[dict]:
    """Load verified ReviseQA scenarios.

    data_dir: e.g. 'reviseqa_data/nl/verified'.  Reads ex_*.json (skipping
    *_truncated.json by default), returns the raw scenario schema verbatim
    (does NOT flatten into our speaker/text turns+queries format — ReviseQA
    is an edit-sequence benchmark, not a dialogue one). Each scenario gets
    id = filename stem (e.g., "ex_712").
    """
    out: list[dict] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"ReviseQA data_dir not found: {data_dir}")
    fnames = sorted(os.listdir(data_dir))
    for fname in fnames:
        if not fname.endswith(".json"):
            continue
        if skip_truncated and fname.endswith("_truncated.json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            d = json.load(f)
        # Required keys guard (fail-fast if schema drifts).
        for k in ("original_context", "conclusion", "answer", "edits"):
            if k not in d:
                print(f"[warn] skip {fname}: missing key {k}")
                break
        else:
            d["id"] = fname[:-5]  # strip .json
            out.append(d)
            if max_scenarios is not None and len(out) >= max_scenarios:
                break
    return out


# ---- ECM "benchmark-native structured update" state ----
# A lightweight side-state we maintain in parallel with EpistemicEngine:
#   fol_to_id: FOL canonical key -> observation id (f_<n> for fact, r_<n>
#              for rule); stable per scenario.
#   active:    set of currently-registered observation ids (shrinks on
#              removed_facts/rules, grows on added_facts/rules).
#   dep_h1:    set of observation ids declared as Dep(h1).
# We do NOT store in engine dictionaries beyond what symbolic_engine.py
# already supports.


def _reviseqa_register_premise(engine, fol_to_id: dict, active: set,
                               counter: dict, fol_str: str, nl_text: str,
                               kind: str) -> str:
    """Register one premise as an engine observation, unless already present.
    kind is 'fact' (prefix 'f') or 'rule' (prefix 'r').
    Returns the observation id used."""
    key = _reviseqa_fol_key(fol_str)
    if key in fol_to_id:
        # Duplicate FOL string (can happen if the same fact re-appears across
        # edits); re-activate without double-registering.
        oid = fol_to_id[key]
        active.add(oid)
        return oid
    prefix = "f" if kind == "fact" else "r"
    counter[prefix] = counter.get(prefix, 0) + 1
    oid = f"{prefix}_{counter[prefix]}"
    engine.observe(oid, nl_text, turn="reviseqa", speaker="dataset")
    fol_to_id[key] = oid
    active.add(oid)
    return oid


def _reviseqa_retract_premise(engine, fol_to_id: dict, active: set,
                              dep_h1: set, fol_str: str) -> tuple[bool, str]:
    """Actually remove a premise from engine state. symbolic_engine's
    retract_assumption() is non-destructive (only reports Affected), so we
    mutate the engine dicts directly here (benchmark-local behaviour; does
    NOT alter symbolic_engine.py).  Returns (found, oid)."""
    key = _reviseqa_fol_key(fol_str)
    if key not in fol_to_id:
        return False, ""
    oid = fol_to_id[key]
    # Drop from engine observation table if present
    if oid in getattr(engine, "observations", {}):
        del engine.observations[oid]
    # Drop from awareness
    if oid in getattr(engine, "awareness", set()):
        engine.awareness.discard(oid)
    # Drop from every dependency set
    for hyp_id, deps in list(getattr(engine, "dependencies", {}).items()):
        if oid in deps:
            deps.discard(oid)
    active.discard(oid)
    dep_h1.discard(oid)
    return True, oid


def _reviseqa_build_initial_dep(reasoning_chain: list, fol_to_id: dict) -> set:
    """Build Dep(h1) from the dataset's reasoning_chain. Walks every step,
    unions all {facts, rules} FOL strings used, looks them up in the
    already-populated fol_to_id, and returns the set of matched oids.
    Unmatched FOL strings (rare — reasoning_chain usually references the
    same strings as original_context) are reported separately."""
    wanted_fols: set = set()
    for step in reasoning_chain or []:
        for f in step.get("facts") or []:
            if isinstance(f, dict) and "fol" in f:
                wanted_fols.add(_reviseqa_fol_key(f["fol"]))
        for r in step.get("rules") or []:
            if isinstance(r, dict) and "fol" in r:
                wanted_fols.add(_reviseqa_fol_key(r["fol"]))
    dep_ids = {fol_to_id[k] for k in wanted_fols if k in fol_to_id}
    return dep_ids


def _reviseqa_engine_block_dep_only(engine, h1_id: str) -> str:
    """Dep-map-only block (per Phase 1 spec, no state summary).  Resolves
    each oid in Dep(h1) to its NL text from engine.observations when
    available; otherwise falls back to the bare id."""
    dep_ids = sorted(engine.dependencies.get(h1_id, set()))
    h1 = engine.hypotheses.get(h1_id)
    h1_content = h1.content if h1 else "<hypothesis>"
    lines = [
        "## Dependency map (ECM)",
        f"Hypothesis h1: \"{h1_content}\"",
        "Currently-tracked supporting premises (Dep(h1)):",
    ]
    if not dep_ids:
        lines.append("  (none)")
    else:
        for oid in dep_ids:
            nl = engine.observations[oid].content if oid in engine.observations else f"<retracted {oid}>"
            lines.append(f"  - {oid}: {nl}")
    return "\n".join(lines) + "\n"


def _reviseqa_apply_edit(engine, fol_to_id: dict, active: set, dep_h1: set,
                        counter: dict, edit: dict) -> dict:
    """Apply one edit's add/remove to engine state.  Updates Dep(h1) so
    that it tracks the currently-available subset of originally-depended
    premises PLUS any newly-added premises (conservative: new additions
    may participate in the conclusion's support)."""
    delta = edit.get("edits_made", {}) or {}
    diag = {"edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "n_removed_found": 0, "n_removed_missing": 0,
            "n_added_facts": 0, "n_added_rules": 0}

    # Removals first (so re-added FOLs, if any, can re-register fresh)
    for f in delta.get("removed_facts") or []:
        found, _ = _reviseqa_retract_premise(engine, fol_to_id, active, dep_h1, f["fol"])
        diag["n_removed_found"] += int(found)
        diag["n_removed_missing"] += int(not found)
    for r in delta.get("removed_rules") or []:
        found, _ = _reviseqa_retract_premise(engine, fol_to_id, active, dep_h1, r["fol"])
        diag["n_removed_found"] += int(found)
        diag["n_removed_missing"] += int(not found)

    # Additions
    for f in delta.get("added_facts") or []:
        oid = _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                          f["fol"], f["nl"], "fact")
        dep_h1.add(oid)
        diag["n_added_facts"] += 1
    for r in delta.get("added_rules") or []:
        oid = _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                          r["fol"], r["nl"], "rule")
        dep_h1.add(oid)
        diag["n_added_rules"] += 1

    # Re-write engine.dependencies[h1] so the dep-map-only block sees the
    # current active Dep(h1). We keep only ids still in `active`.
    current = {oid for oid in dep_h1 if oid in active}
    engine.dependencies["h1"] = current

    diag["dep_h1_size_after"] = len(current)
    diag["n_active_after"] = len(active)
    return diag


def _reviseqa_dep_map_stats_from_engine(engine, h1_id: str) -> dict:
    """Diagnostic snapshot of engine dep map + counts (LoCoMo/MT-Bench-101
    parity — h_auto/h_to_h/degenerate flags don't really apply here since
    we're seeding the engine ourselves, but we report them for consistency).
    """
    dep = {k: list(v) for k, v in engine.dependencies.items()}
    total = len(dep)
    h_auto = sum(1 for k in dep if isinstance(k, str) and k.startswith("h_auto_"))
    h_to_h = sum(1 for deps in dep.values()
                 if any(isinstance(d, str) and d.startswith("h") for d in deps))
    degenerate = sum(1 for k, deps in dep.items()
                     if isinstance(k, str) and k.startswith("h_auto_")
                     and len(deps) == 1
                     and isinstance(deps[0], str) and deps[0].startswith("o"))
    return {"n_entries": total, "h_auto": h_auto, "h_to_h": h_to_h,
            "degenerate": degenerate,
            "dep_h1_size": len(dep.get(h1_id, []))}


def _reviseqa_format_explicit_context(edit: dict) -> str:
    """Mirror ReviseQA evaluation.py explicit-context formatting (edit
    delta, not full context).  Used for both baseline and hybrid prompts."""
    delta = edit.get("edits_made", {}) or {}
    parts: list[str] = []
    if delta.get("removed_facts"):
        parts.append("Removed facts:\n"
                     + "\n".join(f"- {f['nl']}" for f in delta["removed_facts"]))
    if delta.get("removed_rules"):
        parts.append("Removed rules:\n"
                     + "\n".join(f"- {r['nl']}" for r in delta["removed_rules"]))
    if delta.get("added_rules"):
        parts.append("Added rules:\n"
                     + "\n".join(f"- {r['nl']}" for r in delta["added_rules"]))
    if delta.get("added_facts"):
        parts.append("Added facts:\n"
                     + "\n".join(f"- {f['nl']}" for f in delta["added_facts"]))
    if not parts:
        parts = ["(no explicit edits)"]
    return "\n\n".join(parts)


REVISEQA_SYSTEM = (
    "When you reply, output *only* a JSON object with exactly "
    "two fields:\n"
    "  - reasoning  (a string; may be empty)\n"
    "  - answer     (one of 'True','False','Uncertain')\n"
    "Do not wrap it in markdown, do not say anything else."
)

REVISEQA_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Options:\nA) True\nB) False\nC) Uncertain\n\n"
)


def _reviseqa_parse_answer(raw: Optional[str]) -> Optional[str]:
    """Extract 'True'/'False'/'Uncertain' from the model's response.
    Tries JSON parse first, then regex fallback."""
    if raw is None:
        return None
    s = raw.strip()
    # Option letters map to the prompt's fixed option list (A) True B) False
    # C) Uncertain). Reasoning-tuned models (DeepSeek-R1 distills) answer
    # with the letter; no published run ever did (0/19,530 responses), so
    # this is a pure superset of the original parser.
    LETTER = {"A": "True", "B": "False", "C": "Uncertain"}

    def _norm_ans(a):
        a = a.strip()
        if a.upper() in LETTER:
            return LETTER[a.upper()]
        m = re.match(r"^([ABC])\)\s*(True|False|Uncertain)$", a, flags=re.I)
        if m:
            return m.group(2).capitalize()
        return a.capitalize()

    # JSON parse first (strips possible ```json fences)
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    try:
        d = json.loads(s2)
        a = d.get("answer")
        if isinstance(a, str) and _norm_ans(a) in {"True", "False", "Uncertain"}:
            return _norm_ans(a)
    except Exception:
        pass
    # Regex fallback for unstructured output (also finds JSON embedded in
    # prose or inside a fenced block that is not at the string start)
    m = re.search(r'"answer"\s*:\s*"([^"]{1,24})"', s)
    if m:
        cand = _norm_ans(m.group(1))
        if cand in {"True", "False", "Uncertain"}:
            return cand
    # Last-ditch: bare token match
    for tok in ["Uncertain", "True", "False"]:
        if re.search(rf'\b{tok}\b', s):
            return tok
    return None


def _reviseqa_chat_call(messages: list, api_key: str,
                        url: str,
                        model: str,
                        temperature: float = 0,
                        max_tokens: Optional[int] = None,
                        max_retries: int = 3) -> Optional[str]:
    """Multi-turn chat completion for ReviseQA cumulative conversation.
    Separate from pipeline.call_llm (which is 2-message only) because the
    ReviseQA protocol requires a growing dialogue history per chain.

    max_tokens defaults to 800 (the value used by every published run);
    REVISEQA_MAX_TOKENS overrides it for QA models that spend tokens on
    hidden reasoning before the JSON answer (e.g. DeepSeek-R1 distills
    served with vLLM's reasoning parser)."""
    if max_tokens is None:
        max_tokens = int(os.environ.get("REVISEQA_MAX_TOKENS", "800"))
    try:
        import requests
    except ImportError:
        return None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens, "messages": messages},
                timeout=120)
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            print(f"  [reviseqa] API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}")
        except Exception as e:
            print(f"  [reviseqa] Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def _reviseqa_init_scenario_state(scenario: dict) -> tuple:
    """Fresh EpistemicPipeline; seed engine with original_context facts and
    any rules mined from reasoning_chain; register h1 as the conclusion
    hypothesis with Dep(h1) from reasoning_chain. No LLM calls.

    Returns (pipe, fol_to_id, active, dep_h1, counter, init_diag).
    """
    pipe = EpistemicPipeline(api_key="unused-for-structured-only",
                             verbose=False, max_history_turns=0)
    engine = pipe.engine
    fol_to_id: dict = {}
    active: set = set()
    dep_h1: set = set()
    counter: dict = {"f": 0, "r": 0}

    # 1. original_context facts: original_context_fol lines (positional
    #    match with original_context NL lines)
    ctx_nl = scenario.get("original_context", []) or []
    ctx_fol = scenario.get("original_context_fol", []) or []
    if len(ctx_nl) != len(ctx_fol):
        # partial match: use min
        pass
    n_ctx = min(len(ctx_nl), len(ctx_fol))
    for i in range(n_ctx):
        _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                    ctx_fol[i], ctx_nl[i], "fact")

    # 2. rules found in reasoning_chain that are NOT already in fol_to_id
    #    (original_context often contains facts only; rules are mined
    #    from reasoning_chain)
    for step in scenario.get("reasoning_chain") or []:
        for r in step.get("rules") or []:
            if isinstance(r, dict) and "fol" in r and "text" in r:
                _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                            r["fol"], r["text"], "rule")
        for f in step.get("facts") or []:
            if isinstance(f, dict) and "fol" in f and "text" in f:
                if _reviseqa_fol_key(f["fol"]) not in fol_to_id:
                    _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                                f["fol"], f["text"], "fact")

    # 3. hypothesize h1 with dep from reasoning_chain
    init_dep = _reviseqa_build_initial_dep(
        scenario.get("reasoning_chain") or [], fol_to_id)
    dep_h1 = set(init_dep)
    engine.hypothesize(
        hyp_id="h1",
        content=scenario.get("conclusion", ""),
        turn="reviseqa",
        speaker="dataset",
        explains=[],
        depends_on=list(dep_h1),
    )

    init_diag = {
        "n_ctx_facts_registered": n_ctx,
        "n_total_registered": counter["f"] + counter["r"],
        "n_facts_registered": counter["f"],
        "n_rules_registered": counter["r"],
        "dep_h1_size_initial": len(dep_h1),
        "dep_h1_coverage": (
            round(len(dep_h1) / len(active), 3) if active else 0.0
        ),
    }
    return pipe, fol_to_id, active, dep_h1, counter, init_diag


def evaluate_reviseqa_scenario(
        scenario: dict,
        tested_api_key: str,
        base_url: str,
        model: str,
        include_reasoning: bool,
        include_correction: bool,
        run_baseline: bool,
        run_hybrid: bool,
        ) -> dict:
    """Run one scenario chain (1 demo + 7 edits) under the
    explicit_no_correction_no_reasoning setting.

    - `include_reasoning`: controls whether the 0-shot demo's assistant
      message includes a `reasoning` text (set to False for the no_reasoning
      variant).
    - `include_correction`: if True, a CORRECTION line is appended after
      a wrong answer (set to False for no_correction).
    """
    scenario_id = scenario["id"]
    edits = scenario.get("edits", [])
    orig_ctx_text = "\n".join(scenario.get("original_context", []) or [])
    conclusion = scenario.get("conclusion", "")
    orig_answer = scenario.get("answer", "Uncertain")
    base_question = f"Does the context entail the conclusion '{conclusion}'?"

    demo_reasoning = ""  # no_reasoning: demo carries an empty reasoning
    demo_assistant = json.dumps(
        {"reasoning": demo_reasoning if include_reasoning else "",
         "answer": orig_answer})

    baseline_messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_question)},
        {"role": "assistant", "content": demo_assistant},
    ]
    hybrid_messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_question)},
        {"role": "assistant", "content": demo_assistant},
    ]

    # Engine state for hybrid.
    if run_hybrid:
        pipe, fol_to_id, active, dep_h1, counter, init_diag = \
            _reviseqa_init_scenario_state(scenario)
        engine = pipe.engine
    else:
        pipe = engine = fol_to_id = active = dep_h1 = counter = None
        init_diag = {}

    per_step: list[dict] = []
    baseline_trace: list[bool] = []
    hybrid_trace: list[bool] = []

    for step_idx, edit in enumerate(edits, start=1):
        gt = edit.get("answer", "")
        step_conclusion = edit.get("conclusion", conclusion)
        step_question = f"Does the context entail the conclusion '{step_conclusion}'?"
        delta_ctx = _reviseqa_format_explicit_context(edit)
        user_msg_plain = REVISEQA_PROMPT_TEMPLATE.format(
            context=delta_ctx, question=step_question)

        # ---- baseline ----
        base_pred = None
        base_raw = None
        if run_baseline:
            baseline_messages.append({"role": "user", "content": user_msg_plain})
            base_raw = _reviseqa_chat_call(baseline_messages, tested_api_key,
                                           base_url, model)
            baseline_messages.append({"role": "assistant",
                                       "content": base_raw or ""})
            base_pred = _reviseqa_parse_answer(base_raw)
            base_correct = (base_pred == gt)
            baseline_trace.append(bool(base_correct))
            if include_correction and not base_correct:
                baseline_messages.append({
                    "role": "user",
                    "content": f"You made a mistake, the correct answer was: "
                               f"{gt}. Now answer the next problem."})

        # ---- hybrid (apply edit into engine BEFORE generating prompt) ----
        hyb_pred = None
        hyb_raw = None
        apply_diag = None
        dep_stats = None
        if run_hybrid:
            apply_diag = _reviseqa_apply_edit(
                engine, fol_to_id, active, dep_h1, counter, edit)
            dep_stats = _reviseqa_dep_map_stats_from_engine(engine, "h1")
            engine_block = _reviseqa_engine_block_dep_only(engine, "h1")
            user_msg_hybrid = (
                engine_block + "\n" + REVISEQA_PROMPT_TEMPLATE.format(
                    context=delta_ctx, question=step_question))
            hybrid_messages.append({"role": "user", "content": user_msg_hybrid})
            hyb_raw = _reviseqa_chat_call(hybrid_messages, tested_api_key,
                                          base_url, model)
            hybrid_messages.append({"role": "assistant",
                                     "content": hyb_raw or ""})
            hyb_pred = _reviseqa_parse_answer(hyb_raw)
            hyb_correct = (hyb_pred == gt)
            hybrid_trace.append(bool(hyb_correct))
            if include_correction and not hyb_correct:
                hybrid_messages.append({
                    "role": "user",
                    "content": f"You made a mistake, the correct answer was: "
                               f"{gt}. Now answer the next problem."})

        per_step.append({
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "gt": gt,
            "baseline": None if not run_baseline else {
                "raw": base_raw, "parsed": base_pred,
                "correct": base_pred == gt,
            },
            "hybrid_dep_only": None if not run_hybrid else {
                "raw": hyb_raw, "parsed": hyb_pred,
                "correct": hyb_pred == gt,
                "apply_diag": apply_diag,
                "dep_stats": dep_stats,
            },
        })

    # LCATA traces are per-step bool lists (length = n edits = 7).
    def _lcata_ks(trace):
        return {
            "k2": int(all(trace[:2])) if len(trace) >= 2 else 0,
            "k4": int(all(trace[:4])) if len(trace) >= 4 else 0,
            "k7": int(all(trace[:7])) if len(trace) >= 7 else 0,
        }

    final_stats_hyb = _reviseqa_dep_map_stats_from_engine(engine, "h1") if engine else {}
    return {
        "scenario_id": scenario_id,
        "task_setting": "explicit_no_correction_no_reasoning",
        "n_edits": len(edits),
        "modification_types": [e.get("modification_type") for e in edits],
        "init_diag": init_diag,
        "baseline_trace": baseline_trace,
        "hybrid_dep_only_trace": hybrid_trace,
        "baseline_lcata": _lcata_ks(baseline_trace) if run_baseline else {},
        "hybrid_dep_only_lcata": _lcata_ks(hybrid_trace) if run_hybrid else {},
        "final_dep_stats_hybrid": final_stats_hyb,
        "per_step": per_step,
    }


def _reviseqa_aggregate_lcata(records: list[dict], key: str) -> dict:
    """LCATA@k = fraction of chains with all-correct in first k edits."""
    totals = {"k2": [], "k4": [], "k7": []}
    for r in records:
        lc = r.get(key, {}) or {}
        for k in totals:
            if k in lc:
                totals[k].append(lc[k])
    return {k: (round(sum(v) / len(v), 4) if v else None,
               len(v)) for k, v in totals.items()}


def _reviseqa_preflight_check(records: list[dict]) -> tuple[bool, list[str]]:
    """Gate preflight to full smoke. Conditions:
      - Every scenario has dep_h1_size_initial >= 2
      - Across all edits, the removal-miss rate is < 25%
      - Final Dep(h1) stays non-empty on at least half the scenarios
    Returns (passed, list_of_issues)."""
    issues = []
    small_init = [r["scenario_id"] for r in records
                  if r.get("init_diag", {}).get("dep_h1_size_initial", 0) < 2]
    if small_init:
        issues.append(f"Dep(h1) initial size < 2 on: {small_init}")

    tot_found = tot_missing = 0
    for r in records:
        for step in r["per_step"]:
            d = (step.get("hybrid_dep_only") or {}).get("apply_diag") or {}
            tot_found += d.get("n_removed_found", 0)
            tot_missing += d.get("n_removed_missing", 0)
    miss_rate = (tot_missing / (tot_found + tot_missing)
                 if (tot_found + tot_missing) else 0.0)
    if miss_rate >= 0.25:
        issues.append(f"Removal miss rate {miss_rate:.2%} >= 25% "
                      f"(found={tot_found} missing={tot_missing})")

    empty_final = sum(
        1 for r in records
        if r.get("final_dep_stats_hybrid", {}).get("dep_h1_size", 0) == 0)
    if empty_final > len(records) / 2:
        issues.append(f"Final Dep(h1) empty on {empty_final}/{len(records)} "
                      "scenarios (engine became semantically vacuous).")

    return (len(issues) == 0, issues)


def _reviseqa_atomic_save(out_path: str, data: dict) -> None:
    """Write JSON via temp+rename for crash-safety during long runs.
    If the process dies mid-write, the original file is untouched.
    """
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, out_path)


def _reviseqa_load_checkpoint(out_path: str) -> Optional[dict]:
    """Load an in-progress / completed ReviseQA output for resume.
    Returns None if file does not exist. Raises on malformed JSON."""
    if not os.path.isfile(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


def _reviseqa_recompute_aggregates(records: list[dict]) -> dict:
    """Recompute aggregate_baseline/hybrid_dep_only_lcata + verdict from
    the full `records` list. Used both in-run (for live progress peek)
    and at completion. Returns a dict ready to merge into the output."""
    baseline_lcata = _reviseqa_aggregate_lcata(records, "baseline_lcata")
    hybrid_lcata = _reviseqa_aggregate_lcata(records, "hybrid_dep_only_lcata")

    def _delta(b, h):
        if b[0] is None or h[0] is None:
            return None
        return round(h[0] - b[0], 4)

    go_signals = []
    nogo_signals = []
    for k in ("k2", "k4", "k7"):
        b = baseline_lcata[k]; h = hybrid_lcata[k]
        if b[0] is None or h[0] is None:
            continue
        d = h[0] - b[0]
        if d >= 0.05:
            go_signals.append(f"{k}: +{d:.3f}")
        if d <= -0.05:
            nogo_signals.append(f"{k}: {d:.3f}")

    if go_signals and not nogo_signals:
        verdict = "GO candidate"
    elif not go_signals and not nogo_signals:
        verdict = "FLAT (no go)"
    elif nogo_signals:
        verdict = "STOP (hybrid drop detected)"
    else:
        verdict = "MIXED"

    return {
        "aggregate_baseline_lcata": baseline_lcata,
        "aggregate_hybrid_dep_only_lcata": hybrid_lcata,
        "verdict": verdict,
        "go_signals": go_signals,
        "nogo_signals": nogo_signals,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run epistemic pipeline on benchmarks")
    parser.add_argument("--benchmark", type=str, default="custom",
        choices=["locomo", "mt-bench-101", "reviseqa", "custom"],
        help="Benchmark format")
    parser.add_argument("--data", type=str, required=True,
        help="Path to benchmark data JSON")
    parser.add_argument("--max-conversations", type=int, default=5,
        help="Maximum conversations to process")
    parser.add_argument("--run-baseline", action="store_true",
        help="Run an LLM-only baseline alongside the ECM-augmented path.")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
        help="Output file for results")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic",
        help="API backend (forwarded to pipeline).")
    parser.add_argument("--model", type=str, default=None,
        help="Override model name")
    parser.add_argument("--base-url", type=str, default=None,
        help="Override API URL")
    parser.add_argument("--max-history-turns", type=int, default=20,
        help="Turns of raw history passed to the LLM Interpreter per "
             "pipeline step. Engine state carries long-term memory. "
             "Default 20 fits 16K ctx on 400+ turn LoCoMo convs.")
    parser.add_argument("--max-queries", type=int, default=None,
        help="Cap QA queries per conversation (smoke tests).")
    parser.add_argument("--locomo-mode", choices=["official", "diagnostic"],
        default="official",
        help="LoCoMo evaluation mode. 'official' aligns with the published "
             "LoCoMo QA protocol (CONV_START_PROMPT, DATE:/CONVERSATION: "
             "blocks, cat-2/cat-5 preprocessing, token-budget truncation). "
             "'diagnostic' uses ECM-natural prompts — not a comparable "
             "LoCoMo baseline.")
    parser.add_argument("--with-engine", action="store_true",
        help="Official mode only: also produce an ECM hybrid answer "
             "(conversation + engine state) alongside the LLM-only "
             "baseline. No effect in diagnostic mode (hybrid always on).")
    parser.add_argument("--model-ctx-tokens", type=int, default=16384,
        help="Model context window in tokens. Used for conservative "
             "context truncation. Default 16384 matches the vLLM Qwen-7B "
             "launch setting.")
    parser.add_argument("--locomo-max-turns", type=int, default=None,
        help="LoCoMo only: before evaluation, pre-slice each "
             "conversation to at most N turns chronologically (oldest "
             "sessions first, stop when cumulative turn count reaches "
             "N). Used for the tipping-point length-sweep experiment. "
             "Default None = no cap (use full conversation).")
    parser.add_argument("--engine-state-mode",
        choices=["full", "trimmed", "dep_map_only"],
        default="full",
        help="Ablation / tipping-point flag for the engine-state block "
             "injected into the hybrid prompt. 'full' reproduces prior "
             "runs bit-for-bit (state summary + full dep map). 'trimmed' "
             "keeps the state summary but drops degenerate h_auto_* "
             "single-observation deps and caps the dep map at "
             "--engine-state-top-k (h->h entries first). 'dep_map_only' "
             "drops the `## Epistemic model state` section entirely and "
             "injects only `## Dependency map` with the full dep map — "
             "use this for the LoCoMo tipping-point length sweep where "
             "state-summary verbosity competes with conversation budget. "
             "Baseline context and scoring unchanged across all modes.")
    parser.add_argument("--engine-state-top-k", type=int, default=10,
        help="In 'trimmed' mode, cap the injected dep map at this many "
             "entries (h->h first). No effect in 'full' mode.")
    parser.add_argument("--cached-results", type=str, default=None,
        help="Path to a previously-saved results JSON (e.g. "
             "experiments/results/locomo/official/"
             "locomo_official_3x20.json). When set, the pipeline "
             "ingest step is skipped; each conversation's cached "
             "`dependency_map` is reused as the sole source for the "
             "hybrid engine block. Only the QA-call part re-runs. "
             "Intended for controlled ablation of --engine-state-mode. "
             "The cached engine block is degraded to just "
             "`## Dependency map` (no state summary) since state is "
             "not persisted in cached results. --run-baseline defaults "
             "to False in this mode since baseline F1 is already in "
             "the cached file.")
    # MT-Bench-101 smoke config (Phase 1).
    parser.add_argument("--mtbench101-tasks", type=str,
        default="SC,SA,CM,AR,TS",
        help="Comma-separated MT-Bench-101 task codes to evaluate. "
             "Smoke default is the 5 tasks most aligned with ECM's "
             "belief-revision / memory / dependency claim.")
    parser.add_argument("--mtbench101-per-task", type=int, default=4,
        help="Number of dialogues per task to evaluate (deterministic "
             "selection: first N in the JSONL order that match the task "
             "code). Smoke default is 4.")
    parser.add_argument("--mtbench101-judge-model", type=str,
        default="gpt-4-turbo",
        help="Judge model (OpenAI API). Official MT-Bench-101 config "
             "uses gpt-4-1106-preview (see sefira/opencompass); "
             "gpt-4-turbo is the direct-successor alias and is used "
             "here because the 1106 snapshot is not accessible on "
             "this account. Both are GPT-4 Turbo family.")
    parser.add_argument("--mtbench101-judge-temperature", type=float,
        default=0.8,
        help="Judge temperature. Official value is 0.8 (see "
             "sefira/opencompass eval_subjective_mtbench101.py).")
    parser.add_argument("--mtbench101-judge-api-key-env", type=str,
        default="OPENAI_API_KEY",
        help="Environment variable holding the OpenAI API key used "
             "for judge calls. Separate from the tested-model api_key "
             "(which may be a dummy value for local vLLM).")
    # ReviseQA Phase-1 smoke config.
    parser.add_argument("--reviseqa-task-setting", type=str,
        default="explicit_no_correction_no_reasoning",
        help="ReviseQA evaluation task setting. Phase-1 smoke default is "
             "`explicit_no_correction_no_reasoning` (the one this adapter "
             "currently supports).")
    parser.add_argument("--reviseqa-max-scenarios", type=int, default=20,
        help="Total number of verified scenarios to evaluate in the smoke "
             "(preflight + full smoke). Default 20.")
    parser.add_argument("--reviseqa-preflight-n", type=int, default=3,
        help="Number of scenarios to run as a preflight gate before the "
             "full smoke. Default 3.")
    parser.add_argument("--reviseqa-force-restart", action="store_true",
        help="Ignore any existing ReviseQA checkpoint at --output and "
             "restart from scratch. Default (unset): auto-resume from "
             "the checkpoint if the file exists and is non-empty.")
    args = parser.parse_args()

    # Forward backend settings to the pipeline module globals.
    pipeline.BACKEND = args.backend
    if args.model:
        pipeline.MODEL = args.model
    if args.base_url:
        pipeline.API_URL = args.base_url
    elif args.backend == "openai":
        pipeline.API_URL = "http://localhost:8000/v1/chat/completions"

    if args.backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: set ANTHROPIC_API_KEY"); sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    # ---- ReviseQA Phase-1 smoke (benchmark-native structured update) ----
    if args.benchmark == "reviseqa":
        if args.reviseqa_task_setting != "explicit_no_correction_no_reasoning":
            print(f"Error: only 'explicit_no_correction_no_reasoning' is "
                  f"supported in this Phase-1 adapter "
                  f"(got {args.reviseqa_task_setting!r}).")
            sys.exit(1)
        scenarios = load_reviseqa(args.data,
                                   max_scenarios=args.reviseqa_max_scenarios)
        if not scenarios:
            print("Error: load_reviseqa returned no scenarios; check --data.")
            sys.exit(1)
        preflight_n = min(args.reviseqa_preflight_n, len(scenarios))
        include_reasoning = False   # _no_reasoning
        include_correction = False  # _no_correction

        # ---- Resume: load checkpoint if present and not --force-restart ----
        checkpoint = None
        if not args.reviseqa_force_restart:
            try:
                checkpoint = _reviseqa_load_checkpoint(args.output)
            except json.JSONDecodeError as e:
                print(f"[warn] existing {args.output} is malformed ({e}); "
                      "treating as a fresh run.")
                checkpoint = None

        if checkpoint is not None and checkpoint.get("status") == "preflight_failed":
            print(f"Existing checkpoint at {args.output} records a "
                  "failed preflight. Delete it or pass "
                  "--reviseqa-force-restart to re-run.")
            sys.exit(1)

        all_records: list[dict] = (checkpoint or {}).get("records", []) or []
        completed_ids: set = {r["scenario_id"] for r in all_records}
        resumed = bool(completed_ids)

        print(f"Loaded {len(scenarios)} verified scenarios from {args.data}")
        print(f"Tested model: backend={args.backend} model={pipeline.MODEL} "
              f"url={pipeline.API_URL}")
        print(f"Task setting: {args.reviseqa_task_setting} "
              f"(preflight_n={preflight_n}, total={len(scenarios)})")
        if resumed:
            print(f"[resume] checkpoint found at {args.output}: "
                  f"{len(completed_ids)} scenarios already complete; "
                  f"will skip them and continue from where we left off.")
        else:
            print(f"[fresh run] no checkpoint at {args.output}")

        def _checkpoint_config() -> dict:
            return {
                "tested_backend": args.backend,
                "tested_model": pipeline.MODEL,
                "tested_url": pipeline.API_URL,
                "data_dir": args.data,
                "include_reasoning": include_reasoning,
                "include_correction": include_correction,
                "max_scenarios": args.reviseqa_max_scenarios,
                "engine_block_mode": "dep_map_only_no_state_summary",
                "structured_update_source": "benchmark_native",
            }

        def _save_checkpoint(status: str, extra_fields: Optional[dict] = None):
            """Atomic save. status = in_progress | preflight_failed |
            completed. `extra_fields` merges into the output root."""
            agg = _reviseqa_recompute_aggregates(all_records) \
                if all_records else {
                    "aggregate_baseline_lcata": {},
                    "aggregate_hybrid_dep_only_lcata": {},
                    "verdict": None,
                    "go_signals": [],
                    "nogo_signals": [],
                }
            data = {
                "status": status,
                "task_setting": args.reviseqa_task_setting,
                "preflight_n": preflight_n,
                "preflight_issues": [],
                "completed_scenario_ids": [r["scenario_id"] for r in all_records],
                "n_completed": len(all_records),
                "n_total_target": len(scenarios),
                **agg,
                "records": all_records,
                "config": _checkpoint_config(),
            }
            if extra_fields:
                data.update(extra_fields)
            _reviseqa_atomic_save(args.output, data)

        # ---- Preflight (only if not already past it) ----
        preflight_already_done = (len(completed_ids) >= preflight_n) or \
                                 (checkpoint is not None and
                                  checkpoint.get("status") == "preflight_passed")
        if not preflight_already_done:
            print(f"\n{'='*60}\n[preflight] evaluating first {preflight_n} "
                  f"scenarios to validate FOL mapper & engine semantics\n"
                  f"{'='*60}")
            for i, scenario in enumerate(scenarios[:preflight_n]):
                if scenario["id"] in completed_ids:
                    continue
                print(f"  [pre {i+1}/{preflight_n}] {scenario['id']}", flush=True)
                rec = evaluate_reviseqa_scenario(
                    scenario, tested_api_key=api_key,
                    base_url=pipeline.API_URL, model=pipeline.MODEL,
                    include_reasoning=include_reasoning,
                    include_correction=include_correction,
                    run_baseline=True, run_hybrid=True)
                all_records.append(rec)
                completed_ids.add(scenario["id"])
                init = rec["init_diag"]
                finalst = rec["final_dep_stats_hybrid"]
                print(f"    init: facts={init['n_facts_registered']} "
                      f"rules={init['n_rules_registered']} "
                      f"Dep(h1)_init={init['dep_h1_size_initial']} "
                      f"coverage={init['dep_h1_coverage']}")
                print(f"    final: Dep(h1)={finalst.get('dep_h1_size',0)} "
                      f"active_entries={finalst.get('n_entries',0)}")
                print(f"    baseline_trace: {rec['baseline_trace']} "
                      f"LCATA={rec['baseline_lcata']}")
                print(f"    hybrid_trace:   {rec['hybrid_dep_only_trace']} "
                      f"LCATA={rec['hybrid_dep_only_lcata']}")
                _save_checkpoint("in_progress")

            passed, issues = _reviseqa_preflight_check(
                all_records[:preflight_n])
            if not passed:
                print(f"\n{'='*60}\n[preflight FAILED] Reporting issues:")
                for it in issues:
                    print(f"  - {it}")
                print(f"{'='*60}")
                _save_checkpoint("preflight_failed",
                                 extra_fields={"preflight_issues": issues})
                print(f"\nPartial results saved to {args.output}")
                return
            print(f"\n[preflight PASSED]")
            _save_checkpoint("in_progress")

        # ---- Full smoke on remaining ----
        remaining = [s for s in scenarios if s["id"] not in completed_ids]
        if remaining:
            print(f"\n{'='*60}\n[smoke] evaluating the remaining "
                  f"{len(remaining)} scenarios\n{'='*60}")
            for j, scenario in enumerate(remaining):
                idx_display = len(scenarios) - len(remaining) + j + 1
                print(f"  [{idx_display}/{len(scenarios)}] "
                      f"{scenario['id']}", flush=True)
                rec = evaluate_reviseqa_scenario(
                    scenario, tested_api_key=api_key,
                    base_url=pipeline.API_URL, model=pipeline.MODEL,
                    include_reasoning=include_reasoning,
                    include_correction=include_correction,
                    run_baseline=True, run_hybrid=True)
                all_records.append(rec)
                completed_ids.add(scenario["id"])
                init = rec["init_diag"]
                print(f"    Dep(h1)_init={init['dep_h1_size_initial']} "
                      f"final={rec['final_dep_stats_hybrid'].get('dep_h1_size',0)} "
                      f"base_LCATA={rec['baseline_lcata']} "
                      f"hyb_LCATA={rec['hybrid_dep_only_lcata']}")
                _save_checkpoint("in_progress")

        # ---- Final aggregate + summary + save ----
        agg = _reviseqa_recompute_aggregates(all_records)
        baseline_lcata = agg["aggregate_baseline_lcata"]
        hybrid_lcata = agg["aggregate_hybrid_dep_only_lcata"]
        verdict = agg["verdict"]
        go_signals = agg["go_signals"]
        nogo_signals = agg["nogo_signals"]

        def _delta(b, h):
            if b[0] is None or h[0] is None:
                return None
            return round(h[0] - b[0], 4)

        print(f"\n{'='*60}\nFULL RUN SUMMARY ({len(all_records)} scenarios)"
              f"\n{'='*60}")
        print(f"  {'metric':<10} {'baseline':<18} {'hybrid_dep_only':<20} {'delta':<8}")
        for k in ("k2", "k4", "k7"):
            b = baseline_lcata[k]; h = hybrid_lcata[k]
            print(f"  {k:<10} {str(b):<18} {str(h):<20} {str(_delta(b,h)):<8}")

        print(f"\n  Verdict: {verdict}")
        if go_signals: print(f"    positive: {go_signals}")
        if nogo_signals: print(f"    negative: {nogo_signals}")

        _save_checkpoint("completed")
        print(f"\nResults saved to {args.output}")
        return
    # ---- End ReviseQA smoke ----

    loaders = {
        "locomo": load_locomo,
        "mt-bench-101": load_mt_bench_101,
        "custom": load_custom_json,
    }
    benchmarks = loaders[args.benchmark](args.data)
    print(f"Loaded {len(benchmarks)} conversations from {args.data}")

    # Tipping-point length-sweep pre-slice (LoCoMo only).
    if args.benchmark == "locomo" and args.locomo_max_turns is not None:
        if args.locomo_max_turns <= 0:
            print(f"Error: --locomo-max-turns must be positive "
                  f"(got {args.locomo_max_turns}).")
            sys.exit(1)
        print(f"[tipping-point] truncating each LoCoMo conversation to "
              f"first {args.locomo_max_turns} turns chronologically "
              f"(oldest sessions first)")
        _locomo_truncate_to_n_turns(benchmarks, args.locomo_max_turns)
        for b in benchmarks:
            print(f"  {b['id']}: {b['n_turns_after_truncation']} turns "
                  f"across {len(b.get('turns_by_session') or [])} sessions")

    # ---- MT-Bench-101 Phase-1 smoke (official per-turn protocol) ----
    if args.benchmark == "mt-bench-101":
        task_filter = [t.strip() for t in args.mtbench101_tasks.split(",") if t.strip()]
        judge_api_key = os.environ.get(args.mtbench101_judge_api_key_env, "")
        if not judge_api_key:
            print(f"Error: set {args.mtbench101_judge_api_key_env} env var "
                  "with an OpenAI API key for the GPT-4 judge.")
            sys.exit(1)
        # Deterministic per-task selection: first N dialogues per task
        # in JSONL order.
        selected: list[dict] = []
        seen_per_task: dict = {t: 0 for t in task_filter}
        for bench in benchmarks:
            t = bench["task"]
            if t in seen_per_task and seen_per_task[t] < args.mtbench101_per_task:
                selected.append(bench)
                seen_per_task[t] += 1
        missing = [t for t, c in seen_per_task.items() if c < args.mtbench101_per_task]
        if missing:
            print(f"[warn] tasks under-filled (fewer than {args.mtbench101_per_task} "
                  f"dialogues available): {missing}")
        print(f"Selected {len(selected)} dialogues across tasks "
              f"{sorted(seen_per_task.keys())}: "
              f"{[(t,c) for t,c in sorted(seen_per_task.items())]}")
        print(f"Tested model: backend={args.backend} model={pipeline.MODEL} "
              f"url={pipeline.API_URL}")
        print(f"Judge model: {args.mtbench101_judge_model} "
              f"(temp={args.mtbench101_judge_temperature})")

        def _new_pipe():
            return EpistemicPipeline(api_key, verbose=False,
                                     max_history_turns=args.max_history_turns)

        dialogue_records: list[dict] = []
        for i, bench in enumerate(selected):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(selected)}] {bench['id']} "
                  f"(task={bench['task']}, "
                  f"{len(bench['raw_history'])} turns)")
            rec = evaluate_mtbench101_official(
                pipe_factory=_new_pipe,
                bench=bench,
                tested_api_key=api_key,
                judge_api_key=judge_api_key,
                judge_model=args.mtbench101_judge_model,
                judge_temperature=args.mtbench101_judge_temperature,
                run_baseline=True,
                run_hybrid_dep_only=True,
            )
            dialogue_records.append(rec)
            s = rec["final_dep_map_stats"]
            print(f"  final dep_map: n={s['n_entries']} "
                  f"(h_auto={s['h_auto']}, h->h={s['h_to_h']}, "
                  f"degenerate={s['degenerate']})")
            print(f"  dialogue min scores: "
                  f"baseline={rec['dialogue_baseline_min']}, "
                  f"hybrid_dep_only={rec['dialogue_hybrid_dep_only_min']}")

        # Aggregate: per-task mean of dialogue-min scores (official).
        per_task: dict = {}
        for rec in dialogue_records:
            t = rec["task"]
            per_task.setdefault(t, {"baseline": [], "hybrid_dep_only": []})
            if rec["dialogue_baseline_min"] is not None:
                per_task[t]["baseline"].append(rec["dialogue_baseline_min"])
            if rec["dialogue_hybrid_dep_only_min"] is not None:
                per_task[t]["hybrid_dep_only"].append(
                    rec["dialogue_hybrid_dep_only_min"])

        def _mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        per_task_summary = {
            t: {
                "n_dialogues": len(v["baseline"]),
                "baseline_mean": _mean(v["baseline"]),
                "hybrid_dep_only_mean": _mean(v["hybrid_dep_only"]),
                "delta": (round(_mean(v["hybrid_dep_only"])
                                - _mean(v["baseline"]), 4)
                          if _mean(v["baseline"]) is not None
                          and _mean(v["hybrid_dep_only"]) is not None
                          else None),
            }
            for t, v in per_task.items()
        }
        b_means = [v["baseline_mean"] for v in per_task_summary.values()
                   if v["baseline_mean"] is not None]
        h_means = [v["hybrid_dep_only_mean"] for v in per_task_summary.values()
                   if v["hybrid_dep_only_mean"] is not None]
        macro = {
            "baseline_macro_mean": _mean(b_means),
            "hybrid_dep_only_macro_mean": _mean(h_means),
            "delta": (round(_mean(h_means) - _mean(b_means), 4)
                      if _mean(b_means) is not None
                      and _mean(h_means) is not None else None),
        }
        # Degenerate dep map heuristic (matches LoCoMo-style pattern):
        # flag a dialogue if h_auto >= 3 and h_to_h == 0 at final state.
        degenerate_flags = [
            {
                "id": r["id"],
                "task": r["task"],
                "final_dep_map_stats": r["final_dep_map_stats"],
                "degenerate_pattern": (
                    r["final_dep_map_stats"]["h_auto"] >= 3
                    and r["final_dep_map_stats"]["h_to_h"] == 0
                ),
            }
            for r in dialogue_records
        ]
        n_degenerate = sum(1 for d in degenerate_flags if d["degenerate_pattern"])

        out = {
            "config": {
                "benchmark": "mt-bench-101",
                "tested_backend": args.backend,
                "tested_model": pipeline.MODEL,
                "tested_url": pipeline.API_URL,
                "judge_model": args.mtbench101_judge_model,
                "judge_temperature": args.mtbench101_judge_temperature,
                "tested_temperature": 0,
                "history_teacher_forcing": True,
                "paths": ["baseline", "hybrid_dep_only"],
                "hybrid_engine_block": "dep_map_only (no state summary)",
                "tasks": sorted(task_filter),
                "per_task_requested": args.mtbench101_per_task,
                "data_path": args.data,
                "skip_first_tasks": sorted(MTBENCH101_SKIP_FIRST_TASKS),
                "need_ref_tasks": sorted(MTBENCH101_NEED_REF_TASKS),
                "aggregation": "turn_score -> dialogue_min -> task_mean",
            },
            "per_task_summary": per_task_summary,
            "macro_5_task_summary": macro,
            "degenerate_dep_pattern_flags": degenerate_flags,
            "n_degenerate_dialogues": n_degenerate,
            "dialogue_records": dialogue_records,
        }

        print(f"\n{'='*60}")
        print("SMOKE SUMMARY")
        print(f"{'='*60}")
        print(f"Per-task (dialogue-min then task-mean):")
        for t in sorted(per_task_summary):
            v = per_task_summary[t]
            print(f"  {t}: baseline={v['baseline_mean']}  "
                  f"hybrid_dep_only={v['hybrid_dep_only_mean']}  "
                  f"delta={v['delta']}  n={v['n_dialogues']}")
        print(f"5-task macro: baseline={macro['baseline_macro_mean']}  "
              f"hybrid_dep_only={macro['hybrid_dep_only_macro_mean']}  "
              f"delta={macro['delta']}")
        print(f"Degenerate dep-map dialogues (h_auto>=3 & h_to_h==0): "
              f"{n_degenerate}/{len(dialogue_records)}")

        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
        return
    # ---- End MT-Bench-101 Phase-1 smoke ----

    # ---- Controlled ablation path: reuse cached dep maps, skip ingest ----
    if args.cached_results is not None:
        if args.benchmark != "locomo":
            print("Error: --cached-results currently supports locomo only")
            sys.exit(1)
        with open(args.cached_results) as f:
            cached = json.load(f)
        bench_by_id = {b["id"]: b for b in benchmarks}

        all_results = []
        for cached_rec in cached:
            conv_id = cached_rec["id"]
            if conv_id not in bench_by_id:
                print(f"[warn] cached conv {conv_id} not in --data source; skip")
                continue
            bench = bench_by_id[conv_id]
            cached_dep_map = cached_rec.get("dependency_map", {}) or {}

            queries = bench["queries"]
            if args.max_queries is not None:
                queries = queries[:args.max_queries]

            print(f"\n{'='*60}")
            print(f"[cached ablation] {conv_id}: "
                  f"{len(bench['turns'])} turns, "
                  f"{len(cached_dep_map)} cached dep entries, "
                  f"{len(queries)} queries")

            qa_records, diag = evaluate_locomo_official(
                pipe=None, bench=bench, queries=queries, api_key=api_key,
                model_ctx_tokens=args.model_ctx_tokens,
                with_engine=True,
                run_baseline=args.run_baseline,
                engine_state_mode=args.engine_state_mode,
                engine_state_top_k=args.engine_state_top_k,
                cached_dep_map=cached_dep_map)

            record: dict = {
                "id": conv_id,
                "n_turns": len(bench["turns"]),
                "n_sessions": len(bench.get("session_dates", {})),
                "dependency_map": cached_dep_map,
                "locomo_mode": "official",
                "engine_state_mode": args.engine_state_mode,
                "engine_state_top_k": args.engine_state_top_k,
                "cached_source": args.cached_results,
                "diagnostics": diag,
                "qa_results": qa_records,
            }
            es = diag.get("engine_state", {})
            if es:
                print(f"  engine_state: mode={es.get('mode')} "
                      f"kept={es.get('kept_dep_entries')}/"
                      f"{es.get('original_dep_entries')} "
                      f"(dropped_degenerate={es.get('n_dropped_degenerate', 0)}, "
                      f"h->h={es.get('h_to_h_entries', 0)}) "
                      f"chars={es.get('engine_block_chars')} "
                      f"~tok={es.get('engine_block_approx_tokens')}")
            if args.run_baseline:
                record["metrics_official_baseline"] = compute_metrics(
                    qa_records, "official_baseline", include_engine=False)
                print(f"  metrics_official_baseline: "
                      f"{record['metrics_official_baseline']}")
            record["metrics_official_hybrid"] = compute_metrics(
                qa_records, "official_hybrid", include_engine=False)
            print(f"  metrics_official_hybrid_{args.engine_state_mode}: "
                  f"{record['metrics_official_hybrid']}")
            if diag["truncation_events"]:
                print(f"  [warn] context truncated on "
                      f"{diag['truncation_events']}/{len(qa_records)} queries.")
            all_results.append(record)

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return
    # ---- End cached ablation path ----

    all_results = []
    for i, bench in enumerate(benchmarks[:args.max_conversations]):
        raw_turns = bench["turns"]
        n_sessions = len(bench.get("session_dates", {}))
        print(f"\n{'='*60}")
        print(f"Conversation {i+1}: {bench['id']} "
              f"({len(raw_turns)} turns, {n_sessions} sessions)")

        pipe = EpistemicPipeline(api_key, verbose=False,
                                 max_history_turns=args.max_history_turns)
        for t in raw_turns:
            pipe.process_turn(t["speaker"], t["text"])

        queries = bench["queries"]
        if args.max_queries is not None:
            queries = queries[:args.max_queries]

        record: dict = {
            "id": bench["id"],
            "n_turns": len(raw_turns),
            "n_sessions": n_sessions,
            "dependency_map": pipe.get_dependency_map(),
        }

        if not queries or args.benchmark != "locomo":
            # Non-LoCoMo or no queries: fall back to the minimal hybrid eval.
            if queries:
                record["qa_results"] = evaluate_generic_queries(pipe, queries, api_key)
            all_results.append(record)
            continue

        if args.locomo_mode == "official":
            qa_records, diag = evaluate_locomo_official(
                pipe, bench, queries, api_key,
                model_ctx_tokens=args.model_ctx_tokens,
                with_engine=args.with_engine,
                run_baseline=args.run_baseline,
                engine_state_mode=args.engine_state_mode,
                engine_state_top_k=args.engine_state_top_k)
            record["qa_results"] = qa_records
            record["locomo_mode"] = "official"
            record["engine_state_mode"] = args.engine_state_mode
            record["engine_state_top_k"] = args.engine_state_top_k
            record["diagnostics"] = diag
            es = diag.get("engine_state", {})
            if es:
                print(f"  engine_state: mode={es.get('mode')} "
                      f"kept={es.get('kept_dep_entries')}/"
                      f"{es.get('original_dep_entries')} "
                      f"(dropped_degenerate={es.get('n_dropped_degenerate', 0)}, "
                      f"h->h={es.get('h_to_h_entries', 0)}) "
                      f"chars={es.get('engine_block_chars')} "
                      f"~tok={es.get('engine_block_approx_tokens')}")
            if args.run_baseline:
                record["metrics_official_baseline"] = compute_metrics(
                    qa_records, "official_baseline", include_engine=False)
                print(f"  metrics_official_baseline: "
                      f"{record['metrics_official_baseline']}")
            if args.with_engine:
                # Engine-direct coverage is not tracked in official mode
                # because the prompt is not ECM-shaped; include_engine=False.
                record["metrics_official_hybrid"] = compute_metrics(
                    qa_records, "official_hybrid", include_engine=False)
                print(f"  metrics_official_hybrid:   "
                      f"{record['metrics_official_hybrid']}")
            if diag["truncation_events"]:
                print(f"  [warn] context truncated on "
                      f"{diag['truncation_events']}/{len(qa_records)} queries "
                      f"(approx chars/token = {_APPROX_CHARS_PER_TOKEN}).")
            if diag["skipped"].get("cat5_missing_answer"):
                print(f"  [info] skipped {diag['skipped']['cat5_missing_answer']} "
                      "cat-5 items missing 'answer' (not official-comparable).")
        else:
            qa_records, diag = evaluate_locomo_diagnostic(
                pipe, bench, queries, api_key,
                model_ctx_tokens=args.model_ctx_tokens,
                run_baseline=args.run_baseline)
            record["qa_results"] = qa_records
            record["locomo_mode"] = "diagnostic"
            record["diagnostics"] = diag
            record["metrics_diagnostic_hybrid"] = compute_metrics(
                qa_records, "diagnostic_hybrid", include_engine=True)
            print(f"  metrics_diagnostic_hybrid:   "
                  f"{record['metrics_diagnostic_hybrid']}")
            if args.run_baseline:
                record["metrics_diagnostic_baseline"] = compute_metrics(
                    qa_records, "diagnostic_baseline", include_engine=False)
                print(f"  metrics_diagnostic_baseline: "
                      f"{record['metrics_diagnostic_baseline']}")
            if diag["truncation_events"]:
                print(f"  [warn] context truncated on "
                      f"{diag['truncation_events']}/{len(qa_records)} queries.")
            if diag["skipped"].get("cat5_missing_answer"):
                print(f"  [info] skipped {diag['skipped']['cat5_missing_answer']} "
                      "cat-5 items missing 'answer'.")

        all_results.append(record)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
