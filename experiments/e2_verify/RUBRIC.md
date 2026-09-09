# Grounded / Ungrounded Annotation Rubric for E2 (Direct Verifier Evaluation)

## What this rubric is for

The E2 experiment compares two systems on the same task:
- **Verifier**: structural reachability check `Verify(c, D_t)` over the
  pipeline's epistemic state.
- **LLM-only baseline**: full transcript + candidate continuation `c`,
  free-form judgement.

This rubric defines the **ground truth label** that both systems are scored
against. The labels are set by the test-set author and an independent
re-annotator, not by either system being evaluated.

## The label

For each item `(t, c)` — a Phase 2 truncation at turn `t` (turns 1..t are
visible) and a candidate continuation `c` — the label is one of:

- `grounded`   — `c` rests on conversation content already present in
  turns 1..t and is not contradicted or abandoned by any later content
  visible at time `t`.
- `ungrounded` — `c` either has no support in turns 1..t at all, or rests
  on content that was abandoned, weakened, or contradicted within turns
  1..t.

**The label is a property of `c` relative to the truncation `t`. It is not
about whether `c` is later proven correct in the full conversation.**

## Decision procedure (apply in order)

1. **No support at all.** If nothing in turns 1..t mentions, implies, or
   hints at the content of `c`, label `ungrounded`.
   - Example: at `t=5`, candidate "We should adopt CRDTs over Operational
     Transformation" — Phase 2 is a debugging conversation; no CRDT
     content appears at any `t`. → `ungrounded`.

2. **Surface presence only.** If `c` reuses vocabulary from turns 1..t
   but the meaning is incompatible with what was said, label `ungrounded`.
   - Example: at `t=10`, candidate "Redis is causing the auth failures
     via session lookup errors" — Redis and auth both appear in earlier
     turns, but Alice already established at T7/T9 that auth's 401s are
     not from Redis (different error code). The candidate restates a
     position that was explicitly weakened. → `ungrounded`.

3. **Restates abandoned hypothesis.** If `c` re-asserts a hypothesis
   that has been abandoned, weakened, or revised within turns 1..t,
   label `ungrounded` regardless of how recently it was abandoned.
   - Example: after T9 Alice abandons h2 ("Redis → auth failures"). At
     `t=12`, candidate "Auth is failing because Redis sessions are
     unavailable" → `ungrounded` (restates h2 after abandonment).

4. **Contradicts a resolved conclusion.** If `c` directly contradicts a
   conclusion that was resolved (`Resolve` operation) within turns 1..t,
   label `ungrounded`.
   - Example: after T13 the team resolves on causal direction
     `auth bug → retry storm → Redis exhaustion`. At `t=13`, candidate
     "The cascade is Redis → Auth → Payment" contradicts the resolution.
     → `ungrounded`.

5. **Rests on entities present in `D_t`.** If every load-bearing concept
   in `c` is either an observation introduced in turns 1..t or a
   hypothesis still active or resolved at time `t`, label `grounded`.
   - Example: at `t=8`, candidate "The Redis pool exhaustion explains
     the Stripe 429s because Payment service falls through to Stripe
     when rate-limit checks fail" rests on o8 (Redis timeout, T3),
     o7 (Stripe 429s, T3), and the rate-limit-bypass mechanism (T5).
     All three exist in `D_8`. → `grounded`.

## Edge cases and how to handle them

- **Paraphrase vs.~direct quote.** `c` does not have to use the exact
  wording from earlier turns. If a reasonable rephrasing of `c` matches
  an earlier hypothesis or observation, that's a match. The point is
  semantic grounding, not lexical match.

- **Partial grounding.** If `c` rests on three concepts and two are in
  `D_t` but one is genuinely new and unsupported, label `ungrounded`.
  The verifier requires **every** load-bearing concept to be supported.

- **Continuations that introduce new hypotheses.** A `c` of the form
  "What about <new theory>?" with no support in `D_t` is `ungrounded`
  even if the new theory is plausible. Plausibility is not grounding.
  Only the support relation in `D_t` matters.

- **Time travel.** `c` is judged at time `t`. Do not let knowledge of
  later turns leak in. If you find yourself thinking "but later they
  abandon this", that only matters if the abandonment happens at some
  `t' ≤ t`.

- **When in doubt, write a comment.** Each item has a `notes` field
  for cases that needed thought. The independent annotator should look
  at these only after their own pass.

## What goes in `asserts_id`

Each item has an `asserts_id` field naming the entity in `D_t` that `c`
restates, if any. The verifier uses this to walk the dependency graph.
- Set `asserts_id` to the canonical id (`o5`, `h2`, etc.) of the entity
  being restated.
- Set `asserts_id: null` if `c` introduces a genuinely new claim with
  no corresponding entity in `D_t`.
- For *stale* claims, set `asserts_id` to the abandoned hypothesis's
  id (`h2` after T9). The verifier should walk the graph and discover
  the abandonment.
- For *cross-conv* and *counterfactual* claims, `asserts_id` is
  typically `null` (no matching entity in `D_t`).

## Independent re-annotation protocol

The annotator receives a **blinded subset** of 20 items (the
`_blinded.yaml` file): only the truncation `t`, the candidate `c`, and
the category-stripped item id. They do **not** see the test-set author's
labels, `asserts_id` choices, or `notes`. They produce their own
`grounded`/`ungrounded` label per item.

After both passes, compute Cohen's κ on the 20-item overlap.
- κ ≥ 0.7: substantial agreement; report κ, proceed with the test set
  as-is.
- 0.6 ≤ κ < 0.7: moderate agreement; revise the rubric or disputed items
  before reporting.
- κ < 0.6: re-do the test set design.

**Until an independent annotator has completed the overlap, do NOT
report κ in the paper.**
