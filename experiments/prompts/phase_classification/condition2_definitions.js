// Experiment Condition 2: Definitions Prompt (Fair, No Leaking)
// Grounded Continuation — Phase 2 Classification
// Model: claude-sonnet-4-20250514 | 5 runs (1×temp=0, 4×temp=1)
//
// This script sends the full 13-turn debugging conversation to Claude Sonnet
// with clear general definitions, GENERIC examples (not from this conversation),
// explicit guidance on critical distinctions (Revise vs Undermine, Expand-Awareness
// vs Observe), and implicit question detection. NO conversation-specific hints,
// NO epistemic model state.
//
// Result: F1=0.85, Exact=54%, Key Shift F1=0.74, Key Shift Exact=25%

const MODEL = "claude-sonnet-4-20250514";

const CONVERSATION = `T1 -- Carol: P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation. Customer complaints started around 2:15am.

T2 -- Alice: I'm seeing 401 Unauthorized spikes in auth logs starting around 2am. Tokens being rejected as expired. But I also see our auth request volume is way up -- about 3x normal. That's strange. More users shouldn't be logging in at 2am.

T3 -- Bob: Payment side -- I see Stripe returning 429s. We're sending them way more requests than normal. And I can confirm I see the database alert too -- Redis connection timeouts from Payment Service.

T4 -- Carol: OK, so the database alert is about Redis, not the primary DB. Still concerning. Bob, are you seeing Redis issues from Payment's side?

T5 -- Bob: Yes. Connection pool exhausted. We can't get connections to Redis. That's why our rate-limit checks are failing -- we store rate-limit counters in Redis. When we can't check the counter, our code falls through and sends the request to Stripe anyway. That would explain the 429s -- we're hitting Stripe without rate limiting.

T6 -- Carol: So the chain might be: Redis is sick -> Payment loses rate limiting -> Stripe gets hammered. And separately, Redis being sick -> Auth has problems too? Alice, does Auth use Redis?

T7 -- Alice: Yes, Auth uses the same Redis cluster for session caching. If Redis is down, token validation would fail because we can't look up the session. That would explain the 401s... but wait, the 401s I'm seeing are specifically 'token expired,' not 'session lookup failed.' Those are different error codes. The tokens are being rejected because their expiry timestamps are in the past, not because Redis is unavailable.

T8 -- Carol: Hmm. So the auth failures might not be caused by Redis after all?

T9 -- Alice: I don't think so. The error type is wrong. If Redis were down, Auth would return a 503 Service Unavailable, not a 401 with 'token expired.' I'm seeing 401s. So the token expiry issue is a different problem from the Redis problem.

T10 -- Bob: But then why is auth traffic 3x normal? If the token expiry issue is independent of Redis, what's generating all that auth traffic?

T11 -- Alice: Could be the frontend retrying. When a user gets a 'token expired' error, the frontend automatically tries to refresh the token. If the new token is also expired, it retries again. That's a retry loop. So the token bug generates its own amplified traffic.

T12 -- Carol: Wait -- so the retry storm from the token bug could be what's exhausting Redis? Not Redis causing the auth problem, but the auth problem causing the Redis overload?

T13 -- Alice: That's... actually plausible. If auth traffic is 3x, and every auth request hits Redis for session lookup, the Redis connection pool could be overwhelmed. So the chain would be: token bug -> retry storm -> Redis exhaustion. And then Redis exhaustion -> Payment rate-limit bypass -> Stripe 429s. The whole thing cascades from the token bug.`;

const PROMPT = `You are classifying utterances in a multi-agent conversation into epistemic operations. Each utterance may have one or more operations.

## Operation Definitions

**Observe**: Report factual data without explanation. New measurements, error codes, metrics, timestamps.
  Generic example: "The server logs show 502 errors starting at 3pm."

**Hypothesize**: Propose a NEW explanatory causal chain that did not previously exist in the conversation. Must introduce a novel explanation, not restate someone else's.
  Generic example: "Maybe the memory leak is causing the timeouts."

**Support**: Provide evidence or reasoning that STRENGTHENS an existing hypothesis already proposed by someone. The hypothesis must already be on the table.
  Generic example: "That's consistent with what I'm seeing in the logs too."

**Undermine**: Provide evidence or reasoning that WEAKENS an existing hypothesis. The hypothesis becomes less plausible but is not yet abandoned.
  Generic example: "But the timing doesn't match -- the errors started before the deployment."

**Revise**: ABANDON a hypothesis entirely, or FUNDAMENTALLY RESTRUCTURE the group's causal understanding. Key indicators: explicit statements that a previous belief is wrong ("X is a different problem from Y"), or reversing the causal direction between two phenomena ("not A causing B, but B causing A").
  Generic example: "So the network issue isn't related to the database at all -- they're independent problems."
  Generic example: "Wait, it's the opposite -- the client is overwhelming the server, not the server failing on its own."

**Expand-Awareness**: Introduce a COMPLETELY NEW dimension of reasoning that was previously UNCONCEIVED by anyone in the conversation. This is NOT the same as learning a new fact (Observe) or changing a belief (Revise). It means discovering that there is a relevant concept or possibility that nobody had been considering at all.
  Generic example: "Actually, the 'disk full' alert is being triggered by the logging system, not actual disk usage -- the monitoring is misconfigured."
  (Here, the idea that monitoring itself could be wrong was not in anyone's reasoning.)

**Resolve**: Elevate a tentative hypothesis to an accepted group conclusion. The group converges and treats the explanation as settled.
  Generic example: "Yes, that's it -- the root cause is the configuration change from yesterday."

**Question**: Ask for information or open a new line of inquiry. Includes IMPLICIT questions where a speaker opens a topic for investigation or flags something anomalous that requires explanation.
  Generic example: "What's going on with the payment service?"
  Generic example (implicit): "Something is off -- we shouldn't be seeing this much traffic at night." (implicitly asks "why?")

## Key Distinctions

- **Undermine vs Revise**: Undermine weakens a hypothesis; Revise abandons it or restructures the causal model. If someone explicitly says a prior explanation is wrong or separates two things previously thought to be connected, that is Revise.

- **Expand-Awareness vs Observe**: Observe adds a new data point within existing reasoning dimensions. Expand-Awareness adds an entirely new dimension -- a concept or possibility that nobody had been thinking about.

- **Hypothesize+Revise**: When someone proposes a new causal chain that simultaneously reverses a previous causal understanding, classify as BOTH Hypothesize and Revise.

## Conversation to classify

${CONVERSATION}

## Task

For each turn T1 through T13, provide the epistemic operations. Respond with ONLY a JSON object:
{
  "T1": ["Op1", "Op2"],
  "T2": ["Op1"],
  ...
}`;

const GROUND_TRUTH = {
  T1: ["Observe", "Question"], T2: ["Observe", "Question"],
  T3: ["Observe"], T4: ["Expand-Awareness", "Question"],
  T5: ["Observe", "Hypothesize"], T6: ["Support", "Hypothesize", "Question"],
  T7: ["Observe", "Undermine"], T8: ["Question"],
  T9: ["Revise"], T10: ["Question"],
  T11: ["Hypothesize"], T12: ["Hypothesize", "Revise"],
  T13: ["Support", "Resolve"],
};

// To run: send PROMPT to the API with model=claude-sonnet-4-20250514,
// max_tokens=1000, temperature=0 (run 1) or 1 (runs 2-5).
// Parse the JSON response and compute set-based F1 per turn against GROUND_TRUTH.
