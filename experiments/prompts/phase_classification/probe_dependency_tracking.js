// Experiment: Downstream Probe Questions — Dependency Tracking
// Grounded Continuation — Phase 2 Scenario
// Model: claude-sonnet-4-20250514 | temperature=0
//
// This script tests whether the structured epistemic model improves
// downstream reasoning by posing selective-dependency questions to
// the LLM under two conditions:
//   Raw: conversation text only
//   Model-informed: conversation + structured epistemic state
//
// An LLM judge scores answers 1-5 against ground truth.
// Result: Raw=4.0/5, Model-informed=4.4/5, Delta=+0.4

const MODEL = "claude-sonnet-4-20250514";

// Full 13-turn conversation (same as classification experiment)
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

// Structured epistemic state (what the symbolic engine would produce)
const MODEL_STATE = `STRUCTURED EPISTEMIC MODEL (final state after T13):

HYPOTHESIS LIFECYCLE:
  h1: "Redis exhaustion -> rate-limit bypass -> Stripe 429s"
    Created: T5 (Bob). Status: RESOLVED. Subsumed into unified chain.
    Depends on: observation o8 (Redis timeouts from Payment).

  h2: "Redis failure -> Auth failures (via shared session cache)"
    Created: T6 (Carol). ABANDONED at T9.
    Undermined by: o9 (error code = "token expired" 401, not "session lookup failed" 503).
    Abandonment reasoning: wrong error type proves Redis is not causing auth failures.

  h3: "Token bug -> frontend retry loop -> 3x traffic amplification"
    Created: T11 (Alice). Status: RESOLVED. Subsumed into unified chain.
    Depends on: o6 (3x traffic), assumption that frontend retries on token expiry.

  h4: "Token bug -> retry storm -> Redis pool exhaustion"
    Created: T12 (Carol). Status: RESOLVED.
    KEY REVERSAL: Auth problems cause Redis exhaustion (opposite of h2).
    Depends on: h3 (retry mechanism), o8 (Redis exhaustion observed).

UNIFIED CAUSAL CHAIN:
  token bug -> expired tokens -> frontend retries -> 3x auth traffic -> Redis pool exhaustion -> rate-limit bypass -> Stripe 429s

DEPENDENCY MAP:
  "Token bug is root cause" depends on: o9 (error code evidence), o6 (3x traffic), h3 (retry mechanism)
  "Redis exhaustion is downstream" depends on: h4 (reversal), o8 (Redis timeouts)
  "Stripe 429s" depends on: h1 (rate-limit bypass mechanism), Redis being exhausted
  "Auth and Redis are independent" -- OBSOLETE after T12 (they ARE connected, but in reverse direction)

OBSERVATION REGISTRY:
  o1: auth failure rate elevated (T1, Carol)
  o2: payment failure rate elevated (T1, Carol)
  o3: "database health" alert -- RECLASSIFIED at T4 as Redis alert (monitoring miscategorisation)
  o5: 401 "token expired" errors (T2, Alice) -- KEY EVIDENCE against h2
  o6: 3x auth traffic at 2am (T2, Alice) -- explained by h3 (retry loop)
  o7: Stripe 429 rate-limit errors (T3, Bob) -- explained by h1
  o8: Redis connection timeouts (T3, Bob) -- explained by h4 (retry storm overwhelms Redis)
  o9: error code specificity: "token expired" not "session lookup failed" (T7, Alice) -- KEY EVIDENCE

KEY EPISTEMIC SHIFTS:
  T4: Awareness expansion (monitoring miscategorisation discovered)
  T7: h2 undermined (error code discrepancy)
  T9: h2 abandoned (auth != Redis)
  T12: Causal reversal (Auth->Redis, not Redis->Auth)`;

// Probe questions with ground truth
const PROBES = [
  {
    id: "Q1", type: "selective-retraction",
    q: "Suppose it turns out Bob's Redis timeout observation (T3) was a misreading -- Redis is actually fine and always was. Which parts of the team's final conclusion would need to be revised, and which parts would remain valid? Be specific.",
    gt: "If Redis timeouts (o8) were retracted: (1) h1 (Redis->rate-limit bypass->Stripe 429s) COLLAPSES -- its entire basis was Redis being down. Stripe 429s need new explanation. (2) h4 (retry storm->Redis exhaustion) becomes MOOT -- Redis was never exhausted. (3) HOWEVER, token bug diagnosis REMAINS VALID: o9 (error code evidence) independently proves token expiry is the auth issue, h3 (retry loop->3x traffic) still explains traffic anomaly, h2 abandonment still holds (disproved by error codes, not Redis status). Auth-side diagnosis survives; Redis/payment-side collapses.",
  },
  {
    id: "Q2", type: "selective-retraction",
    q: "Suppose Alice's error code observation at T7 was wrong -- the errors actually were '503 session lookup failed,' not '401 token expired.' How would this change the team's final conclusion?",
    gt: "If o9 were 503 instead of 401: (1) h2 (Redis->Auth) would NOT have been undermined/abandoned -- it would remain active. (2) Single root cause picture from T6 (Redis causes everything) would still hold. (3) h3 (retry loop) and h4 (causal reversal) would likely NEVER HAVE BEEN PROPOSED -- only needed because h2 was abandoned. (4) Entire diagnostic trajectory after T7 changes. (5) h1 remains valid regardless.",
  },
  {
    id: "Q3", type: "partial-counterfactual",
    q: "The team's final conclusion has the token bug as root cause. List ALL the evidence this depends on. Which single piece, if retracted, would do the MOST damage?",
    gt: "Depends on: (1) o9: error codes are 'token expired' not 'session lookup failed'. (2) o6: 3x traffic. (3) Retry mechanism assumption (Alice, T11). (4) o8: Redis connection timeouts. (5) 3x traffic sufficient to exhaust Redis (T13 argument). MOST DAMAGING retraction: o9 (error code evidence), because it disproved h2, leading to the entire diagnostic pivot. Without o9, team would still believe Redis is root cause.",
  },
  {
    id: "Q4", type: "state-tracking",
    q: "At the end of T9 (before T10), list: (a) all active hypotheses, (b) all abandoned hypotheses and why, (c) all unexplained observations, (d) open questions.",
    gt: "(a) Active: h1 (Redis->rate-limit bypass->Stripe 429s). (b) Abandoned: h2 (Redis->Auth) -- error code is 'token expired' (401) not 'session lookup failed' (503). (c) Unexplained: o5 (WHY are tokens expiring?), o6 (3x traffic -- no hypothesis). (d) Open: What causes token expiry? Why is traffic 3x? Are auth and payment truly independent?",
  },
  {
    id: "Q5", type: "attribution",
    q: "For each: (a) Stripe 429 errors, (b) 3x auth traffic, (c) the 'database health' alert -- state which hypothesis explains it in the final model and when that explanation was first proposed.",
    gt: "(a) Stripe 429s: h1 (Redis exhaustion->rate-limit bypass), first proposed T5 by Bob. In final model, Redis exhaustion caused by token bug retry storm (h4, T12). (b) 3x traffic: h3 (token bug->retry loop), first proposed T11 by Alice. Unexplained from T2 until T11. (c) Database alert: awareness expansion at T4 -- Carol discovered monitoring miscategorises Redis errors as 'database health.' Not a causal hypothesis; monitoring configuration issue (red herring).",
  },
];

// To reproduce:
// For each probe question, make two API calls:
//   Raw: system="analyze debugging conversation, answer from text only"
//        user=CONVERSATION + question
//   Model: system="analyze with structured epistemic model"
//          user=CONVERSATION + MODEL_STATE + question
// Then judge each answer against ground truth using a third API call.
// Score 1-5 on causal dependency tracking accuracy.
