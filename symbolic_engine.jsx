import { useState, useCallback, useRef, useMemo } from "react";

// ============================================================
// SYMBOLIC EPISTEMIC ENGINE
// ============================================================
class EpistemicEngine {
  constructor() {
    this.observations = new Map();    // id → {content, turn, speaker, explained_by}
    this.hypotheses = new Map();      // id → {content, turn, speaker, status, plausibility, supports, undermines, explains}
    this.awareness = new Set();       // proposition ids in play
    this.dependencies = new Map();    // hypothesis_id → Set of assumption ids
    this.arguments = [];              // {id, claim, speaker, turn, type, target}
    this.openQuestions = [];
    this.history = [];                // log of all operations applied
    this.turn = 0;
  }

  clone() {
    const e = new EpistemicEngine();
    e.observations = new Map(JSON.parse(JSON.stringify([...this.observations])));
    e.hypotheses = new Map(JSON.parse(JSON.stringify([...this.hypotheses])));
    e.awareness = new Set(this.awareness);
    e.dependencies = new Map([...this.dependencies].map(([k, v]) => [k, new Set(v)]));
    e.arguments = JSON.parse(JSON.stringify(this.arguments));
    e.openQuestions = [...this.openQuestions];
    e.history = [...this.history];
    e.turn = this.turn;
    return e;
  }

  // Core operations
  observe(id, content, turn, speaker) {
    this.observations.set(id, { content, turn, speaker, explained_by: null });
    this.awareness.add(id);
    this.history.push({ op: "Observe", id, content, turn });
  }

  hypothesize(id, content, turn, speaker, explains = [], depends_on = []) {
    this.hypotheses.set(id, {
      content, turn, speaker, status: "active", plausibility: "medium",
      supports: [], undermines: [], explains: [...explains]
    });
    this.awareness.add(id);
    if (depends_on.length) this.dependencies.set(id, new Set(depends_on));
    // Mark observations as explained
    for (const obsId of explains) {
      const obs = this.observations.get(obsId);
      if (obs) obs.explained_by = id;
    }
    this.history.push({ op: "Hypothesize", id, content, turn });
  }

  support(hyp_id, evidence, turn, speaker) {
    const h = this.hypotheses.get(hyp_id);
    if (h) {
      h.supports.push({ evidence, turn, speaker });
      if (h.status === "active") h.plausibility = "high";
    }
    this.history.push({ op: "Support", target: hyp_id, evidence, turn });
  }

  undermine(hyp_id, evidence, turn, speaker) {
    const h = this.hypotheses.get(hyp_id);
    if (h) {
      h.undermines.push({ evidence, turn, speaker });
      if (h.status === "active") {
        h.status = "weakened";
        h.plausibility = "low";
      }
    }
    this.history.push({ op: "Undermine", target: hyp_id, evidence, turn });
  }

  revise(hyp_id, reason, turn, speaker) {
    const h = this.hypotheses.get(hyp_id);
    if (h) {
      h.status = "abandoned";
      h.plausibility = "none";
      h.abandoned_reason = reason;
      h.abandoned_turn = turn;
      // Un-explain observations
      for (const obsId of (h.explains || [])) {
        const obs = this.observations.get(obsId);
        if (obs && obs.explained_by === hyp_id) obs.explained_by = null;
      }
    }
    this.history.push({ op: "Revise", target: hyp_id, reason, turn });
  }

  expandAwareness(id, content, turn, speaker) {
    this.awareness.add(id);
    this.history.push({ op: "Expand-Awareness", id, content, turn });
  }

  resolve(hyp_id, turn, speaker) {
    const h = this.hypotheses.get(hyp_id);
    if (h) {
      h.status = "resolved";
      h.plausibility = "certain";
    }
    this.history.push({ op: "Resolve", target: hyp_id, turn });
  }

  question(content, turn, speaker) {
    this.openQuestions.push({ content, turn, speaker, resolved: false });
    this.history.push({ op: "Question", content, turn });
  }

  // ============================================================
  // DEPENDENCY TRACKING (Proposition 1)
  // ============================================================
  getAffected(assumption_id) {
    const affected = [];
    for (const [hyp_id, deps] of this.dependencies) {
      if (deps.has(assumption_id)) {
        affected.push(hyp_id);
      }
    }
    return affected;
  }

  retractAssumption(assumption_id) {
    const affected = this.getAffected(assumption_id);
    const result = {
      retracted: assumption_id,
      affected_hypotheses: affected,
      unaffected_hypotheses: [],
      flagged_for_review: [],
    };
    for (const [hyp_id, h] of this.hypotheses) {
      if (affected.includes(hyp_id)) {
        result.flagged_for_review.push({
          id: hyp_id, content: h.content, status: h.status,
          reason: `Depends on retracted assumption: ${assumption_id}`
        });
      } else {
        result.unaffected_hypotheses.push({ id: hyp_id, content: h.content, status: h.status });
      }
    }
    return result;
  }

  // ============================================================
  // CONSISTENCY CHECKS (what the engine validates)
  // ============================================================
  checkConsistency() {
    const issues = [];
    // Check: weakened hypothesis being undermined again should trigger Revise
    for (const [id, h] of this.hypotheses) {
      if (h.status === "weakened" && h.undermines.length >= 2) {
        issues.push({ type: "suggest_revise", hyp: id, msg: `${id} undermined ${h.undermines.length} times — consider Revise` });
      }
    }
    // Check: unexplained observations
    for (const [id, obs] of this.observations) {
      if (!obs.explained_by) {
        const anyExplains = [...this.hypotheses.values()].some(h => h.status === "active" && (h.explains || []).includes(id));
        if (!anyExplains) {
          issues.push({ type: "unexplained", obs: id, msg: `Observation ${id} has no active explanatory hypothesis` });
        }
      }
    }
    return issues;
  }

  // State summary
  getSummary() {
    const active = [...this.hypotheses].filter(([, h]) => h.status === "active" || h.status === "resolved");
    const abandoned = [...this.hypotheses].filter(([, h]) => h.status === "abandoned");
    const weakened = [...this.hypotheses].filter(([, h]) => h.status === "weakened");
    const unexplained = [...this.observations].filter(([, o]) => !o.explained_by);
    return { active, abandoned, weakened, unexplained, openQuestions: this.openQuestions.filter(q => !q.resolved) };
  }
}

// ============================================================
// PHASE 2 SCENARIO: Pre-scripted engine updates
// ============================================================
const PHASE2_TURNS = [
  { id: "T1", speaker: "Carol", text: "P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation. Customer complaints started around 2:15am.",
    apply: (e) => { e.observe("o1","auth failure rate elevated","T1","Carol"); e.observe("o2","payment failure rate elevated","T1","Carol"); e.observe("o3","database health degradation alert","T1","Carol"); e.observe("o4","customer complaints from ~2:15am","T1","Carol"); e.question("What is causing the failures?","T1","Carol"); }},
  { id: "T2", speaker: "Alice", text: "I'm seeing 401 Unauthorized spikes in auth logs starting around 2am. Tokens being rejected as expired. But I also see our auth request volume is way up — about 3x normal. That's strange.",
    apply: (e) => { e.observe("o5","401 'token expired' errors from ~2am","T2","Alice"); e.observe("o6","auth traffic 3x normal at 2am","T2","Alice"); e.question("Why is auth traffic 3x at 2am?","T2","Alice"); }},
  { id: "T3", speaker: "Bob", text: "Payment side — I see Stripe returning 429s. We're sending them way more requests than normal. And I can confirm the database alert too — Redis connection timeouts from Payment Service.",
    apply: (e) => { e.observe("o7","Stripe 429 rate-limit errors","T3","Bob"); e.observe("o8","Redis connection timeouts from Payment","T3","Bob"); }},
  { id: "T4", speaker: "Carol", text: "OK, so the database alert is about Redis, not the primary DB. Still concerning. Bob, are you seeing Redis issues from Payment's side?",
    apply: (e) => { e.expandAwareness("mis_monitor","monitoring miscategorises Redis as 'database health'","T4","Carol"); e.question("Redis issues from Payment side?","T4","Carol"); }},
  { id: "T5", speaker: "Bob", text: "Yes. Connection pool exhausted. We can't get connections to Redis. That's why our rate-limit checks are failing — we store rate-limit counters in Redis. When we can't check the counter, our code falls through and sends the request to Stripe anyway.",
    apply: (e) => { e.observe("o8b","Redis connection pool exhausted","T5","Bob"); e.hypothesize("h1","Redis pool exhaustion → rate-limit bypass → Stripe 429s","T5","Bob",["o7","o8"],["o8"]); }},
  { id: "T6", speaker: "Carol", text: "So the chain might be: Redis is sick → Payment loses rate limiting → Stripe gets hammered. And separately, Redis being sick → Auth has problems too? Alice, does Auth use Redis?",
    apply: (e) => { e.support("h1","Carol restates the chain","T6","Carol"); e.hypothesize("h2","Redis failure → Auth failures (via shared session cache)","T6","Carol",["o1","o5"],["o8"]); e.question("Does Auth use Redis?","T6","Carol"); }},
  { id: "T7", speaker: "Alice", text: "Yes, Auth uses the same Redis cluster for session caching. If Redis is down, token validation would fail... but wait, the 401s I'm seeing are specifically 'token expired,' not 'session lookup failed.' Those are different error codes.",
    apply: (e) => { e.observe("o9","error code is 'token expired' (401), NOT 'session lookup failed' (503)","T7","Alice"); e.undermine("h2","error code inconsistent: h2 predicts 503, observed 401","T7","Alice"); }},
  { id: "T8", speaker: "Carol", text: "Hmm. So the auth failures might not be caused by Redis after all?",
    apply: (e) => { e.question("Are auth failures caused by Redis?","T8","Carol"); }},
  { id: "T9", speaker: "Alice", text: "I don't think so. The error type is wrong. If Redis were down, Auth would return a 503 Service Unavailable, not a 401 with 'token expired.' So the token expiry issue is a different problem from the Redis problem.",
    apply: (e) => { e.revise("h2","error code proves auth failures are independent of Redis","T9","Alice"); }},
  { id: "T10", speaker: "Bob", text: "But then why is auth traffic 3x normal? If the token expiry issue is independent of Redis, what's generating all that auth traffic?",
    apply: (e) => { e.question("Why is auth traffic 3x if not caused by Redis?","T10","Bob"); }},
  { id: "T11", speaker: "Alice", text: "Could be the frontend retrying. When a user gets a 'token expired' error, the frontend automatically tries to refresh the token. If the new token is also expired, it retries again. That's a retry loop.",
    apply: (e) => { e.hypothesize("h3","token bug → frontend retry loop → 3x traffic amplification","T11","Alice",["o6"],["o5"]); }},
  { id: "T12", speaker: "Carol", text: "Wait — so the retry storm from the token bug could be what's exhausting Redis? Not Redis causing the auth problem, but the auth problem causing the Redis overload?",
    apply: (e) => { e.hypothesize("h4","token bug → retry storm → Redis pool exhaustion (CAUSAL REVERSAL: Auth→Redis)","T12","Carol",["o8"],["h3","o6"]); }},
  { id: "T13", speaker: "Alice", text: "That's... actually plausible. If auth traffic is 3x, and every auth request hits Redis for session lookup, the Redis connection pool could be overwhelmed. The whole thing cascades from the token bug.",
    apply: (e) => { e.support("h4","mechanistic argument: 3x traffic × Redis lookups = pool exhaustion","T13","Alice"); e.resolve("h4","T13","Alice"); e.resolve("h1","T13","Alice"); e.resolve("h3","T13","Alice");
      // Mark unified chain dependencies
      e.dependencies.set("h4", new Set(["o9","o6","h3"]));
      e.dependencies.set("h1", new Set(["o8","h4"]));
      e.dependencies.set("unified", new Set(["o9","o6","o8","h3","h4"]));
    }},
];

// Counterfactual scenarios
const COUNTERFACTUALS = [
  { id: "cf1", label: "Retract o9 (error code evidence)",
    desc: "What if Alice's error code observation were wrong — errors were 503, not 401?",
    assumption: "o9",
    expected: "h2 abandonment (T9) would not have happened. The single-root-cause picture (Redis→everything) would persist. h3, h4, and the causal reversal would likely never have been proposed." },
  { id: "cf2", label: "Retract o8 (Redis timeouts)",
    desc: "What if Bob's Redis timeout observation were a misreading — Redis is fine?",
    assumption: "o8",
    expected: "h1 (Redis→Stripe) collapses entirely. h4 (retry→Redis) becomes moot. But the token bug diagnosis survives (proved by o9 independently). Auth-side story intact; Redis/payment-side collapses." },
  { id: "cf3", label: "Retract o6 (3x traffic)",
    desc: "What if the 3x traffic reading were a monitoring glitch?",
    assumption: "o6",
    expected: "h3 (retry→3x traffic) loses its key evidence. h4 (retry→Redis exhaustion) is weakened because the mechanism depends on amplified traffic. The token bug is still the auth root cause (via o9), but the link to Redis exhaustion becomes speculative." },
];

// ============================================================
// UI COMPONENT
// ============================================================
const STATUS_COLORS = {
  active: "#16a34a", weakened: "#ca8a04", abandoned: "#dc2626",
  resolved: "#2563eb", none: "#94a3b8"
};

export default function App() {
  const [engine] = useState(() => new EpistemicEngine());
  const [currentTurn, setCurrentTurn] = useState(0);
  const [stateSnapshots, setStateSnapshots] = useState([]);
  const [selectedCf, setSelectedCf] = useState(null);
  const [cfResult, setCfResult] = useState(null);

  const advanceTurn = useCallback(() => {
    if (currentTurn >= PHASE2_TURNS.length) return;
    const turn = PHASE2_TURNS[currentTurn];
    turn.apply(engine);
    engine.turn = currentTurn + 1;
    const summary = engine.getSummary();
    const consistency = engine.checkConsistency();
    setStateSnapshots(prev => [...prev, {
      turn: turn.id, speaker: turn.speaker, text: turn.text,
      hypotheses: JSON.parse(JSON.stringify([...engine.hypotheses])),
      observations: JSON.parse(JSON.stringify([...engine.observations])),
      awareness: [...engine.awareness],
      history: [...engine.history],
      summary, consistency,
      deps: JSON.parse(JSON.stringify([...engine.dependencies].map(([k,v])=>[k,[...v]])))
    }]);
    setCurrentTurn(prev => prev + 1);
    setSelectedCf(null);
    setCfResult(null);
  }, [currentTurn, engine]);

  const runAllTurns = useCallback(() => {
    let t = currentTurn;
    const snaps = [...stateSnapshots];
    while (t < PHASE2_TURNS.length) {
      const turn = PHASE2_TURNS[t];
      turn.apply(engine);
      engine.turn = t + 1;
      snaps.push({
        turn: turn.id, speaker: turn.speaker, text: turn.text,
        hypotheses: JSON.parse(JSON.stringify([...engine.hypotheses])),
        observations: JSON.parse(JSON.stringify([...engine.observations])),
        awareness: [...engine.awareness],
        history: [...engine.history],
        summary: engine.getSummary(),
        consistency: engine.checkConsistency(),
        deps: JSON.parse(JSON.stringify([...engine.dependencies].map(([k,v])=>[k,[...v]])))
      });
      t++;
    }
    setStateSnapshots(snaps);
    setCurrentTurn(t);
  }, [currentTurn, engine, stateSnapshots]);

  const runCounterfactual = useCallback((cf) => {
    setSelectedCf(cf.id);
    const result = engine.retractAssumption(cf.assumption);
    setCfResult(result);
  }, [engine]);

  const latest = stateSnapshots.length > 0 ? stateSnapshots[stateSnapshots.length - 1] : null;

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", padding: 16, maxWidth: 1100, background: "#fafaf9", minHeight: "100vh" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet" />

      <div style={{ fontFamily: "'IBM Plex Sans', sans-serif", marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0, color: "#1a1a1a" }}>
          Symbolic Epistemic Engine
        </h1>
        <p style={{ fontSize: 13, color: "#666", margin: "4px 0 0" }}>
          Phase 2 Debugging Scenario — 13 turns — Hypothesis tracking, dependency propagation, counterfactual reasoning
        </p>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button onClick={advanceTurn} disabled={currentTurn >= PHASE2_TURNS.length}
          style={{ padding: "8px 16px", fontSize: 13, fontWeight: 600, border: "none", borderRadius: 4, cursor: "pointer",
            background: currentTurn >= PHASE2_TURNS.length ? "#d4d4d4" : "#1a1a1a", color: "white" }}>
          {currentTurn >= PHASE2_TURNS.length ? "All turns processed" : `Process T${currentTurn + 1}`}
        </button>
        <button onClick={runAllTurns} disabled={currentTurn >= PHASE2_TURNS.length}
          style={{ padding: "8px 16px", fontSize: 13, border: "1px solid #1a1a1a", borderRadius: 4, cursor: "pointer", background: "white" }}>
          Run All
        </button>
        <span style={{ fontSize: 12, color: "#888", alignSelf: "center" }}>
          Turn {currentTurn}/{PHASE2_TURNS.length}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Left: Conversation + Operations */}
        <div>
          <h3 style={{ fontFamily: "'IBM Plex Sans'", fontSize: 14, fontWeight: 700, margin: "0 0 8px", textTransform: "uppercase", letterSpacing: 1, color: "#888" }}>
            Conversation
          </h3>
          <div style={{ maxHeight: 500, overflow: "auto", background: "white", border: "1px solid #e5e5e5", borderRadius: 6, padding: 12 }}>
            {stateSnapshots.map((snap, i) => (
              <div key={i} style={{ marginBottom: 12, borderLeft: `3px solid ${snap.speaker === "Alice" ? "#3b82f6" : snap.speaker === "Bob" ? "#10b981" : "#f59e0b"}`, paddingLeft: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#888" }}>{snap.turn} — {snap.speaker}</div>
                <div style={{ fontSize: 12, color: "#333", margin: "2px 0" }}>{snap.text}</div>
                <div style={{ fontSize: 10, color: "#999", marginTop: 2 }}>
                  ops: {snap.history.filter(h => h.turn === snap.turn).map(h => h.op).join(", ")}
                </div>
              </div>
            ))}
            {currentTurn === 0 && <div style={{ color: "#aaa", fontSize: 12 }}>Click "Process T1" to begin...</div>}
          </div>
        </div>

        {/* Right: Engine State */}
        <div>
          <h3 style={{ fontFamily: "'IBM Plex Sans'", fontSize: 14, fontWeight: 700, margin: "0 0 8px", textTransform: "uppercase", letterSpacing: 1, color: "#888" }}>
            Engine State
          </h3>
          <div style={{ background: "white", border: "1px solid #e5e5e5", borderRadius: 6, padding: 12, maxHeight: 500, overflow: "auto" }}>
            {latest ? (
              <>
                {/* Hypotheses */}
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "#888", marginBottom: 4 }}>HYPOTHESES</div>
                  {latest.hypotheses.length === 0 ? (
                    <div style={{ fontSize: 11, color: "#bbb" }}>No hypotheses yet</div>
                  ) : latest.hypotheses.map(([id, h]) => (
                    <div key={id} style={{ fontSize: 11, padding: "4px 6px", marginBottom: 3, background: "#f8f8f8", borderRadius: 3, borderLeft: `3px solid ${STATUS_COLORS[h.status] || "#ccc"}` }}>
                      <span style={{ fontWeight: 600 }}>{id}</span>
                      <span style={{ color: STATUS_COLORS[h.status], fontWeight: 600, marginLeft: 6, fontSize: 10, textTransform: "uppercase" }}>{h.status}</span>
                      <div style={{ color: "#555", marginTop: 1 }}>{h.content}</div>
                      {h.undermines?.length > 0 && <div style={{ color: "#dc2626", fontSize: 10 }}>⚡ Undermined: {h.undermines.map(u => u.evidence).join("; ")}</div>}
                      {h.abandoned_reason && <div style={{ color: "#dc2626", fontSize: 10 }}>✗ Abandoned (T{h.abandoned_turn}): {h.abandoned_reason}</div>}
                    </div>
                  ))}
                </div>
                {/* Dependencies */}
                {latest.deps.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "#888", marginBottom: 4 }}>DEPENDENCIES (Dep)</div>
                    {latest.deps.map(([id, deps]) => (
                      <div key={id} style={{ fontSize: 10, color: "#555", padding: "2px 6px" }}>
                        <span style={{ fontWeight: 600 }}>{id}</span> → {"{"}{deps.join(", ")}{"}"}
                      </div>
                    ))}
                  </div>
                )}
                {/* Unexplained */}
                {latest.summary.unexplained.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "#ca8a04", marginBottom: 4 }}>⚠ UNEXPLAINED OBSERVATIONS</div>
                    {latest.summary.unexplained.map(([id, o]) => (
                      <div key={id} style={{ fontSize: 10, color: "#92400e", padding: "2px 6px" }}>{id}: {o.content}</div>
                    ))}
                  </div>
                )}
                {/* Consistency */}
                {latest.consistency.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "#dc2626", marginBottom: 4 }}>🔍 ENGINE CHECKS</div>
                    {latest.consistency.map((c, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#991b1b", padding: "2px 6px" }}>{c.msg}</div>
                    ))}
                  </div>
                )}
                {/* Observations count */}
                <div style={{ fontSize: 10, color: "#999", marginTop: 8 }}>
                  {latest.observations.length} observations | {latest.awareness.length} propositions in awareness
                </div>
              </>
            ) : (
              <div style={{ color: "#aaa", fontSize: 12 }}>Engine empty — process turns to populate</div>
            )}
          </div>
        </div>
      </div>

      {/* Counterfactual Panel */}
      {currentTurn >= PHASE2_TURNS.length && (
        <div style={{ marginTop: 20, background: "#fffbeb", border: "1px solid #fbbf24", borderRadius: 6, padding: 16 }}>
          <h3 style={{ fontFamily: "'IBM Plex Sans'", fontSize: 14, fontWeight: 700, margin: "0 0 8px", color: "#92400e" }}>
            Counterfactual Reasoning — Dep() and Affected()
          </h3>
          <p style={{ fontSize: 12, color: "#78716c", margin: "0 0 12px" }}>
            Retract an assumption and see which conclusions are affected. This is the capability that requires the symbolic engine.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {COUNTERFACTUALS.map(cf => (
              <button key={cf.id} onClick={() => runCounterfactual(cf)}
                style={{ padding: "6px 12px", fontSize: 12, border: `1px solid ${selectedCf === cf.id ? "#dc2626" : "#d6d3d1"}`,
                  borderRadius: 4, cursor: "pointer", background: selectedCf === cf.id ? "#fef2f2" : "white",
                  fontWeight: selectedCf === cf.id ? 600 : 400 }}>
                {cf.label}
              </button>
            ))}
          </div>
          {cfResult && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#991b1b", marginBottom: 6 }}>
                Retracting: {cfResult.retracted}
              </div>
              <div style={{ fontSize: 12, marginBottom: 4 }}>
                <span style={{ fontWeight: 600, color: "#dc2626" }}>Affected</span> ({cfResult.affected_hypotheses.length}):
              </div>
              {cfResult.flagged_for_review.map(h => (
                <div key={h.id} style={{ fontSize: 11, color: "#991b1b", padding: "3px 8px", background: "#fef2f2", borderRadius: 3, marginBottom: 2 }}>
                  <span style={{ fontWeight: 600 }}>{h.id}</span> [{h.status}]: {h.content}
                  <div style={{ fontSize: 10, color: "#b91c1c" }}>→ {h.reason}</div>
                </div>
              ))}
              <div style={{ fontSize: 12, marginTop: 8, marginBottom: 4 }}>
                <span style={{ fontWeight: 600, color: "#16a34a" }}>Unaffected</span> ({cfResult.unaffected_hypotheses.length}):
              </div>
              {cfResult.unaffected_hypotheses.map(h => (
                <div key={h.id} style={{ fontSize: 11, color: "#166534", padding: "3px 8px", background: "#f0fdf4", borderRadius: 3, marginBottom: 2 }}>
                  <span style={{ fontWeight: 600 }}>{h.id}</span> [{h.status}]: {h.content}
                </div>
              ))}
              <div style={{ marginTop: 10, fontSize: 11, color: "#78716c", background: "#f5f5f4", padding: 8, borderRadius: 4 }}>
                <strong>Expected:</strong> {COUNTERFACTUALS.find(c => c.id === selectedCf)?.expected}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
