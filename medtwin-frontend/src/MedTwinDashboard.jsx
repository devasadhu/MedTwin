import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell } from "recharts";

const API = "http://127.0.0.1:8000";

const C = {
  bg: "#060910", surface: "#0d1117", card: "#111820", border: "#1e2d3d",
  accent: "#00d4ff", green: "#00ff9d", red: "#ff4d6d", amber: "#ffb347",
  purple: "#bd93f9", text: "#e6edf3", muted: "#8b949e",
};

const SCENARIO_COLORS = { none:"#ff4d6d", metformin:"#00d4ff", lifestyle:"#00ff9d", combined:"#bd93f9", statin:"#ffb347", sleep_therapy:"#f9d93e" };
const SCENARIO_LABELS = { none:"No Treatment", metformin:"Metformin", lifestyle:"Lifestyle Only", combined:"Combined", statin:"Statin", sleep_therapy:"Sleep Therapy" };

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function riskColor(score) {
  if (score >= 0.55) return C.red;
  if (score >= 0.35) return C.amber;
  return C.green;
}

function RiskGauge({ value, label }) {
  const pct = Math.round(value * 100);
  const color = riskColor(value);
  const circ = 2 * Math.PI * 54;
  const offset = circ * (1 - value);
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
      <svg width={130} height={130} viewBox="0 0 130 130">
        <circle cx={65} cy={65} r={54} fill="none" stroke={C.border} strokeWidth={10}/>
        <circle cx={65} cy={65} r={54} fill="none" stroke={color} strokeWidth={10}
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 65 65)" style={{ transition:"stroke-dashoffset 1.2s ease" }}/>
        <text x={65} y={60} textAnchor="middle" fill={color} fontSize={24} fontWeight={700} fontFamily="'Courier New',monospace">{pct}%</text>
        <text x={65} y={78} textAnchor="middle" fill={C.muted} fontSize={10} fontFamily="Georgia,serif">
          {pct>=75?"CRITICAL":pct>=55?"HIGH":pct>=35?"MODERATE":"LOW"}
        </text>
      </svg>
      <span style={{ color:C.muted, fontSize:11, letterSpacing:2, textTransform:"uppercase" }}>{label}</span>
    </div>
  );
}

function Card({ children, style }) {
  return <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:12, padding:24, ...style }}>{children}</div>;
}

function SectionTitle({ children }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:20 }}>
      <div style={{ width:3, height:18, background:C.accent, borderRadius:2 }}/>
      <h2 style={{ margin:0, fontSize:11, letterSpacing:3, textTransform:"uppercase", color:C.muted, fontFamily:"Georgia,serif", fontWeight:400 }}>{children}</h2>
    </div>
  );
}

function ShapBar({ label, value }) {
  const w = Math.min((Math.abs(value) / 1.5) * 100, 100);
  const color = value > 0 ? C.red : C.green;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:10 }}>
      <span style={{ width:200, fontSize:12, color:C.text, fontFamily:"'Courier New',monospace", textAlign:"right" }}>{label}</span>
      <div style={{ flex:1, height:8, background:C.border, borderRadius:4, overflow:"hidden" }}>
        <div style={{ width:`${w}%`, height:"100%", background:color, borderRadius:4, transition:"width 0.8s ease" }}/>
      </div>
      <span style={{ width:60, fontSize:11, color, fontFamily:"'Courier New',monospace", textAlign:"right" }}>
        {value>0?"+":""}{value.toFixed(4)}
      </span>
    </div>
  );
}

function Tag({ children, color }) {
  return <span style={{ background:color+"22", color, border:`1px solid ${color}44`, borderRadius:4, padding:"2px 8px", fontSize:11, letterSpacing:1, fontFamily:"'Courier New',monospace" }}>{children}</span>;
}

const btn = (color=C.accent, disabled=false) => ({
  background: disabled ? C.border : color+"22",
  border: `1px solid ${disabled?C.border:color}`,
  color: disabled ? C.muted : color,
  padding:"10px 20px", borderRadius:8, cursor:disabled?"not-allowed":"pointer",
  fontSize:12, letterSpacing:2, textTransform:"uppercase", fontFamily:"Georgia,serif",
  transition:"all 0.2s",
});

const inputStyle = {
  background:C.surface, border:`1px solid ${C.border}`, color:C.text,
  borderRadius:6, padding:"8px 12px", fontSize:13,
  fontFamily:"'Courier New',monospace", outline:"none", width:"100%", boxSizing:"border-box",
};

const DEFAULT = {
  name:"John Doe", age:47, gender:"male", glucose:158, bp_systolic:82,
  bmi:33.6, cholesterol:210, hba1c:7.2, sleep_hours:6, stress_level:65,
  activity_score:30, pregnancies:3, skin_thickness:32, insulin:0,
  diabetes_pedigree:0.627,
};

const SUGGESTED_QUESTIONS = [
  "What is this patient's biggest health risk?",
  "Which intervention would you recommend first?",
  "What does a glucose of 158 mean clinically?",
  "How serious is an HbA1c of 7.2%?",
  "What lifestyle changes would help most?",
  "Explain the composite risk score to me.",
];

// ── RL tab constants ──────────────────────────────────────────────────────────
const RL_AGENT_COLORS = { dqn: C.accent, ppo: C.purple };
const RL_AGENT_LABELS = { dqn: "DQN (Dueling Double)", ppo: "PPO (Clipped Surrogate)" };

const RL_ALGO_INFO = {
  dqn: {
    name: "Deep Q-Network (DQN)",
    type: "Off-policy",
    arch: "Dueling Double DQN",
    details: [
      "Separates value V(s) and advantage A(s,a) streams",
      "Experience replay buffer (20K transitions)",
      "Soft target network update (τ = 0.005)",
      "ε-greedy exploration with linear decay",
      "Smooth L1 (Huber) loss for stability",
    ],
    color: C.accent,
    icon: "⚡",
  },
  ppo: {
    name: "Proximal Policy Optimization (PPO)",
    type: "On-policy",
    arch: "Actor-Critic + GAE",
    details: [
      "Clipped surrogate objective (ε = 0.2)",
      "Generalized Advantage Estimation (λ = 0.95)",
      "Shared backbone, separate actor/critic heads",
      "Entropy bonus for exploration (coef = 0.01)",
      "4 mini-batch update epochs per rollout",
    ],
    color: C.purple,
    icon: "🔮",
  },
};

export default function MedTwinDashboard() {
  const [tab, setTab]             = useState("create");
  const [p, setP]                 = useState(DEFAULT);
  const [patientId, setPid]       = useState(null);
  const [risk, setRisk]           = useState(null);
  const [explain, setExplain]     = useState(null);
  const [scenarios, setScenarios] = useState(null);
  const [trajectory, setTraj]     = useState(null);
  const [loading, setLoading]     = useState({});
  const [error, setError]         = useState(null);
  const [health, setHealth]       = useState(null);
  const [log, setLog]             = useState([]);

  // RL state
  const [rlAgent, setRlAgent]         = useState("dqn");
  const [rlResult, setRlResult]       = useState(null);
  const [rlCompare, setRlCompare]     = useState(null); // both agents side by side

  // Chat state
  const [chatMessages, setChatMessages] = useState([{
    role: "assistant",
    content: "Hello! I'm MedTwin AI. Create a patient profile first, then ask me anything about their health risks, interventions, or biomarkers.",
    ts: new Date().toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" }),
  }]);
  const [chatInput, setChatInput]     = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  const addLog = (msg, type="info") => setLog(l => [...l.slice(-30), { msg, type }]);

  useEffect(() => {
    apiFetch("/health")
      .then(d => { setHealth(d); addLog("Backend connected ✓","ok"); })
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior:"smooth" });
  }, [chatMessages]);

  async function run(key, fn) {
    setLoading(l => ({...l,[key]:true})); setError(null);
    try { await fn(); }
    catch(e) { setError(e.message); addLog(e.message,"error"); }
    finally { setLoading(l => ({...l,[key]:false})); }
  }

  const patientPayload = () => ({
    name: p.name, age: Math.round(p.age), gender: p.gender,
    glucose: +p.glucose, bp_systolic: +p.bp_systolic, bmi: +p.bmi,
    cholesterol: +p.cholesterol, hba1c: +p.hba1c, sleep_hours: +p.sleep_hours,
    stress_level: +p.stress_level, activity_score: +p.activity_score,
    pregnancies: Math.round(p.pregnancies), skin_thickness: +p.skin_thickness,
    insulin: +p.insulin, diabetes_pedigree: +p.diabetes_pedigree,
    diseases: [], medications: [],
  });

  async function createPatient() {
    await run("create", async () => {
      const d = await apiFetch("/patient/create", { method:"POST", body:JSON.stringify(patientPayload()) });
      setPid(d.patient_id);
      addLog(`Patient created: ${d.patient_id}`,"ok");
      setTab("risk");
    });
  }

  async function predictRisk() {
    await run("risk", async () => {
      const payload = patientId ? { patient_id: patientId } : { patient: patientPayload() };
      const d = await apiFetch("/predict/risk", { method:"POST", body:JSON.stringify(payload) });
      setRisk(d);
      addLog(`Diabetes: ${Math.round(d.diabetes.risk_score*100)}% | CV: ${Math.round(d.cardiovascular.risk_score*100)}%`,"ok");
    });
  }

  async function explainRisk() {
    await run("explain", async () => {
      const payload = { model_type:"diabetes", ...(patientId ? {patient_id:patientId} : {patient:patientPayload()}) };
      const d = await apiFetch("/explain", { method:"POST", body:JSON.stringify(payload) });
      setExplain(d);
      addLog("SHAP explanation loaded","ok");
    });
  }

  async function runScenarios() {
    await run("scenarios", async () => {
      const payload = patientId ? { patient_id:patientId } : { patient:patientPayload() };
      const d = await apiFetch("/simulate/scenarios", { method:"POST", body:JSON.stringify(payload) });
      setScenarios(d);
      addLog(`Best: ${d.best_intervention}`,"ok");
    });
  }

  async function runTrajectory() {
    await run("traj", async () => {
      const payload = {
        ...(patientId ? {patient_id:patientId} : {patient:patientPayload()}),
        intervention:"lifestyle", checkpoints:[0,3,6,9,12,18,24],
      };
      const d = await apiFetch("/simulate/trajectory", { method:"POST", body:JSON.stringify(payload) });
      setTraj(d);
      addLog(`Trend: ${d.summary.trend} | Δ${(d.summary.risk_delta*100).toFixed(1)}%`,"ok");
    });
  }

  // ── RL functions ────────────────────────────────────────────────────────────
  async function runRlAgent(agentType) {
    await run(`rl_${agentType}`, async () => {
      const payload = {
        agent_type: agentType,
        ...(patientId ? {patient_id:patientId} : {patient:patientPayload()}),
      };
      const d = await apiFetch("/optimize/rl", { method:"POST", body:JSON.stringify(payload) });
      setRlResult(d);
      addLog(`${agentType.toUpperCase()} · ${d.total_risk_reduction_pct}% reduction`, "ok");
    });
  }

  async function runRlCompare() {
    await run("rl_compare", async () => {
      const base = patientId ? {patient_id:patientId} : {patient:patientPayload()};
      const [dqn, ppo] = await Promise.all([
        apiFetch("/optimize/rl", { method:"POST", body:JSON.stringify({...base, agent_type:"dqn"}) }),
        apiFetch("/optimize/rl", { method:"POST", body:JSON.stringify({...base, agent_type:"ppo"}) }),
      ]);
      setRlCompare({ dqn, ppo });
      addLog(`DQN −${dqn.total_risk_reduction_pct}% | PPO −${ppo.total_risk_reduction_pct}%`, "ok");
    });
  }

  // ── Chat ─────────────────────────────────────────────────────────────────────
  async function sendChat(questionOverride) {
    const question = (questionOverride || chatInput).trim();
    if (!question) return;
    const userMsg = { role:"user", content:question, ts:new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}) };
    setChatMessages(m => [...m, userMsg]);
    setChatInput("");
    setChatLoading(true);
    const history = chatMessages
      .filter(m => m.role !== "assistant" || m.content !== chatMessages[0].content)
      .map(m => ({ role:m.role, content:m.content }));
    try {
      const payload = {
        question, history,
        ...(patientId ? {patient_id:patientId} : {patient:patientPayload()}),
        ...(risk ? {risk_context:risk} : {}),
      };
      const d = await apiFetch("/chat", { method:"POST", body:JSON.stringify(payload) });
      setChatMessages(m => [...m, { role:"assistant", content:d.answer, tokens:d.tokens, ts:new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}) }]);
      addLog(`Chat reply · ${d.tokens} tokens`, "ok");
    } catch(e) {
      setChatMessages(m => [...m, { role:"assistant", content:`⚠ Error: ${e.message}`, ts:new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}), isError:true }]);
      addLog(e.message, "error");
    } finally { setChatLoading(false); }
  }

  function handleChatKey(e) {
    if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  }

  // ── Chart data ────────────────────────────────────────────────────────────────
  const scenarioChartData = scenarios
    ? [0,3,6,12,18,24].map(m => {
        const row = { month:`M${m}` };
        Object.keys(scenarios.scenarios).forEach(k => {
          const val = scenarios.scenarios[k].trajectory[m]?.composite_risk;
          row[SCENARIO_LABELS[k]] = val !== undefined ? +(val*100).toFixed(1) : null;
        });
        return row;
      })
    : [];

  const trajChartData = trajectory
    ? Object.entries(trajectory.trajectory).map(([m, state]) => ({
        month:`M${m}`,
        Glucose: +state.glucose?.toFixed(1),
        "BP Systolic": +state.bp_systolic?.toFixed(1),
        "HbA1c %": +state.hba1c?.toFixed(2),
        "Risk %": +(state.composite_risk*100).toFixed(1),
      }))
    : [];

  // RL trajectory chart: months on x axis, risk on y
  const rlChartData = rlResult
    ? [{ month:"M0", risk: +(rlResult.base_risk*100).toFixed(1) },
       ...rlResult.optimal_sequence.map(s => ({
         month: `M${s.month}`,
         risk:  +(s.risk*100).toFixed(1),
         intervention: s.intervention,
       }))]
    : [];

  // RL compare chart data
  const rlCompareChartData = rlCompare
    ? [{ month:"M0", DQN: +(rlCompare.dqn.base_risk*100).toFixed(1), PPO: +(rlCompare.ppo.base_risk*100).toFixed(1) },
       ...rlCompare.dqn.optimal_sequence.map((s, i) => ({
         month: `M${s.month}`,
         DQN:   +(s.risk*100).toFixed(1),
         PPO:   +(rlCompare.ppo.optimal_sequence[i]?.risk*100).toFixed(1),
       }))]
    : [];

  const TABS = [
    {id:"create",   label:"01 · Patient"},
    {id:"risk",     label:"02 · Risk"},
    {id:"simulate", label:"03 · Scenarios"},
    {id:"timeline", label:"04 · Timeline"},
    {id:"rl",       label:"06 · RL Agent", special: C.amber},
    {id:"chat",     label:"05 · AI Chat",  special: C.purple},
  ];

  const field = (label, key, type="number") => (
    <div key={key}>
      <div style={{ fontSize:10, color:C.muted, letterSpacing:1, marginBottom:4, textTransform:"uppercase" }}>{label}</div>
      {key==="gender"
        ? <select value={p[key]} onChange={e=>setP(x=>({...x,[key]:e.target.value}))} style={inputStyle}>
            <option value="male">Male</option><option value="female">Female</option><option value="other">Other</option>
          </select>
        : <input type={type} value={p[key]}
            onChange={e=>setP(x=>({...x,[key]:type==="number"?parseFloat(e.target.value)||0:e.target.value}))}
            style={inputStyle}/>
      }
    </div>
  );

  return (
    <div style={{ minHeight:"100vh", background:C.bg, color:C.text, fontFamily:"Georgia,serif" }}>

      {/* Header */}
      <div style={{ borderBottom:`1px solid ${C.border}`, padding:"0 32px" }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", maxWidth:1400, margin:"0 auto", height:64 }}>
          <div style={{ display:"flex", alignItems:"center", gap:16 }}>
            <svg width={32} height={32} viewBox="0 0 32 32">
              <circle cx={16} cy={16} r={14} fill="none" stroke={C.accent} strokeWidth={2}/>
              <circle cx={16} cy={16} r={5} fill={C.accent}/>
              <line x1={16} y1={2} x2={16} y2={10} stroke={C.accent} strokeWidth={1.5}/>
              <line x1={16} y1={22} x2={16} y2={30} stroke={C.accent} strokeWidth={1.5}/>
              <line x1={2} y1={16} x2={10} y2={16} stroke={C.accent} strokeWidth={1.5}/>
              <line x1={22} y1={16} x2={30} y2={16} stroke={C.accent} strokeWidth={1.5}/>
            </svg>
            <div>
              <div style={{ fontSize:15, fontWeight:700, letterSpacing:4, color:C.accent, fontFamily:"'Courier New',monospace" }}>MEDTWIN AI</div>
              <div style={{ fontSize:9, color:C.muted, letterSpacing:2 }}>DIGITAL TWIN PATIENT SIMULATOR</div>
            </div>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:16 }}>
            {patientId && <Tag color={C.green}>ID · {patientId}</Tag>}
            <div style={{ display:"flex", alignItems:"center", gap:6 }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background:health?C.green:C.red, boxShadow:`0 0 6px ${health?C.green:C.red}` }}/>
              <span style={{ fontSize:10, color:C.muted, letterSpacing:1 }}>{health?"API LIVE":"OFFLINE"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom:`1px solid ${C.border}`, padding:"0 32px" }}>
        <div style={{ display:"flex", maxWidth:1400, margin:"0 auto" }}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              background:"none", border:"none",
              borderBottom:`2px solid ${tab===t.id?(t.special||C.accent):"transparent"}`,
              color: tab===t.id ? (t.special||C.accent) : (t.special ? t.special+"99" : C.muted),
              padding:"16px 20px", cursor:"pointer", fontSize:11, letterSpacing:2,
              textTransform:"uppercase", fontFamily:"Georgia,serif", transition:"all 0.2s",
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth:1400, margin:"0 auto", padding:32 }}>

        {error && (
          <div style={{ background:C.red+"11", border:`1px solid ${C.red}33`, borderRadius:8, padding:"12px 16px", marginBottom:24, fontSize:13, color:C.red }}>
            ⚠ {error}
          </div>
        )}

        {/* ── TAB 1: Create ── */}
        {tab==="create" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 360px", gap:24 }}>
            <Card>
              <SectionTitle>Patient Profile</SectionTitle>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
                {field("Full Name","name","text")}
                {field("Age","age")}
                {field("Gender","gender")}
                {field("Glucose (mg/dL)","glucose")}
                {field("BP Systolic (mmHg)","bp_systolic")}
                {field("BMI","bmi")}
                {field("Cholesterol (mg/dL)","cholesterol")}
                {field("HbA1c (%)","hba1c")}
                {field("Sleep Hours","sleep_hours")}
                {field("Stress Level (0–100)","stress_level")}
                {field("Activity Score (0–100)","activity_score")}
                {field("Pregnancies","pregnancies")}
                {field("Skin Thickness","skin_thickness")}
                {field("Insulin","insulin")}
                {field("Diabetes Pedigree","diabetes_pedigree")}
              </div>
              <div style={{ marginTop:24 }}>
                <button onClick={createPatient} disabled={loading.create} style={btn(C.accent,loading.create)}>
                  {loading.create?"Creating...":"→ Create Digital Twin"}
                </button>
              </div>
            </Card>
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
              <Card>
                <SectionTitle>System Status</SectionTitle>
                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  {health ? Object.entries({
                    "Models Loaded": health.models_loaded?.length,
                    "Patients Stored": health.patients_stored,
                    "Status": health.status?.toUpperCase(),
                    "Groq LLM": health.groq_enabled ? "ENABLED ✓" : "DISABLED ✗",
                  }).map(([k,v])=>(
                    <div key={k} style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
                      <span style={{ color:C.muted }}>{k}</span>
                      <span style={{ color:k==="Groq LLM"?(health.groq_enabled?C.green:C.red):C.accent, fontFamily:"'Courier New',monospace" }}>{v}</span>
                    </div>
                  )) : <span style={{ color:C.red, fontSize:12 }}>Backend offline — run medtwin_phase3_backend.py</span>}
                </div>
              </Card>
              <Card style={{ flex:1 }}>
                <SectionTitle>Activity Log</SectionTitle>
                <div style={{ fontFamily:"'Courier New',monospace", fontSize:11, display:"flex", flexDirection:"column", gap:5, maxHeight:320, overflowY:"auto" }}>
                  {log.length===0 && <span style={{ color:C.border }}>— awaiting actions —</span>}
                  {log.map((l,i)=>(
                    <div key={i} style={{ color:l.type==="error"?C.red:l.type==="ok"?C.green:C.muted }}>
                      <span style={{ color:C.border }}>› </span>{l.msg}
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* ── TAB 2: Risk ── */}
        {tab==="risk" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
            <Card>
              <SectionTitle>Risk Scoring</SectionTitle>
              <p style={{ fontSize:13, color:C.muted, marginBottom:20 }}>XGBoost models trained on Pima Diabetes + Cleveland Heart datasets.</p>
              <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
                <button onClick={predictRisk} disabled={loading.risk} style={btn(C.accent,loading.risk)}>
                  {loading.risk?"Scoring...":"→ Predict Risk"}
                </button>
                <button onClick={explainRisk} disabled={loading.explain} style={btn(C.purple,loading.explain)}>
                  {loading.explain?"Explaining...":"→ Explain (SHAP)"}
                </button>
              </div>
              {risk && (
                <div style={{ marginTop:28 }}>
                  <div style={{ display:"flex", justifyContent:"space-around" }}>
                    <RiskGauge value={risk.diabetes.risk_score} label="Diabetes"/>
                    <RiskGauge value={risk.cardiovascular.risk_score} label="Cardiovascular"/>
                  </div>
                  <div style={{ marginTop:16, display:"flex", justifyContent:"center", gap:8, flexWrap:"wrap" }}>
                    <Tag color={riskColor(risk.composite.risk_score)}>Composite · {Math.round(risk.composite.risk_score*100)}%</Tag>
                    <Tag color={riskColor(risk.composite.risk_score)}>{risk.composite.risk_level}</Tag>
                  </div>
                </div>
              )}
            </Card>
            <Card>
              <SectionTitle>SHAP Explanation</SectionTitle>
              {!explain && <p style={{ color:C.muted, fontSize:13 }}>Click "Explain (SHAP)" to see which factors drive the risk.</p>}
              {explain && (
                <>
                  <p style={{ fontSize:13, color:C.text, lineHeight:1.7, marginBottom:20 }}>{explain.explanation}</p>
                  {explain.top_factors?.map((f,i)=><ShapBar key={i} label={f.feature} value={f.impact}/>)}
                  <div style={{ marginTop:12, padding:"10px 14px", background:C.surface, borderRadius:8, fontSize:12, color:C.amber }}>💡 {explain.recommendation}</div>
                  <div style={{ marginTop:10, fontSize:10, color:C.muted, fontFamily:"'Courier New',monospace" }}>Red ↑ increases risk · Green ↓ decreases risk</div>
                </>
              )}
            </Card>
          </div>
        )}

        {/* ── TAB 3: Scenarios ── */}
        {tab==="simulate" && (
          <div style={{ display:"flex", flexDirection:"column", gap:24 }}>
            <Card>
              <SectionTitle>Scenario Comparison</SectionTitle>
              <p style={{ fontSize:13, color:C.muted, marginBottom:20 }}>Compare all 6 interventions over 24 months side-by-side.</p>
              <button onClick={runScenarios} disabled={loading.scenarios} style={btn(C.accent,loading.scenarios)}>
                {loading.scenarios?"Simulating...":"→ Run All Scenarios"}
              </button>
            </Card>
            {scenarios && (
              <>
                <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
                  <Tag color={C.green}>Best: {SCENARIO_LABELS[scenarios.best_intervention]}</Tag>
                  <Tag color={C.red}>Worst: {SCENARIO_LABELS[scenarios.worst_intervention]}</Tag>
                  <Tag color={C.accent}>Baseline: {Math.round(scenarios.patient_baseline_risk*100)}%</Tag>
                </div>
                <Card>
                  <SectionTitle>Risk Evolution (24 months)</SectionTitle>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={scenarioChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                      <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                      <YAxis tick={{ fill:C.muted, fontSize:11 }} unit="%" domain={[20,80]}/>
                      <Tooltip contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}/>
                      <Legend wrapperStyle={{ fontSize:11 }}/>
                      {Object.keys(SCENARIO_COLORS).map(k=>(
                        <Line key={k} type="monotone" dataKey={SCENARIO_LABELS[k]} stroke={SCENARIO_COLORS[k]} strokeWidth={2} dot={false}/>
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
                <Card>
                  <SectionTitle>Outcome Table</SectionTitle>
                  <div style={{ overflowX:"auto" }}>
                    <table style={{ width:"100%", borderCollapse:"collapse", fontFamily:"'Courier New',monospace", fontSize:12 }}>
                      <thead>
                        <tr>{["Rank","Intervention","Risk @M0","Risk @M24","Δ Risk","Glucose","BP","BMI"].map(h=>(
                          <th key={h} style={{ textAlign:"left", padding:"8px 14px", color:C.muted, borderBottom:`1px solid ${C.border}`, fontWeight:400, letterSpacing:1 }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {scenarios.ranking?.map((r)=>{
                          const sc = scenarios.scenarios[r.intervention];
                          const r0 = sc.trajectory[0]?.composite_risk;
                          const delta = sc.risk_delta;
                          return (
                            <tr key={r.intervention} style={{ borderBottom:`1px solid ${C.border}22` }}>
                              <td style={{ padding:"10px 14px", color:C.muted }}>#{r.rank}</td>
                              <td style={{ padding:"10px 14px", color:SCENARIO_COLORS[r.intervention] }}>{SCENARIO_LABELS[r.intervention]}</td>
                              <td style={{ padding:"10px 14px", color:C.text }}>{r0!==undefined?Math.round(r0*100)+"%":"—"}</td>
                              <td style={{ padding:"10px 14px", color:riskColor(r.final_risk) }}>{Math.round(r.final_risk*100)}%</td>
                              <td style={{ padding:"10px 14px", color:delta>0?C.red:C.green, fontWeight:700 }}>{delta>0?"↑":"↓"} {Math.abs(delta*100).toFixed(1)}%</td>
                              <td style={{ padding:"10px 14px", color:C.text }}>{sc.glucose_at_end?.toFixed(0)}</td>
                              <td style={{ padding:"10px 14px", color:C.text }}>{sc.bp_at_end?.toFixed(0)}</td>
                              <td style={{ padding:"10px 14px", color:C.text }}>{sc.bmi_at_end?.toFixed(1)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </>
            )}
          </div>
        )}

        {/* ── TAB 4: Timeline ── */}
        {tab==="timeline" && (
          <div style={{ display:"flex", flexDirection:"column", gap:24 }}>
            <Card>
              <SectionTitle>Health Trajectory — Lifestyle Intervention</SectionTitle>
              <p style={{ fontSize:13, color:C.muted, marginBottom:20 }}>Simulates the "Lifestyle Only" intervention over 24 months.</p>
              <button onClick={runTrajectory} disabled={loading.traj} style={btn(C.green,loading.traj)}>
                {loading.traj?"Simulating...":"→ Run Trajectory"}
              </button>
              {trajectory && (
                <div style={{ display:"flex", gap:12, marginTop:16, flexWrap:"wrap" }}>
                  <Tag color={C.accent}>Start: {Math.round(trajectory.summary.risk_at_start*100)}%</Tag>
                  <Tag color={C.green}>End: {Math.round(trajectory.summary.risk_at_end*100)}%</Tag>
                  <Tag color={trajectory.summary.trend==="improving"?C.green:C.red}>{trajectory.summary.trend} · Δ{(trajectory.summary.risk_delta*100).toFixed(1)}%</Tag>
                </div>
              )}
            </Card>
            {trajectory && (
              <>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
                  <Card>
                    <SectionTitle>Glucose & BP Over Time</SectionTitle>
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={trajChartData}>
                        <defs>
                          <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="95%" stopColor={C.accent} stopOpacity={0}/></linearGradient>
                          <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.amber} stopOpacity={0.3}/><stop offset="95%" stopColor={C.amber} stopOpacity={0}/></linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                        <YAxis tick={{ fill:C.muted, fontSize:11 }}/>
                        <Tooltip contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}/>
                        <Legend wrapperStyle={{ fontSize:11 }}/>
                        <Area type="monotone" dataKey="Glucose" stroke={C.accent} fill="url(#g1)" strokeWidth={2} dot={false}/>
                        <Area type="monotone" dataKey="BP Systolic" stroke={C.amber} fill="url(#g2)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <SectionTitle>Composite Risk Trajectory</SectionTitle>
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={trajChartData}>
                        <defs><linearGradient id="g3" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.red} stopOpacity={0.3}/><stop offset="95%" stopColor={C.red} stopOpacity={0}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                        <YAxis tick={{ fill:C.muted, fontSize:11 }} unit="%"/>
                        <Tooltip contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}/>
                        <Area type="monotone" dataKey="Risk %" stroke={C.red} fill="url(#g3)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </Card>
                </div>
                <Card>
                  <SectionTitle>HbA1c Trend</SectionTitle>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={trajChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                      <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                      <YAxis tick={{ fill:C.muted, fontSize:11 }} unit="%"/>
                      <Tooltip contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}/>
                      <Line type="monotone" dataKey="HbA1c %" stroke={C.purple} strokeWidth={2} dot={{ fill:C.purple, r:3 }}/>
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </>
            )}
          </div>
        )}

        {/* ── TAB 6: RL Agent ── */}
        {tab==="rl" && (
          <div style={{ display:"flex", flexDirection:"column", gap:24 }}>

            {/* Algorithm selector + run buttons */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
              {Object.entries(RL_ALGO_INFO).map(([key, info]) => (
                <Card key={key} style={{
                  border:`1px solid ${rlAgent===key ? info.color+"66" : C.border}`,
                  cursor:"pointer", transition:"all 0.2s",
                  background: rlAgent===key ? info.color+"08" : C.card,
                }} onClick={() => setRlAgent(key)}>
                  <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", marginBottom:14 }}>
                    <div style={{ display:"flex", alignItems:"center", gap:12 }}>
                      <span style={{ fontSize:24 }}>{info.icon}</span>
                      <div>
                        <div style={{ fontSize:13, fontWeight:700, color:info.color, fontFamily:"'Courier New',monospace", letterSpacing:1 }}>{info.name}</div>
                        <div style={{ fontSize:10, color:C.muted, letterSpacing:1, marginTop:2 }}>{info.type} · {info.arch}</div>
                      </div>
                    </div>
                    <div style={{
                      width:14, height:14, borderRadius:"50%",
                      background: rlAgent===key ? info.color : "transparent",
                      border:`2px solid ${info.color}`,
                      transition:"all 0.2s",
                    }}/>
                  </div>
                  <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                    {info.details.map((d, i) => (
                      <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:8, fontSize:11, color:C.muted, lineHeight:1.5 }}>
                        <span style={{ color:info.color, marginTop:1 }}>›</span>
                        <span>{d}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              ))}
            </div>

            {/* Action buttons */}
            <Card>
              <SectionTitle>Run RL Optimizer</SectionTitle>
              <p style={{ fontSize:13, color:C.muted, marginBottom:20, lineHeight:1.7 }}>
                The agent was trained on 600 episodes across diverse patient populations.
                It learned a treatment policy by maximizing cumulative reward (−composite_risk) over a 24-month horizon.
              </p>
              <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
                <button
                  onClick={() => runRlAgent(rlAgent)}
                  disabled={loading[`rl_${rlAgent}`]}
                  style={btn(RL_AGENT_COLORS[rlAgent], loading[`rl_${rlAgent}`])}
                >
                  {loading[`rl_${rlAgent}`] ? "Running..." : `→ Run ${rlAgent.toUpperCase()} Agent`}
                </button>
                <button
                  onClick={runRlCompare}
                  disabled={loading.rl_compare}
                  style={btn(C.amber, loading.rl_compare)}
                >
                  {loading.rl_compare ? "Comparing..." : "→ Compare DQN vs PPO"}
                </button>
              </div>

              {/* Summary tags */}
              {rlResult && (
                <div style={{ display:"flex", gap:10, marginTop:16, flexWrap:"wrap" }}>
                  <Tag color={C.amber}>{rlResult.agent} Agent</Tag>
                  <Tag color={C.text}>Baseline: {(rlResult.base_risk*100).toFixed(1)}%</Tag>
                  <Tag color={C.green}>Final: {(rlResult.final_risk*100).toFixed(1)}%</Tag>
                  <Tag color={C.green}>↓ {rlResult.total_risk_reduction_pct}% reduction</Tag>
                </div>
              )}
            </Card>

            {/* Single agent results */}
            {rlResult && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>

                {/* Risk trajectory chart */}
                <Card>
                  <SectionTitle>{rlResult.agent} Risk Trajectory</SectionTitle>
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={rlChartData}>
                      <defs>
                        <linearGradient id="rlGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={RL_AGENT_COLORS[rlAgent]} stopOpacity={0.3}/>
                          <stop offset="95%" stopColor={RL_AGENT_COLORS[rlAgent]} stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                      <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                      <YAxis tick={{ fill:C.muted, fontSize:11 }} unit="%" domain={[0, 65]}/>
                      <Tooltip
                        contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}
                        formatter={(val, name, props) => [
                          `${val}%`,
                          props.payload.intervention ? `Risk (${props.payload.intervention})` : "Risk"
                        ]}
                      />
                      <Area type="monotone" dataKey="risk" stroke={RL_AGENT_COLORS[rlAgent]}
                        fill="url(#rlGrad)" strokeWidth={2.5} dot={{ fill:RL_AGENT_COLORS[rlAgent], r:4 }}/>
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                {/* Optimal sequence table */}
                <Card>
                  <SectionTitle>Optimal Treatment Sequence</SectionTitle>
                  <div style={{ overflowX:"auto" }}>
                    <table style={{ width:"100%", borderCollapse:"collapse", fontFamily:"'Courier New',monospace", fontSize:12 }}>
                      <thead>
                        <tr>{["Step","Month","Intervention","Risk","Δ Risk"].map(h=>(
                          <th key={h} style={{ textAlign:"left", padding:"7px 12px", color:C.muted, borderBottom:`1px solid ${C.border}`, fontWeight:400, letterSpacing:1, fontSize:10 }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        <tr style={{ borderBottom:`1px solid ${C.border}22` }}>
                          <td style={{ padding:"8px 12px", color:C.muted }}>—</td>
                          <td style={{ padding:"8px 12px", color:C.muted }}>M0</td>
                          <td style={{ padding:"8px 12px", color:C.muted }}>baseline</td>
                          <td style={{ padding:"8px 12px", color:riskColor(rlResult.base_risk) }}>{(rlResult.base_risk*100).toFixed(1)}%</td>
                          <td style={{ padding:"8px 12px", color:C.muted }}>—</td>
                        </tr>
                        {rlResult.optimal_sequence?.map((s, i) => (
                          <tr key={i} style={{ borderBottom:`1px solid ${C.border}22` }}>
                            <td style={{ padding:"8px 12px", color:C.muted }}>#{s.step}</td>
                            <td style={{ padding:"8px 12px", color:C.text }}>M{s.month}</td>
                            <td style={{ padding:"8px 12px" }}>
                              <span style={{
                                color: SCENARIO_COLORS[s.intervention] || C.accent,
                                background: (SCENARIO_COLORS[s.intervention] || C.accent)+"15",
                                padding:"2px 7px", borderRadius:4,
                              }}>{s.intervention}</span>
                            </td>
                            <td style={{ padding:"8px 12px", color:riskColor(s.risk) }}>{(s.risk*100).toFixed(1)}%</td>
                            <td style={{ padding:"8px 12px", color:s.delta<=0?C.green:C.red, fontWeight:700 }}>
                              {s.delta<=0?"↓":"↑"}{Math.abs(s.delta*100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ marginTop:14, padding:"10px 14px", background:C.surface, borderRadius:8, fontSize:12, color:C.text, lineHeight:1.7 }}>
                    {rlResult.summary}
                  </div>
                </Card>
              </div>
            )}

            {/* DQN vs PPO comparison */}
            {rlCompare && (
              <>
                <Card>
                  <SectionTitle>DQN vs PPO — Risk Trajectory Comparison</SectionTitle>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={rlCompareChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                      <XAxis dataKey="month" tick={{ fill:C.muted, fontSize:11 }}/>
                      <YAxis tick={{ fill:C.muted, fontSize:11 }} unit="%" domain={[0, 65]}/>
                      <Tooltip contentStyle={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }}/>
                      <Legend wrapperStyle={{ fontSize:11 }}/>
                      <Line type="monotone" dataKey="DQN" stroke={C.accent} strokeWidth={2.5} dot={{ r:4 }} name="DQN Agent"/>
                      <Line type="monotone" dataKey="PPO" stroke={C.purple} strokeWidth={2.5} dot={{ r:4 }} strokeDasharray="6 3" name="PPO Agent"/>
                    </LineChart>
                  </ResponsiveContainer>
                </Card>

                {/* Head-to-head stat cards */}
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
                  {["dqn","ppo"].map(key => {
                    const r = rlCompare[key];
                    const info = RL_ALGO_INFO[key];
                    return (
                      <Card key={key} style={{ border:`1px solid ${info.color}33` }}>
                        <SectionTitle>{info.icon} {key.toUpperCase()} Summary</SectionTitle>
                        <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                          {[
                            ["Baseline Risk",  `${(r.base_risk*100).toFixed(1)}%`,  C.text],
                            ["Final Risk",     `${(r.final_risk*100).toFixed(1)}%`, riskColor(r.final_risk)],
                            ["Risk Reduction", `−${r.total_risk_reduction_pct}%`,   C.green],
                            ["Best Step",      r.optimal_sequence?.reduce((best,s) => s.delta < best.delta ? s : best, r.optimal_sequence[0])?.intervention || "—", info.color],
                          ].map(([label, val, color]) => (
                            <div key={label} style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                              <span style={{ fontSize:12, color:C.muted }}>{label}</span>
                              <span style={{ fontSize:13, color, fontFamily:"'Courier New',monospace", fontWeight:700 }}>{val}</span>
                            </div>
                          ))}
                        </div>
                        {/* Per-step intervention breakdown */}
                        <div style={{ marginTop:16, paddingTop:14, borderTop:`1px solid ${C.border}` }}>
                          <div style={{ fontSize:10, color:C.muted, letterSpacing:1, marginBottom:10, textTransform:"uppercase" }}>Intervention Sequence</div>
                          <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                            {r.optimal_sequence?.map((s, i) => (
                              <span key={i} style={{
                                fontSize:10, padding:"3px 8px", borderRadius:4,
                                background:(SCENARIO_COLORS[s.intervention]||info.color)+"18",
                                color: SCENARIO_COLORS[s.intervention]||info.color,
                                border:`1px solid ${(SCENARIO_COLORS[s.intervention]||info.color)}33`,
                                fontFamily:"'Courier New',monospace",
                              }}>M{s.month}:{s.intervention.replace("_","\u00A0")}</span>
                            ))}
                          </div>
                        </div>
                      </Card>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── TAB 5: Chat ── */}
        {tab==="chat" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 280px", gap:24, alignItems:"start" }}>
            <Card style={{ padding:0, display:"flex", flexDirection:"column", height:620 }}>
              <div style={{ padding:"16px 24px", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                  <div style={{ width:32, height:32, borderRadius:"50%", background:C.purple+"22", border:`1px solid ${C.purple}44`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:14 }}>🧬</div>
                  <div>
                    <div style={{ fontSize:12, color:C.text, letterSpacing:2, fontFamily:"'Courier New',monospace" }}>MEDTWIN LLM</div>
                    <div style={{ fontSize:10, color:health?.groq_enabled?C.green:C.red, letterSpacing:1 }}>
                      {health?.groq_enabled?"● Groq · llama-3.3-70b-versatile · Active":"● Groq offline — check GROQ_API_KEY"}
                    </div>
                  </div>
                </div>
                {patientId ? <Tag color={C.green}>Patient · {patientId}</Tag> : <Tag color={C.amber}>No patient stored — using form data</Tag>}
              </div>
              <div style={{ flex:1, overflowY:"auto", padding:"20px 24px", display:"flex", flexDirection:"column", gap:16 }}>
                {chatMessages.map((msg, i) => (
                  <div key={i} style={{ display:"flex", flexDirection:msg.role==="user"?"row-reverse":"row", gap:10, alignItems:"flex-start" }}>
                    <div style={{ width:28, height:28, borderRadius:"50%", flexShrink:0, background:msg.role==="user"?C.accent+"22":C.purple+"22", border:`1px solid ${msg.role==="user"?C.accent+"44":C.purple+"44"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12 }}>
                      {msg.role==="user"?"👤":"🧬"}
                    </div>
                    <div style={{ maxWidth:"75%" }}>
                      <div style={{ background:msg.role==="user"?C.accent+"11":msg.isError?C.red+"11":C.surface, border:`1px solid ${msg.role==="user"?C.accent+"33":msg.isError?C.red+"33":C.border}`, borderRadius:msg.role==="user"?"12px 4px 12px 12px":"4px 12px 12px 12px", padding:"12px 16px", fontSize:13, lineHeight:1.7, color:msg.isError?C.red:C.text, fontFamily:msg.role==="assistant"?"Georgia,serif":"'Courier New',monospace", whiteSpace:"pre-wrap" }}>
                        {msg.content}
                      </div>
                      <div style={{ fontSize:10, color:C.border, marginTop:4, textAlign:msg.role==="user"?"right":"left", fontFamily:"'Courier New',monospace" }}>
                        {msg.ts}{msg.tokens?` · ${msg.tokens} tokens`:""}
                      </div>
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div style={{ display:"flex", gap:10, alignItems:"flex-start" }}>
                    <div style={{ width:28, height:28, borderRadius:"50%", background:C.purple+"22", border:`1px solid ${C.purple}44`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12 }}>🧬</div>
                    <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:"4px 12px 12px 12px", padding:"12px 16px", display:"flex", gap:6, alignItems:"center" }}>
                      {[0,1,2].map(i=><div key={i} style={{ width:6, height:6, borderRadius:"50%", background:C.purple, animation:`pulse 1.2s ease-in-out ${i*0.2}s infinite` }}/>)}
                    </div>
                  </div>
                )}
                <div ref={chatEndRef}/>
              </div>
              <div style={{ padding:"16px 24px", borderTop:`1px solid ${C.border}`, display:"flex", gap:10 }}>
                <textarea value={chatInput} onChange={e=>setChatInput(e.target.value)} onKeyDown={handleChatKey}
                  placeholder="Ask about the patient's risks, biomarkers, or interventions… (Enter to send)" rows={2}
                  style={{ ...inputStyle, resize:"none", lineHeight:1.5, fontFamily:"'Courier New',monospace", fontSize:12 }}/>
                <button onClick={()=>sendChat()} disabled={chatLoading||!chatInput.trim()}
                  style={{ ...btn(C.purple,chatLoading||!chatInput.trim()), padding:"0 20px", whiteSpace:"nowrap", alignSelf:"stretch" }}>
                  {chatLoading?"…":"Send →"}
                </button>
              </div>
            </Card>
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
              <Card>
                <SectionTitle>Quick Questions</SectionTitle>
                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  {SUGGESTED_QUESTIONS.map((q,i)=>(
                    <button key={i} onClick={()=>sendChat(q)} disabled={chatLoading}
                      style={{ background:C.surface, border:`1px solid ${C.border}`, color:C.muted, borderRadius:6, padding:"8px 12px", fontSize:11, textAlign:"left", cursor:chatLoading?"not-allowed":"pointer", fontFamily:"Georgia,serif", lineHeight:1.5, transition:"all 0.2s" }}
                      onMouseEnter={e=>{e.target.style.borderColor=C.purple;e.target.style.color=C.text;}}
                      onMouseLeave={e=>{e.target.style.borderColor=C.border;e.target.style.color=C.muted;}}>
                      {q}
                    </button>
                  ))}
                </div>
              </Card>
              <Card>
                <SectionTitle>Patient Context</SectionTitle>
                <div style={{ fontFamily:"'Courier New',monospace", fontSize:11, display:"flex", flexDirection:"column", gap:6 }}>
                  {[["Glucose",`${p.glucose} mg/dL`],["BP",`${p.bp_systolic} mmHg`],["BMI",p.bmi],["HbA1c",`${p.hba1c}%`],["Sleep",`${p.sleep_hours} hrs`],["Stress",`${p.stress_level}/100`],["Activity",`${p.activity_score}/100`]].map(([k,v])=>(
                    <div key={k} style={{ display:"flex", justifyContent:"space-between" }}>
                      <span style={{ color:C.muted }}>{k}</span><span style={{ color:C.accent }}>{v}</span>
                    </div>
                  ))}
                  {risk && (
                    <div style={{ marginTop:8, paddingTop:8, borderTop:`1px solid ${C.border}` }}>
                      <div style={{ display:"flex", justifyContent:"space-between" }}><span style={{ color:C.muted }}>Diabetes Risk</span><span style={{ color:riskColor(risk.diabetes.risk_score) }}>{Math.round(risk.diabetes.risk_score*100)}%</span></div>
                      <div style={{ display:"flex", justifyContent:"space-between" }}><span style={{ color:C.muted }}>CV Risk</span><span style={{ color:riskColor(risk.cardiovascular.risk_score) }}>{Math.round(risk.cardiovascular.risk_score*100)}%</span></div>
                    </div>
                  )}
                </div>
              </Card>
              <Card>
                <SectionTitle>Tips</SectionTitle>
                <div style={{ fontSize:11, color:C.muted, lineHeight:1.8 }}>
                  • Run <span style={{ color:C.accent }}>Predict Risk</span> first so scores are included in the AI context.<br/>
                  • Press <span style={{ color:C.accent }}>Enter</span> to send, <span style={{ color:C.accent }}>Shift+Enter</span> for new line.<br/>
                  • The AI remembers the full conversation history.
                </div>
              </Card>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 60%, 100% { opacity: 0.2; transform: scale(0.8); }
          30% { opacity: 1; transform: scale(1); }
        }
      `}</style>

      <div style={{ borderTop:`1px solid ${C.border}`, padding:"14px 32px", textAlign:"center" }}>
        <span style={{ fontSize:10, color:C.border, letterSpacing:2, fontFamily:"'Courier New',monospace" }}>
          MEDTWIN AI · DIGITAL TWIN SIMULATOR · NOT FOR CLINICAL USE
        </span>
      </div>
    </div>
  );
}