import { useState, useEffect, useRef } from "react";

const API = "http://127.0.0.1:8000";

const VALID_USERS = [
  { username: "demo",  password: "medtwin2026", name: "Dr. Demo",    role: "Physician"      },
  { username: "admin", password: "admin123",    name: "Admin User",  role: "Administrator"  },
];

/* ─── tiny design tokens ──────────────────────────────── */
const T = {
  bg:     "#04070d",
  bg1:    "#080d15",
  bg2:    "#0c1420",
  bdr:    "#142030",
  bdr2:   "#1d3045",
  c:      "#00e5ff",
  cg:     "#00ffa3",
  cr:     "#ff3860",
  ca:     "#ffcc00",
  cp:     "#c77dff",
  cw:     "#e8f4ff",
  cm:     "#5a7a99",
};

/* ─── Canvas background ──────────────────────────────── */
function BioCanvas() {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx    = canvas.getContext("2d");
    let W = canvas.width  = window.innerWidth;
    let H = canvas.height = window.innerHeight;
    let raf;

    /* particles */
    const pts = Array.from({ length: 70 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35,
      r: Math.random() * 1.8 + .4,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, W, H);

      /* subtle hex grid */
      ctx.strokeStyle = "#142030";
      ctx.lineWidth   = .5;
      const size = 60, h3 = size * Math.sqrt(3);
      for (let row = -1; row < H / h3 + 2; row++) {
        for (let col = -1; col < W / (size * 1.5) + 2; col++) {
          const cx = col * size * 1.5;
          const cy = row * h3 + (col % 2 === 0 ? 0 : h3 / 2);
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const a = (Math.PI / 180) * (60 * i - 30);
            i === 0
              ? ctx.moveTo(cx + size * Math.cos(a), cy + size * Math.sin(a))
              : ctx.lineTo(cx + size * Math.cos(a), cy + size * Math.sin(a));
          }
          ctx.closePath();
          ctx.stroke();
        }
      }

      /* connections */
      pts.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      });
      pts.forEach((a, i) => {
        pts.slice(i + 1).forEach(b => {
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 130) {
            ctx.strokeStyle = `rgba(0,229,255,${.10 * (1 - d / 130)})`;
            ctx.lineWidth   = .6;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        });
        ctx.fillStyle = `rgba(0,229,255,${.35 + p.r * .1})`;
        ctx.beginPath(); ctx.arc(a.x, a.y, p.r, 0, Math.PI * 2); ctx.fill();
      });

      /* radial vignette */
      const grad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * .7);
      grad.addColorStop(0, "rgba(4,7,13,0)");
      grad.addColorStop(1, "rgba(4,7,13,.85)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);

      raf = requestAnimationFrame(draw);
    };
    draw();
    const onResize = () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, []);
  return <canvas ref={ref} style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none" }} />;
}

/* ─── Input component ──────────────────────────────── */
function Field({ label, icon, type="text", value, onChange, placeholder, autoComplete }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ marginBottom: 20 }}>
      <label style={{
        display:"block", fontSize:10, letterSpacing:2.5,
        textTransform:"uppercase", color: focused ? T.c : T.cm,
        marginBottom:8, fontFamily:"'JetBrains Mono',monospace",
        transition:"color .2s",
      }}>{label}</label>
      <div style={{ position:"relative" }}>
        <span style={{
          position:"absolute", left:14, top:"50%", transform:"translateY(-50%)",
          color: focused ? T.c : T.cm, transition:"color .2s", pointerEvents:"none",
          display:"flex", alignItems:"center",
        }}>{icon}</span>
        <input
          value={value}
          onChange={e => onChange(e.target.value)}
          type={type}
          placeholder={placeholder}
          autoComplete={autoComplete}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={{
            width:"100%",
            background: focused ? "#0c1420" : "#080d15",
            border: `1px solid ${focused ? T.c + "77" : T.bdr2}`,
            borderRadius:10,
            color: T.cw,
            padding:"13px 14px 13px 42px",
            fontSize:14,
            fontFamily:"'JetBrains Mono',monospace",
            outline:"none",
            transition:"background .2s, border-color .2s",
            boxShadow: focused ? `0 0 0 4px ${T.c}11, inset 0 0 12px ${T.c}05` : "none",
          }}
        />
      </div>
    </div>
  );
}

/* ─── Main LoginPage ──────────────────────────────── */
export default function LoginPage({ onLogin }) {
  const [username, setUsername]     = useState("");
  const [password, setPassword]     = useState("");
  const [showPass,  setShowPass]    = useState(false);
  const [error,    setError]        = useState("");
  const [loading,  setLoading]      = useState(false);
  const [success,  setSuccess]      = useState(false);
  const [apiOk,    setApiOk]        = useState(null);
  const [mounted,  setMounted]      = useState(false);

  useEffect(() => {
    setTimeout(() => setMounted(true), 80);
    fetch(`${API}/health`).then(r => r.json()).then(() => setApiOk(true)).catch(() => setApiOk(false));
  }, []);

  const handleLogin = async e => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) { setError("Credentials required."); return; }
    setLoading(true); setError("");
    await new Promise(r => setTimeout(r, 900));
    const user = VALID_USERS.find(u => u.username === username && u.password === password);
    if (user) { setSuccess(true); setTimeout(() => onLogin(user), 700); }
    else { setError("Invalid credentials — try demo / medtwin2026"); setLoading(false); }
  };

  return (
    <div style={{ position:"fixed", inset:0, overflow:"hidden", fontFamily:"'Syne',sans-serif", background:T.bg }}>
      <BioCanvas />

      {/* scanline effect */}
      <div style={{
        position:"fixed", inset:0, zIndex:1, pointerEvents:"none",
        backgroundImage:"repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,.03) 3px, rgba(0,0,0,.03) 4px)",
      }} />

      {/* glow blob */}
      <div style={{
        position:"fixed", top:"20%", left:"50%", transform:"translateX(-50%)",
        width:600, height:600, borderRadius:"50%", zIndex:1, pointerEvents:"none",
        background:`radial-gradient(circle, ${T.c}08 0%, transparent 70%)`,
      }} />

      {/* center container */}
      <div style={{
        position:"relative", zIndex:2,
        minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center",
        padding:24,
      }}>
        <div style={{
          width:"100%", maxWidth:460,
          opacity: mounted ? 1 : 0,
          transform: mounted ? "translateY(0)" : "translateY(24px)",
          transition: "opacity .7s ease, transform .7s ease",
        }}>

          {/* Card */}
          <div style={{
            background: success
              ? `linear-gradient(145deg, ${T.cg}08, #080d15 60%)`
              : "linear-gradient(145deg, #0c142088, #080d15ee)",
            backdropFilter: "blur(32px)",
            border: `1px solid ${success ? T.cg + "55" : T.bdr2}`,
            borderRadius: 20,
            overflow:"hidden",
            boxShadow: success
              ? `0 0 80px ${T.cg}22, 0 40px 80px rgba(0,0,0,.6)`
              : `0 0 60px ${T.c}0a, 0 40px 80px rgba(0,0,0,.5)`,
            transition:"all .6s ease",
          }}>

            {/* top gradient bar */}
            <div style={{
              height:2,
              background: success
                ? `linear-gradient(90deg, transparent, ${T.cg}, ${T.c}, ${T.cg}, transparent)`
                : `linear-gradient(90deg, transparent, ${T.c}99, ${T.cp}88, ${T.c}99, transparent)`,
              transition:"all .6s ease",
            }} />

            <div style={{ padding:"44px 44px 36px" }}>

              {/* ── Logo ── */}
              <div style={{ display:"flex", alignItems:"center", gap:18, marginBottom:44 }}>
                <div style={{ position:"relative", flexShrink:0 }}>
                  <svg width={46} height={46} viewBox="0 0 46 46">
                    <circle cx={23} cy={23} r={21} fill="none" stroke={T.c} strokeWidth={1.2} />
                    <circle cx={23} cy={23} r={14} fill="none" stroke={T.c} strokeWidth={.5} strokeDasharray="4 3" style={{ animation:"spin-slow 24s linear infinite", transformOrigin:"23px 23px" }} />
                    <circle cx={23} cy={23} r={5.5} fill={T.c} opacity={.9} />
                    <circle cx={23} cy={2}  r={2.2} fill={T.c} />
                    <circle cx={23} cy={44} r={2.2} fill={T.c} />
                    <circle cx={2}  cy={23} r={2.2} fill={T.c} />
                    <circle cx={44} cy={23} r={2.2} fill={T.c} />
                    <line x1={23} y1={4}  x2={23} y2={9}  stroke={T.c} strokeWidth={1.4} />
                    <line x1={23} y1={37} x2={23} y2={42} stroke={T.c} strokeWidth={1.4} />
                    <line x1={4}  y1={23} x2={9}  y2={23} stroke={T.c} strokeWidth={1.4} />
                    <line x1={37} y1={23} x2={42} y2={23} stroke={T.c} strokeWidth={1.4} />
                  </svg>
                  <div style={{
                    position:"absolute", inset:-4, borderRadius:"50%",
                    boxShadow:`0 0 24px ${T.c}44, 0 0 48px ${T.c}22`,
                    pointerEvents:"none",
                  }} />
                </div>
                <div>
                  <div style={{
                    fontSize:22, fontWeight:800, letterSpacing:4,
                    color:T.c, fontFamily:"'Syne',sans-serif",
                    textShadow:`0 0 32px ${T.c}66`,
                    lineHeight:1,
                  }}>MEDTWIN AI</div>
                  <div style={{
                    fontSize:9.5, color:T.cm, letterSpacing:3.5,
                    marginTop:5, textTransform:"uppercase",
                    fontFamily:"'JetBrains Mono',monospace",
                  }}>Digital Twin Patient Simulator</div>
                </div>
              </div>

              {/* ── Sign In heading ── */}
              <div style={{ marginBottom:32 }}>
                <h1 style={{
                  fontFamily:"'Instrument Serif',serif", fontStyle:"italic",
                  fontSize:28, fontWeight:400, color:T.cw, lineHeight:1,
                  marginBottom:6,
                }}>Welcome back</h1>
                <p style={{ fontSize:13, color:T.cm, fontFamily:"'JetBrains Mono',monospace" }}>
                  Authenticate to access the simulation platform
                </p>
              </div>

              <form onSubmit={handleLogin}>
                <Field
                  label="Username"
                  value={username}
                  onChange={setUsername}
                  placeholder="e.g. demo"
                  autoComplete="username"
                  icon={
                    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                      <circle cx={12} cy={8} r={4}/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                    </svg>
                  }
                />
                <div style={{ position:"relative" }}>
                  <Field
                    label="Password"
                    value={password}
                    onChange={setPassword}
                    type={showPass ? "text" : "password"}
                    placeholder="••••••••••••"
                    autoComplete="current-password"
                    icon={
                      <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <rect x={3} y={11} width={18} height={11} rx={2}/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                      </svg>
                    }
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(s => !s)}
                    style={{
                      position:"absolute", right:14, top:38,
                      background:"none", border:"none", color:T.cm, cursor:"pointer",
                      padding:0, display:"flex", alignItems:"center",
                    }}
                  >
                    {showPass
                      ? <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1={1} y1={1} x2={23} y2={23}/></svg>
                      : <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx={12} cy={12} r={3}/></svg>
                    }
                  </button>
                </div>

                {error && (
                  <div style={{
                    background:`${T.cr}10`, border:`1px solid ${T.cr}44`,
                    borderRadius:10, padding:"11px 14px", marginBottom:18,
                    fontSize:12, color:T.cr, display:"flex", alignItems:"center", gap:8,
                    fontFamily:"'JetBrains Mono',monospace",
                    animation:"fadeUp .3s ease",
                  }}>
                    <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><circle cx={12} cy={12} r={10}/><line x1={12} y1={8} x2={12} y2={12}/><line x1={12} y1={16} x2={12.01} y2={16}/></svg>
                    {error}
                  </div>
                )}

                {/* Submit */}
                <button
                  type="submit"
                  disabled={loading || success}
                  style={{
                    width:"100%", padding:"14px 0", borderRadius:10,
                    background: success
                      ? `linear-gradient(135deg, ${T.cg}22, ${T.cg}10)`
                      : loading
                        ? `${T.c}0d`
                        : `linear-gradient(135deg, ${T.c}22, ${T.c}0d)`,
                    border:`1px solid ${success ? T.cg + "66" : T.c + "55"}`,
                    color: success ? T.cg : T.c,
                    fontSize:13, fontWeight:600, letterSpacing:3,
                    textTransform:"uppercase", cursor: loading||success ? "not-allowed" : "pointer",
                    fontFamily:"'Syne',sans-serif",
                    boxShadow: (!loading && !success) ? `0 0 24px ${T.c}18` : "none",
                    transition:"all .3s",
                  }}
                  onMouseEnter={e => { if(!loading && !success) e.currentTarget.style.boxShadow=`0 0 32px ${T.c}33, 0 8px 32px ${T.c}22`; }}
                  onMouseLeave={e => { e.currentTarget.style.boxShadow=(!loading && !success)?`0 0 24px ${T.c}18`:"none"; }}
                >
                  {success ? "✓ Access Granted" : loading ? "Authenticating…" : "→ Enter System"}
                </button>

                {/* Demo fill */}
                <button
                  type="button"
                  onClick={() => { setUsername("demo"); setPassword("medtwin2026"); setError(""); }}
                  style={{
                    width:"100%", marginTop:10, padding:"11px 0", borderRadius:10,
                    background:"transparent", border:`1px solid ${T.bdr2}`,
                    color:T.cm, cursor:"pointer", fontSize:11, letterSpacing:2,
                    textTransform:"uppercase", fontFamily:"'JetBrains Mono',monospace",
                    transition:"all .2s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor=T.ca+"77"; e.currentTarget.style.color=T.ca; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor=T.bdr2; e.currentTarget.style.color=T.cm; }}
                >
                  Use Demo Credentials
                </button>
              </form>
            </div>

            {/* Footer */}
            <div style={{
              borderTop:`1px solid ${T.bdr}`,
              padding:"14px 44px",
              display:"flex", alignItems:"center", justifyContent:"space-between",
            }}>
              <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                <div style={{
                  width:7, height:7, borderRadius:"50%",
                  background: apiOk===null ? T.ca : apiOk ? T.cg : T.cr,
                  boxShadow:`0 0 8px ${apiOk===null ? T.ca : apiOk ? T.cg : T.cr}`,
                  animation: "pulse-dot 2s ease-in-out infinite",
                }} />
                <span style={{ fontSize:9.5, color:T.cm, letterSpacing:2, fontFamily:"'JetBrains Mono',monospace" }}>
                  {apiOk===null ? "CHECKING API" : apiOk ? "API LIVE" : "API OFFLINE"}
                </span>
              </div>
              <span style={{ fontSize:9, color:T.bdr2, letterSpacing:2, fontFamily:"'JetBrains Mono',monospace" }}>
                NOT FOR CLINICAL USE
              </span>
            </div>
          </div>

          {/* Below-card hint */}
          <p style={{
            textAlign:"center", marginTop:18,
            fontSize:11, color:T.cm, fontFamily:"'JetBrains Mono',monospace",
            letterSpacing:1, lineHeight:1.8,
            opacity: mounted ? .7 : 0, transition:"opacity 1s ease .4s",
          }}>
            Authorized personnel only · <span style={{ color:T.c }}>demo</span> / <span style={{ color:T.c }}>medtwin2026</span>
          </p>
        </div>
      </div>

      <style>{`
        @keyframes spin-slow { to { transform: rotate(360deg); } }
        @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        input::placeholder { color: #5a7a9944; }
      `}</style>
    </div>
  );
}