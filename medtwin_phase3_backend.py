"""
MedTwin AI — Unified Backend (Phase 3 + Phase 5 + Auth)
========================================================
All endpoints in one file:

  POST /auth/register             — create user account
  POST /auth/login                — returns JWT access token
  GET  /auth/me                   — get current logged-in user info

  GET  /health                    — server / model status
  POST /patient/create            — validate & store patient profile (auth required)
  GET  /patient/{patient_id}      — retrieve stored patient (auth required)
  GET  /patients                  — list all your patients (auth required)

  POST /predict/risk              — diabetes + cardiovascular risk (SHAP)
  POST /simulate/trajectory       — multi-month forecast under one intervention
  POST /simulate/trajectory_ci    — same but with Monte Carlo confidence bands
  POST /simulate/scenarios        — compare all 6 interventions side-by-side
  POST /explain                   — SHAP top factors + NL explanation
  POST /explain/counterfactual    — "If glucose → 120, risk drops 66% → 41%"
  POST /optimize/treatment        — RL Q-learning optimal treatment sequence
  POST /optimize/rl               — Trained DQN or PPO agent inference
  POST /chat                      — Groq LLM medical assistant

Prerequisites:
  pip install fastapi uvicorn pydantic joblib numpy pandas groq python-dotenv torch pymongo python-jose[cryptography] passlib[bcrypt]

.env keys needed:
  GROQ_API_KEY=...
  MONGODB_URI=mongodb+srv://...
  SECRET_KEY=some-long-random-string   (optional — has a default for dev)

Run:
  python medtwin_phase3_backend.py

Auto-docs:
  http://localhost:8000/docs
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from collections import defaultdict
from datetime import datetime, timedelta
import os
import random
import uuid

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dotenv import load_dotenv
load_dotenv()

# Auth
from jose import JWTError, jwt
from passlib.context import CryptContext

# -----------------------------------------------------------------------------
# AUTH CONFIG
# -----------------------------------------------------------------------------

SECRET_KEY                  = os.environ.get("SECRET_KEY", "medtwin-dev-secret-change-in-production")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24      # 24 hours

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -----------------------------------------------------------------------------
# MONGODB
# -----------------------------------------------------------------------------

from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGODB_URI", "")
if not MONGO_URI:
    print("WARNING: MONGODB_URI not set in .env — data will not persist.")
    patients_col = None
    users_col    = None
else:
    _mongo_client = MongoClient(MONGO_URI)
    _mongo_db     = _mongo_client["medtwin"]
    patients_col  = _mongo_db["patients"]
    users_col     = _mongo_db["users"]
    patients_col.create_index("patient_id", unique=True)
    users_col.create_index("email",         unique=True)
    users_col.create_index("username",      unique=True)
    print("MongoDB connected.")

# -----------------------------------------------------------------------------
# GROQ CLIENT
# -----------------------------------------------------------------------------

try:
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None
    print("groq not installed — /chat endpoint disabled.")

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="MedTwin AI API",
    description=(
        "Digital Twin Patient Simulation — "
        "Risk Scoring, Trajectory Forecasting, Counterfactual Explanations & RL Optimisation"
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------

MODELS: Dict[str, object] = {}

def load_models() -> bool:
    base     = "models"
    required = [
        "diabetes_risk.pkl", "diabetes_risk_shap.pkl",
        "heart_risk.pkl",    "heart_risk_shap.pkl",
        "progression_model.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(base, f))]
    if missing:
        print(f"Missing model files: {missing}")
        print("   Run medtwin_phase1.py and medtwin_phase2.py first.")
        return False
    MODELS["diabetes"]      = joblib.load(f"{base}/diabetes_risk.pkl")
    MODELS["diabetes_shap"] = joblib.load(f"{base}/diabetes_risk_shap.pkl")
    MODELS["heart"]         = joblib.load(f"{base}/heart_risk.pkl")
    MODELS["heart_shap"]    = joblib.load(f"{base}/heart_risk_shap.pkl")
    MODELS["progression"]   = joblib.load(f"{base}/progression_model.pkl")
    print("All models loaded.")
    return True

load_models()

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

INTERVENTION_CODES = {
    "none": 0, "metformin": 1, "lifestyle": 2,
    "combined": 3, "statin": 4, "sleep_therapy": 5,
}

HEALTH_FEATURES = [
    "glucose", "bp_systolic", "bmi", "cholesterol",
    "sleep_hours", "stress_level", "activity_score", "hba1c",
]

CF_FEATURES = {
    "glucose":        {"min": 70,  "max": 250, "step": 10,  "label": "Glucose",        "unit": "mg/dL"},
    "bp_systolic":    {"min": 90,  "max": 180, "step": 5,   "label": "Blood Pressure", "unit": "mmHg"},
    "bmi":            {"min": 18,  "max": 45,  "step": 1,   "label": "BMI",            "unit": ""},
    "cholesterol":    {"min": 130, "max": 320, "step": 10,  "label": "Cholesterol",    "unit": "mg/dL"},
    "sleep_hours":    {"min": 3,   "max": 10,  "step": 0.5, "label": "Sleep Hours",    "unit": "hrs"},
    "stress_level":   {"min": 0,   "max": 100, "step": 10,  "label": "Stress Level",   "unit": ""},
    "activity_score": {"min": 0,   "max": 100, "step": 10,  "label": "Activity Score", "unit": ""},
    "hba1c":          {"min": 4.5, "max": 12,  "step": 0.5, "label": "HbA1c",          "unit": "%"},
}

GROQ_SYSTEM_PROMPT = """You are MedTwin AI, an intelligent medical simulation assistant.
You help doctors, students, and researchers understand patient health trajectories.
You have access to the patient's digital twin including risk scores, SHAP factors, and simulation results.

Rules:
- Be precise and data-driven. Always reference the patient's actual numbers.
- Explain in clear, professional but accessible language.
- When discussing risk, always give the percentage and what it means clinically.
- Suggest actionable interventions when relevant.
- Never make definitive clinical diagnoses. Say "simulation suggests" not "you have".
- Keep responses concise (3-5 sentences) unless asked to elaborate.
- If asked about something outside the patient data, say so clearly.
"""

# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS — Auth
# -----------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username:  str            = Field(..., min_length=3, max_length=30)
    email:     str            = Field(..., description="Valid email address")
    password:  str            = Field(..., min_length=6)
    full_name: Optional[str]  = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    username:     str
    full_name:    str


class UserInfo(BaseModel):
    user_id:   str
    username:  str
    email:     str
    full_name: str

# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS — Patient & endpoints
# -----------------------------------------------------------------------------

class PatientProfile(BaseModel):
    name:              Optional[str]       = "Anonymous"
    age:               int                 = Field(..., ge=18,  le=100)
    gender:            str                 = Field(..., pattern="^(male|female|other)$")
    glucose:           float               = Field(..., ge=50,   le=400)
    bp_systolic:       float               = Field(..., ge=80,   le=220)
    bmi:               float               = Field(..., ge=10,   le=60)
    cholesterol:       float               = Field(..., ge=100,  le=400)
    hba1c:             float               = Field(..., ge=4.0,  le=15.0)
    sleep_hours:       float               = Field(..., ge=2,    le=12)
    stress_level:      float               = Field(..., ge=0,    le=100)
    activity_score:    float               = Field(..., ge=0,    le=100)
    pregnancies:       Optional[int]       = 0
    skin_thickness:    Optional[float]     = 20.0
    insulin:           Optional[float]     = 80.0
    diabetes_pedigree: Optional[float]     = 0.3
    diseases:          Optional[List[str]] = []
    medications:       Optional[List[str]] = []


class RiskRequest(BaseModel):
    patient_id: Optional[str]            = None
    patient:    Optional[PatientProfile] = None


class SimulationRequest(BaseModel):
    patient_id:   Optional[str]            = None
    patient:      Optional[PatientProfile] = None
    intervention: str                      = Field("none")
    checkpoints:  List[int]                = Field([0, 3, 6, 12, 18, 24])


class ScenarioRequest(BaseModel):
    patient_id:  Optional[str]            = None
    patient:     Optional[PatientProfile] = None
    checkpoints: List[int]                = Field([0, 3, 6, 12, 18, 24])


class ExplainRequest(BaseModel):
    patient_id: Optional[str]            = None
    patient:    Optional[PatientProfile] = None
    model_type: str                      = Field("diabetes")


class CounterfactualRequest(BaseModel):
    patient_id: Optional[str]  = None
    patient:    Optional[dict] = None


class TrajectoryWithCIRequest(BaseModel):
    patient_id:   Optional[str]  = None
    patient:      Optional[dict] = None
    intervention: str            = Field("none")
    checkpoints:  List[int]      = Field([0, 3, 6, 9, 12, 18, 24])
    n_samples:    int            = Field(50)


class OptimizeRequest(BaseModel):
    patient_id: Optional[str]  = None
    patient:    Optional[dict] = None
    horizon:    int            = Field(12)
    episodes:   int            = Field(200)


class RLOptimizeRequest(BaseModel):
    patient_id:  Optional[str]  = None
    patient:     Optional[dict] = None
    agent_type:  str            = Field("dqn")


class ChatRequest(BaseModel):
    patient_id:   Optional[str]        = None
    patient:      Optional[dict]       = None
    question:     str
    risk_context: Optional[dict]       = None
    history:      Optional[List[dict]] = []

# -----------------------------------------------------------------------------
# AUTH HELPERS
# -----------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency — decodes JWT and returns the user document."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if users_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")

    user = users_col.find_one({"user_id": user_id}, {"_id": 0, "hashed_password": 0})
    if user is None:
        raise credentials_exception
    return user

# -----------------------------------------------------------------------------
# PATIENT HELPERS (MongoDB, scoped to user_id)
# -----------------------------------------------------------------------------

def _fetch_patient_doc(patient_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    if patients_col is None or not patient_id:
        return None
    query = {"patient_id": patient_id}
    if user_id:
        query["user_id"] = user_id
    return patients_col.find_one(query, {"_id": 0})


def resolve_patient(
    patient_id: Optional[str],
    patient_obj: Optional[PatientProfile],
    user_id: Optional[str] = None,
) -> PatientProfile:
    if patient_id:
        doc = _fetch_patient_doc(patient_id, user_id)
        if doc:
            return PatientProfile(**doc["profile"])
    if patient_obj:
        return patient_obj
    raise HTTPException(status_code=404, detail="Patient not found.")


def resolve_patient_dict(
    patient_id: Optional[str],
    patient_dict: Optional[dict],
    user_id: Optional[str] = None,
) -> dict:
    if patient_id:
        doc = _fetch_patient_doc(patient_id, user_id)
        if doc:
            return doc["profile"]
    if patient_dict:
        return patient_dict
    raise HTTPException(status_code=404, detail="Patient not found.")

# -----------------------------------------------------------------------------
# RISK / SIMULATION HELPERS
# -----------------------------------------------------------------------------

def compute_composite_risk(state: dict) -> float:
    glucose_risk   = np.clip((state["glucose"]        - 70)  / 180, 0, 1)
    bp_risk        = np.clip((state["bp_systolic"]    - 90)  / 90,  0, 1)
    bmi_risk       = np.clip((state["bmi"]            - 18)  / 27,  0, 1)
    chol_risk      = np.clip((state["cholesterol"]    - 130) / 190, 0, 1)
    hba1c_risk     = np.clip((state["hba1c"]          - 4.5) / 7.5, 0, 1)
    lifestyle_risk = np.clip(1 - state["activity_score"] / 100,     0, 1)
    return round(float(
        0.30 * glucose_risk  +
        0.20 * bp_risk       +
        0.15 * bmi_risk      +
        0.15 * chol_risk     +
        0.12 * hba1c_risk    +
        0.08 * lifestyle_risk
    ), 4)


def risk_label(score: float) -> str:
    if score > 0.75: return "Critical"
    if score > 0.55: return "High"
    if score > 0.35: return "Moderate"
    return "Low"


def simulate_trajectory(patient: PatientProfile, intervention: str, checkpoints: list) -> dict:
    artifact          = MODELS["progression"]
    model, scaler     = artifact["model"], artifact["scaler"]
    features, targets = artifact["features"], artifact["targets"]
    results = {}
    for month in checkpoints:
        row = {
            "init_glucose":        patient.glucose,
            "init_bp_systolic":    patient.bp_systolic,
            "init_bmi":            patient.bmi,
            "init_cholesterol":    patient.cholesterol,
            "init_sleep_hours":    patient.sleep_hours,
            "init_stress_level":   patient.stress_level,
            "init_activity_score": patient.activity_score,
            "init_hba1c":          patient.hba1c,
            "intervention":        INTERVENTION_CODES[intervention],
            "month":               month,
        }
        X     = pd.DataFrame([row])[features]
        pred  = model.predict(scaler.transform(X))[0]
        state = {feat: round(float(val), 2) for feat, val in zip(targets, pred)}
        results[month] = {
            **state,
            "composite_risk": compute_composite_risk(state),
            "risk_level":     risk_label(compute_composite_risk(state)),
        }
    return results


def get_shap_factors(patient: PatientProfile, model_type: str) -> list:
    if model_type == "diabetes":
        artifact  = MODELS["diabetes"]
        explainer = MODELS["diabetes_shap"]
        features  = artifact["features"]
        row = {
            "Pregnancies": patient.pregnancies, "Glucose": patient.glucose,
            "BloodPressure": patient.bp_systolic, "SkinThickness": patient.skin_thickness,
            "Insulin": patient.insulin, "BMI": patient.bmi,
            "DiabetesPedigreeFunction": patient.diabetes_pedigree, "Age": patient.age,
        }
    else:
        artifact  = MODELS["heart"]
        explainer = MODELS["heart_shap"]
        features  = artifact["features"]
        row = {f: 0 for f in features}
        row.update({"age": patient.age, "trestbps": patient.bp_systolic, "chol": patient.cholesterol})

    X         = pd.DataFrame([row])[features]
    shap_vals = explainer.shap_values(X)[0]
    factors   = sorted(zip(features, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   feat,
            "impact":    round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk",
            "magnitude": "high" if abs(val) > 0.5 else "moderate" if abs(val) > 0.2 else "low",
        }
        for feat, val in factors[:5]
    ]


def generate_counterfactuals(patient_dict: dict) -> list:
    base_state = {k: patient_dict[k] for k in HEALTH_FEATURES}
    base_risk  = compute_composite_risk(base_state)
    results    = []
    for feature, cfg in CF_FEATURES.items():
        current_val      = base_state[feature]
        best_val, best_risk = current_val, base_risk
        steps = int((cfg["max"] - cfg["min"]) / cfg["step"])
        for i in range(steps + 1):
            candidate = round(cfg["min"] + i * cfg["step"], 2)
            if candidate == current_val:
                continue
            test_risk = compute_composite_risk({**base_state, feature: candidate})
            if test_risk < best_risk:
                best_risk, best_val = test_risk, candidate
        delta = best_risk - base_risk
        if abs(delta) < 0.001:
            continue
        action = "reduce" if best_val < current_val else "increase"
        results.append({
            "feature": feature, "label": cfg["label"],
            "current_value": round(current_val, 2), "target_value": round(best_val, 2),
            "unit": cfg["unit"], "current_risk": round(base_risk, 4),
            "new_risk": round(best_risk, 4), "risk_reduction": round(abs(delta), 4),
            "risk_reduction_pct": round(abs(delta) * 100, 1), "action": action,
            "sentence": (
                f"If {cfg['label']} {action}d from "
                f"{round(current_val,1)}{cfg['unit']} to {round(best_val,1)}{cfg['unit']}, "
                f"composite risk drops from {round(base_risk*100,1)}% "
                f"to {round(best_risk*100,1)}% (-{round(abs(delta)*100,1)}%)"
            ),
        })
    results.sort(key=lambda x: x["risk_reduction"], reverse=True)
    return results[:6]


def simulate_trajectory_with_ci(patient_dict, intervention, checkpoints, n_samples=50):
    artifact          = MODELS["progression"]
    model, scaler     = artifact["model"], artifact["scaler"]
    features, targets = artifact["features"], artifact["targets"]
    all_risks         = defaultdict(list)
    for _ in range(n_samples):
        for month in checkpoints:
            row = {
                "init_glucose":        patient_dict["glucose"]        + random.gauss(0, 4),
                "init_bp_systolic":    patient_dict["bp_systolic"]    + random.gauss(0, 3),
                "init_bmi":            patient_dict["bmi"]            + random.gauss(0, 0.3),
                "init_cholesterol":    patient_dict["cholesterol"]    + random.gauss(0, 6),
                "init_sleep_hours":    patient_dict["sleep_hours"]    + random.gauss(0, 0.3),
                "init_stress_level":   patient_dict["stress_level"]   + random.gauss(0, 5),
                "init_activity_score": patient_dict["activity_score"] + random.gauss(0, 5),
                "init_hba1c":          patient_dict["hba1c"]          + random.gauss(0, 0.2),
                "intervention": INTERVENTION_CODES[intervention],
                "month":        month,
            }
            X    = pd.DataFrame([row])[features]
            pred = model.predict(scaler.transform(X))[0]
            state = {feat: float(val) for feat, val in zip(targets, pred)}
            all_risks[month].append(compute_composite_risk(state))
    result = {}
    for month in checkpoints:
        s = all_risks[month]
        result[month] = {
            "low":  round(float(np.percentile(s, 10)), 4),
            "mid":  round(float(np.percentile(s, 50)), 4),
            "high": round(float(np.percentile(s, 90)), 4),
            "mean": round(float(np.mean(s)), 4),
            "risk_level": risk_label(float(np.median(s))),
        }
    return result


def rl_optimize_treatment(patient_dict, horizon=12, episodes=200):
    interventions = list(INTERVENTION_CODES.keys())
    n_actions     = len(interventions)
    checkpoints   = list(range(0, horizon + 1, 3))
    Q             = np.zeros((5, 5, n_actions))
    alpha, gamma  = 0.1, 0.95

    def rb(r): return min(int(r * 5), 4)
    def mb(m): return min(m // 3, 4)

    def step_sim(p, interv, month):
        art = MODELS["progression"]
        row = {f"init_{k}": p[k] for k in HEALTH_FEATURES}
        row["intervention"] = INTERVENTION_CODES[interv]
        row["month"]        = month
        X    = pd.DataFrame([row])[art["features"]]
        pred = art["model"].predict(art["scaler"].transform(X))[0]
        return compute_composite_risk({feat: float(val) for feat, val in zip(art["targets"], pred)})

    for ep in range(episodes):
        epsilon  = max(0.05, 1.0 - ep / (episodes * 0.7))
        cur_risk = step_sim(patient_dict, "none", 0)
        for ci, month in enumerate(checkpoints[:-1]):
            action_idx = random.randint(0, n_actions-1) if random.random() < epsilon \
                         else int(np.argmax(Q[mb(month), rb(cur_risk)]))
            interv     = interventions[action_idx]
            next_month = checkpoints[ci + 1]
            next_risk  = step_sim(patient_dict, interv, next_month)
            Q[mb(month), rb(cur_risk), action_idx] += alpha * (
                -next_risk + gamma * np.max(Q[mb(next_month), rb(next_risk)])
                - Q[mb(month), rb(cur_risk), action_idx]
            )
            cur_risk = next_risk

    base_risk = step_sim(patient_dict, "none", 0)
    sequence, cur_risk = [], base_risk
    for ci, month in enumerate(checkpoints[:-1]):
        best_action = int(np.argmax(Q[mb(month), rb(cur_risk)]))
        interv      = interventions[best_action]
        next_month  = checkpoints[ci + 1]
        next_risk   = step_sim(patient_dict, interv, next_month)
        sequence.append({
            "month": month, "to_month": next_month, "intervention": interv,
            "label": interv.replace("_", " ").title(),
            "risk_at_start": round(cur_risk, 4), "risk_at_end": round(next_risk, 4),
            "delta": round(next_risk - cur_risk, 4),
        })
        cur_risk = next_risk

    no_treat_final = step_sim(patient_dict, "none", horizon)
    return {
        "base_risk": round(base_risk, 4), "optimized_final_risk": round(cur_risk, 4),
        "no_treatment_risk": round(no_treat_final, 4),
        "total_risk_reduction_pct": round((base_risk - cur_risk) * 100, 1),
        "optimal_sequence": sequence,
        "summary": (
            f"RL optimiser reduced risk from {round(base_risk*100,1)}% "
            f"to {round(cur_risk*100,1)}% over {horizon} months "
            f"(vs {round(no_treat_final*100,1)}% with no treatment)."
        ),
    }

# -----------------------------------------------------------------------------
# RL NEURAL AGENTS
# -----------------------------------------------------------------------------

_RL_FEATURE_BOUNDS = {
    "glucose": (70,250), "bp_systolic": (90,180), "bmi": (18,45),
    "cholesterol": (130,320), "sleep_hours": (3,10),
    "stress_level": (0,100), "activity_score": (0,100), "hba1c": (4.5,12),
}
_RL_INTERVENTIONS      = ["none","metformin","lifestyle","combined","statin","sleep_therapy"]
_RL_INTERVENTION_CODES = {k: i for i, k in enumerate(_RL_INTERVENTIONS)}
_RL_STEPS              = 8
_RL_MONTHS_PER_STEP    = 3


class _DQNNet(nn.Module):
    def __init__(self, state_dim=8, n_actions=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,128),       nn.LayerNorm(128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.adv   = nn.Linear(128, n_actions)
    def forward(self, x):
        h = self.shared(x)
        return self.value(h) + (self.adv(h) - self.adv(h).mean(-1, keepdim=True))


class _ActorCritic(nn.Module):
    def __init__(self, state_dim=8, n_actions=6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,128),       nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,128),       nn.LayerNorm(128), nn.ReLU(),
        )
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h)


def _normalize_state(state_dict):
    vec = []
    for feat in HEALTH_FEATURES:
        lo, hi = _RL_FEATURE_BOUNDS[feat]
        vec.append(float(np.clip((state_dict[feat] - lo) / (hi - lo), 0, 1)))
    return torch.FloatTensor(vec).unsqueeze(0)


def _rl_step(state_dict, intervention, month, artifact):
    row = {f"init_{k}": state_dict[k] for k in HEALTH_FEATURES}
    row["intervention"] = _RL_INTERVENTION_CODES[intervention]
    row["month"]        = month
    X    = pd.DataFrame([row])[artifact["features"]]
    pred = artifact["model"].predict(artifact["scaler"].transform(X))[0]
    result = {feat: float(val) for feat, val in zip(artifact["targets"], pred)}
    for feat, (lo, hi) in _RL_FEATURE_BOUNDS.items():
        result[feat] = float(np.clip(result[feat], lo, hi))
    return result


def _run_rl_inference(patient_dict, agent_type="dqn"):
    artifact = MODELS["progression"]
    if agent_type == "dqn":
        net  = _DQNNet()
        ckpt = torch.load("models/dqn_agent.pt", map_location="cpu")
        net.load_state_dict(ckpt["online"])
        net.eval()
        def select(st):
            with torch.no_grad(): return int(net(st).argmax(dim=1).item())
    else:
        net  = _ActorCritic()
        ckpt = torch.load("models/ppo_agent.pt", map_location="cpu")
        net.load_state_dict(ckpt["ac"])
        net.eval()
        def select(st):
            with torch.no_grad():
                logits, _ = net(st)
                return int(logits.argmax(dim=1).item())

    state     = {k: float(patient_dict[k]) for k in HEALTH_FEATURES}
    base_risk = compute_composite_risk(state)
    sequence  = []
    for step in range(_RL_STEPS):
        action_idx   = select(_normalize_state(state))
        intervention = _RL_INTERVENTIONS[action_idx]
        month_next   = (step + 1) * _RL_MONTHS_PER_STEP
        prev_risk    = compute_composite_risk(state)
        next_state   = _rl_step(state, intervention, month_next, artifact)
        next_risk    = compute_composite_risk(next_state)
        sequence.append({
            "step": step+1, "month": month_next, "intervention": intervention,
            "risk": round(next_risk,4), "prev_risk": round(prev_risk,4),
            "delta": round(next_risk - prev_risk, 4),
        })
        state = next_state

    final_risk = compute_composite_risk(state)
    return {
        "agent": agent_type.upper(), "base_risk": round(base_risk,4),
        "final_risk": round(final_risk,4),
        "total_risk_reduction_pct": round((base_risk - final_risk)*100, 1),
        "optimal_sequence": sequence,
        "summary": (
            f"{agent_type.upper()} agent reduced composite risk from "
            f"{round(base_risk*100,1)}% to {round(final_risk*100,1)}% "
            f"over 24 months (-{round((base_risk-final_risk)*100,1)}%)."
        ),
    }


def build_patient_context(patient_dict, risk_context):
    lines = ["=== PATIENT DATA ==="]
    if patient_dict:
        lines += [
            f"Age: {patient_dict.get('age')}, Gender: {patient_dict.get('gender')}",
            f"Glucose: {patient_dict.get('glucose')} mg/dL,  BP: {patient_dict.get('bp_systolic')} mmHg",
            f"BMI: {patient_dict.get('bmi')},  Cholesterol: {patient_dict.get('cholesterol')} mg/dL",
            f"HbA1c: {patient_dict.get('hba1c')}%,  Sleep: {patient_dict.get('sleep_hours')} hrs",
            f"Stress: {patient_dict.get('stress_level')}/100,  Activity: {patient_dict.get('activity_score')}/100",
        ]
    if risk_context:
        lines.append("\n=== RISK SCORES ===")
        for key in ("diabetes", "cardiovascular", "composite"):
            if key in risk_context:
                r = risk_context[key]
                lines.append(f"{key.title()}: {round(r.get('risk_score',0)*100,1)}% ({r.get('risk_level','?')})")
    return "\n".join(lines)

# =============================================================================
# ROUTES
# =============================================================================

# -----------------------------------------------------------------------------
# AUTH
# -----------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest):
    """
    Create a new account.
    Returns a JWT token immediately so the user is logged in right after signup.
    """
    if users_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")

    if users_col.find_one({"$or": [{"email": req.email}, {"username": req.username}]}):
        raise HTTPException(status_code=400, detail="Username or email already registered.")

    user_id = str(uuid.uuid4())
    users_col.insert_one({
        "user_id":         user_id,
        "username":        req.username,
        "email":           req.email,
        "full_name":       req.full_name or "",
        "hashed_password": hash_password(req.password),
        "created_at":      datetime.utcnow().isoformat(),
    })

    token = create_access_token({"sub": user_id})
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        username=req.username,
        full_name=req.full_name or "",
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Login with username (or email) + password.
    Returns a JWT bearer token valid for 24 hours.
    In Swagger UI click 'Authorize' and paste the token, or send it as:
      Authorization: Bearer <token>
    """
    if users_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")

    user = users_col.find_one(
        {"$or": [{"username": form.username}, {"email": form.username}]}
    )
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    token = create_access_token({"sub": user["user_id"]})
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        username=user["username"],
        full_name=user.get("full_name", ""),
    )


@app.get("/auth/me", response_model=UserInfo, tags=["Auth"])
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the profile of the currently logged-in user."""
    return UserInfo(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user.get("full_name", ""),
    )

# -----------------------------------------------------------------------------
# INFRASTRUCTURE
# -----------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":           "ok",
        "models_loaded":    list(MODELS.keys()),
        "patients_stored":  patients_col.count_documents({}) if patients_col is not None else 0,
        "users_registered": users_col.count_documents({})    if users_col    is not None else 0,
        "mongo_connected":  patients_col is not None,
        "groq_enabled":     groq_client is not None,
        "timestamp":        datetime.utcnow().isoformat(),
    }


@app.post("/patient/create", tags=["Patients"])
def create_patient(
    patient: PatientProfile,
    current_user: dict = Depends(get_current_user),
):
    """Save a new patient profile. Only you can access patients you create."""
    if patients_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")

    pid = str(uuid.uuid4())[:8]
    patients_col.insert_one({
        "patient_id": pid,
        "user_id":    current_user["user_id"],
        "profile":    patient.dict(),
        "created_at": datetime.utcnow().isoformat(),
    })
    return {
        "patient_id": pid,
        "message":    f"Patient '{patient.name}' created successfully.",
        "profile":    patient.dict(),
    }


@app.get("/patient/{patient_id}", tags=["Patients"])
def get_patient(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve one of your saved patient profiles by ID."""
    doc = _fetch_patient_doc(patient_id, current_user["user_id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return doc


@app.get("/patients", tags=["Patients"])
def list_patients(current_user: dict = Depends(get_current_user)):
    """List all patient profiles belonging to you."""
    if patients_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured.")
    docs = list(patients_col.find({"user_id": current_user["user_id"]}, {"_id": 0}))
    return {"patients": docs, "count": len(docs)}

# -----------------------------------------------------------------------------
# RISK PREDICTION
# -----------------------------------------------------------------------------

@app.post("/predict/risk", tags=["ML"])
def predict_risk(
    req: RiskRequest,
    current_user: dict = Depends(get_current_user),
):
    patient    = resolve_patient(req.patient_id, req.patient, current_user["user_id"])
    d_artifact = MODELS["diabetes"]
    d_row = {
        "Pregnancies": patient.pregnancies, "Glucose": patient.glucose,
        "BloodPressure": patient.bp_systolic, "SkinThickness": patient.skin_thickness,
        "Insulin": patient.insulin, "BMI": patient.bmi,
        "DiabetesPedigreeFunction": patient.diabetes_pedigree, "Age": patient.age,
    }
    d_X     = pd.DataFrame([d_row])[d_artifact["features"]]
    d_score = float(d_artifact["model"].predict_proba(d_X)[0][1])

    h_artifact = MODELS["heart"]
    h_row = {f: 0 for f in h_artifact["features"]}
    h_row.update({
        "age": patient.age, "trestbps": patient.bp_systolic, "chol": patient.cholesterol,
        "sex": 1 if patient.gender == "male" else 0,
        "fbs": 1 if patient.glucose > 120 else 0,
    })
    h_X     = pd.DataFrame([h_row])[h_artifact["features"]]
    h_score = float(h_artifact["model"].predict_proba(h_X)[0][1])

    composite = compute_composite_risk({k: getattr(patient, k) for k in HEALTH_FEATURES})
    return {
        "diabetes":      {"risk_score": round(d_score,4), "risk_level": risk_label(d_score),
                          "percentage": f"{d_score:.1%}", "top_factors": get_shap_factors(patient,"diabetes")},
        "cardiovascular":{"risk_score": round(h_score,4), "risk_level": risk_label(h_score),
                          "percentage": f"{h_score:.1%}", "top_factors": get_shap_factors(patient,"heart")},
        "composite":     {"risk_score": composite,        "risk_level": risk_label(composite)},
    }

# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------

@app.post("/simulate/trajectory", tags=["Simulation"])
def simulate_patient_trajectory(
    req: SimulationRequest,
    current_user: dict = Depends(get_current_user),
):
    if req.intervention not in INTERVENTION_CODES:
        raise HTTPException(status_code=400, detail=f"Invalid intervention. Choose from: {list(INTERVENTION_CODES.keys())}")
    patient    = resolve_patient(req.patient_id, req.patient, current_user["user_id"])
    trajectory = simulate_trajectory(patient, req.intervention, req.checkpoints)
    return {
        "intervention": req.intervention, "checkpoints": req.checkpoints, "trajectory": trajectory,
        "summary": {
            "risk_at_start": trajectory[req.checkpoints[0]]["composite_risk"],
            "risk_at_end":   trajectory[req.checkpoints[-1]]["composite_risk"],
            "risk_delta":    round(trajectory[req.checkpoints[-1]]["composite_risk"] - trajectory[req.checkpoints[0]]["composite_risk"], 4),
            "trend": "improving" if trajectory[req.checkpoints[-1]]["composite_risk"] < trajectory[req.checkpoints[0]]["composite_risk"] else "worsening",
        },
    }


@app.post("/simulate/trajectory_ci", tags=["Simulation"])
def trajectory_with_confidence(
    req: TrajectoryWithCIRequest,
    current_user: dict = Depends(get_current_user),
):
    if req.intervention not in INTERVENTION_CODES:
        raise HTTPException(status_code=400, detail="Invalid intervention.")
    patient_dict = resolve_patient_dict(req.patient_id, req.patient, current_user["user_id"])
    ci_data      = simulate_trajectory_with_ci(patient_dict, req.intervention, req.checkpoints, req.n_samples)
    start = ci_data[req.checkpoints[0]]["mid"]
    end   = ci_data[req.checkpoints[-1]]["mid"]
    return {
        "intervention": req.intervention, "checkpoints": req.checkpoints, "trajectory_ci": ci_data,
        "summary": {
            "risk_at_start": start, "risk_at_end": end, "risk_delta": round(end - start, 4),
            "trend": "improving" if end < start else "worsening",
            "ci_width_at_end": round(ci_data[req.checkpoints[-1]]["high"] - ci_data[req.checkpoints[-1]]["low"], 4),
        },
    }


@app.post("/simulate/scenarios", tags=["Simulation"])
def compare_scenarios(
    req: ScenarioRequest,
    current_user: dict = Depends(get_current_user),
):
    patient = resolve_patient(req.patient_id, req.patient, current_user["user_id"])
    results = {}
    for intervention in INTERVENTION_CODES:
        traj        = simulate_trajectory(patient, intervention, req.checkpoints)
        final_month = req.checkpoints[-1]
        results[intervention] = {
            "trajectory": traj,
            "risk_at_end": traj[final_month]["composite_risk"],
            "risk_level":  traj[final_month]["risk_level"],
            "risk_delta":  round(traj[final_month]["composite_risk"] - traj[0]["composite_risk"], 4),
            "glucose_at_end": traj[final_month]["glucose"],
            "bp_at_end":      traj[final_month]["bp_systolic"],
            "bmi_at_end":     traj[final_month]["bmi"],
        }
    ranked = sorted(results.items(), key=lambda x: x[1]["risk_at_end"])
    return {
        "patient_baseline_risk": compute_composite_risk({k: getattr(patient, k) for k in HEALTH_FEATURES}),
        "scenarios": results,
        "ranking": [{"rank": i+1, "intervention": k, "final_risk": v["risk_at_end"]} for i, (k,v) in enumerate(ranked)],
        "best_intervention":  ranked[0][0],
        "worst_intervention": ranked[-1][0],
    }

# -----------------------------------------------------------------------------
# EXPLANATION
# -----------------------------------------------------------------------------

@app.post("/explain", tags=["Explanation"])
def explain_risk(
    req: ExplainRequest,
    current_user: dict = Depends(get_current_user),
):
    patient    = resolve_patient(req.patient_id, req.patient, current_user["user_id"])
    factors    = get_shap_factors(patient, req.model_type)
    top        = factors[0]["feature"] if factors else "unknown"
    drivers    = [f["feature"] for f in factors if f["direction"] == "increases risk"]
    protective = [f["feature"] for f in factors if f["direction"] == "decreases risk"]
    explanation = f"The primary driver of this patient's {req.model_type} risk is {top}. "
    if drivers:    explanation += f"Key risk-increasing factors: {', '.join(drivers[:3])}. "
    if protective: explanation += f"Protective factors: {', '.join(protective[:2])}."
    return {
        "model_type": req.model_type, "top_factors": factors, "explanation": explanation,
        "recommendation": f"Focus on improving {drivers[0] if drivers else 'overall lifestyle'} first.",
    }


@app.post("/explain/counterfactual", tags=["Explanation"])
def counterfactual_explain(
    req: CounterfactualRequest,
    current_user: dict = Depends(get_current_user),
):
    patient_dict    = resolve_patient_dict(req.patient_id, req.patient, current_user["user_id"])
    counterfactuals = generate_counterfactuals(patient_dict)
    base_state      = {k: patient_dict[k] for k in HEALTH_FEATURES}
    top             = counterfactuals[0] if counterfactuals else {}
    return {
        "counterfactuals": counterfactuals,
        "summary":  top.get("sentence", "No significant counterfactuals found."),
        "base_risk": compute_composite_risk(base_state),
    }

# -----------------------------------------------------------------------------
# RL OPTIMISERS
# -----------------------------------------------------------------------------

@app.post("/optimize/treatment", tags=["RL"])
def optimize_treatment(
    req: OptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    patient_dict = resolve_patient_dict(req.patient_id, req.patient, current_user["user_id"])
    return rl_optimize_treatment(patient_dict, req.horizon, req.episodes)


@app.post("/optimize/rl", tags=["RL"])
def optimize_rl(
    req: RLOptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    if req.agent_type not in ("dqn", "ppo"):
        raise HTTPException(status_code=400, detail="agent_type must be 'dqn' or 'ppo'")
    model_path = f"models/{req.agent_type}_agent.pt"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail=f"RL model not found: {model_path}. Run medtwin_rl.py first.")
    patient_dict = resolve_patient_dict(req.patient_id, req.patient, current_user["user_id"])
    return _run_rl_inference(patient_dict, req.agent_type)

# -----------------------------------------------------------------------------
# CHAT
# -----------------------------------------------------------------------------

@app.post("/chat", tags=["Chat"])
def chat_with_medtwin(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq API key not configured.")

    patient_dict = None
    try:
        patient_dict = resolve_patient_dict(req.patient_id, req.patient, current_user["user_id"])
    except Exception:
        pass

    context  = build_patient_context(patient_dict or {}, req.risk_context or {})
    messages = [{"role": "system", "content": GROQ_SYSTEM_PROMPT + "\n\n" + context}]
    for h in (req.history or []):
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.question})

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=400,
        temperature=0.4,
    )
    return {
        "answer": completion.choices[0].message.content,
        "model":  "llama-3.3-70b-versatile",
        "tokens": completion.usage.total_tokens,
    }

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("medtwin_phase3_backend:app", host="0.0.0.0", port=8000, reload=True)