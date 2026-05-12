# 🧬 MedTwin AI — Digital Twin Patient Simulation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-RL-EE4C2C?logo=pytorch&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-black?logo=groq&logoColor=white)

> ⚠️ **NOT FOR CLINICAL USE.** This is a student research prototype built for academic and portfolio purposes only. All predictions are simulated and must never be used for real medical decisions.

---

## What is MedTwin AI?

Most health AI tools answer one question: *"Is this patient at risk?"*

MedTwin AI goes further — it answers: **"What happens NEXT, and what should we do about it?"**

You input a virtual patient's health profile, and MedTwin simulates:

- **Disease risk** scored by XGBoost models (diabetes + cardiovascular)
- **24-month health progression** across 6 intervention scenarios
- **Optimal treatment sequences** discovered by Reinforcement Learning (DQN + PPO)
- **Explainability** via SHAP values and natural language summaries
- **Counterfactual analysis** — *"If glucose drops to 110, risk falls from 55% → 38%"*
- **AI chat assistant** powered by Groq (Llama 3.3 70B) with full patient context

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│   Patient · Risk · Scenarios · Timeline · RL · Chat     │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│   /auth  /patient  /predict  /simulate  /explain  /chat  │
└──────┬────────────┬──────────────┬───────────────────────┘
       │            │              │
┌──────▼──┐  ┌──────▼──────┐  ┌───▼──────────────────────┐
│ MongoDB  │  │  ML Models   │  │     RL Agents             │
│  Atlas   │  │  XGBoost     │  │  DQN (Dueling Double)     │
│  Users   │  │  GBM Prog.   │  │  PPO (Actor-Critic + GAE) │
│ Patients │  │  SHAP Expls. │  │  PyTorch · 600 episodes   │
└──────────┘  └─────────────┘  └──────────────────────────┘
```

---

## Features

### Tab 1 — Patient Profile
Create a virtual patient with 15 clinical parameters (glucose, BMI, blood pressure, HbA1c, cholesterol, age, smoking, etc.). System health status and activity log shown live.

### Tab 2 — Risk Scoring
Dual XGBoost classifiers predict diabetes and cardiovascular risk with probability scores. SHAP bar charts show which features drive the prediction.

### Tab 3 — Scenario Comparison
Simulate all 6 interventions simultaneously and compare 24-month outcomes:
- No treatment · Metformin · Lifestyle change · Combined · Statin · Sleep therapy

### Tab 4 — Progression Timeline
Area charts for glucose, blood pressure, HbA1c, and composite risk across all checkpoints (0, 3, 6, 9, 12, 18, 24 months).

### Tab 5 — RL Treatment Optimizer
Select DQN or PPO agent. The agent recommends a month-by-month treatment sequence optimized to minimize long-term risk. Demo result: **54% risk reduction over 24 months**.

### Tab 6 — AI Chat
Multi-turn conversation with Groq Llama 3.3 70B. The model has full access to the current patient's data and simulation results.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js, Recharts, inline CSS |
| Backend | FastAPI, Uvicorn, Pydantic |
| Risk Models | XGBoost (AUC: Diabetes 0.831, Heart 0.881) |
| Progression | GBM MultiOutputRegressor (R² > 0.97) |
| RL Agents | PyTorch — Dueling Double DQN + PPO |
| Explainability | SHAP |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Database | MongoDB Atlas (Motor async driver) |
| Auth | JWT + bcrypt |

---

## Project Structure

```
MedTwin/
├── medtwin_phase1.py          # XGBoost risk scorer (diabetes + heart)
├── medtwin_phase2.py          # GBM progression forecaster
├── medtwin_phase3_backend.py  # FastAPI backend — all endpoints
├── medtwin_rl.py              # DQN + PPO training
├── .env                       # API keys (not committed)
├── models/                    # Trained model files (not committed)
│   ├── diabetes_risk.pkl
│   ├── heart_risk.pkl
│   ├── progression_model.pkl
│   ├── dqn_agent.pt
│   └── ppo_agent.pt
├── data/                      # Training datasets (not committed)
│   ├── diabetes.csv           # Pima Indians dataset
│   └── heart.csv              # Cleveland Heart dataset
└── medtwin-frontend/
    └── src/
        ├── App.js
        ├── Loginpage.jsx
        └── MedTwinDashboard.jsx
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB Atlas account (free tier works)
- Groq API key (free at console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/devasadhu/medtwin-ai.git
cd medtwin-ai
```

### 2. Set up environment variables
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_key_here
MONGODB_URI=mongodb+srv://your_connection_string
SECRET_KEY=any_random_string_for_jwt
```

### 3. Install Python dependencies
```bash
pip install fastapi uvicorn pydantic joblib numpy pandas groq python-dotenv \
            xgboost shap scikit-learn torch motor pymongo \
            python-jose[cryptography] passlib[bcrypt]
```

### 4. Train the models
```bash
python medtwin_phase1.py    # trains diabetes_risk.pkl + heart_risk.pkl
python medtwin_phase2.py    # trains progression_model.pkl
python medtwin_rl.py        # trains dqn_agent.pt + ppo_agent.pt
```
> Models are excluded from the repo due to size. Run these once and they save to `models/`.

### 5. Start the backend
```bash
python medtwin_phase3_backend.py
# Running at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 6. Start the frontend
```bash
cd medtwin-frontend
npm install
npm start
# Running at http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create user account |
| POST | `/auth/login` | Get JWT token |
| GET | `/auth/me` | Current user info |
| POST | `/patient/create` | Create virtual patient |
| GET | `/patient/{id}` | Fetch patient data |
| POST | `/predict/risk` | Diabetes + heart risk + SHAP |
| POST | `/simulate/trajectory` | Single intervention forecast |
| POST | `/simulate/trajectory_ci` | Monte Carlo confidence bands |
| POST | `/simulate/scenarios` | All 6 interventions compared |
| POST | `/explain` | SHAP explanation + NL summary |
| POST | `/explain/counterfactual` | "If X changes, risk changes by Y%" |
| POST | `/optimize/rl` | DQN or PPO treatment sequence |
| POST | `/chat` | Groq LLM with patient context |

Full interactive docs: `http://localhost:8000/docs`

---

## ML & RL Design Notes

**Risk Models** use real public datasets:
- Pima Indians Diabetes dataset (768 samples) — AUC 0.831
- Cleveland Heart Disease dataset (303 samples) — AUC 0.881

**Progression Model** uses synthetic longitudinal data (50,000 samples × 8 checkpoints) because real longitudinal patient data (e.g. MIMIC-III) requires IRB credentialing. This is standard practice for student prototypes and is documented transparently.

**RL Environment:**
- State: 8-dimensional normalized health vector
- Actions: 6 interventions
- Reward: negative composite risk score
- Episode: 8 steps × 3 months = 24-month horizon
- Composite risk: `0.30×glucose + 0.20×BP + 0.15×BMI + 0.15×cholesterol + 0.12×HbA1c + 0.08×lifestyle`

**DQN:** Dueling Double DQN with experience replay (20K buffer) and soft target updates.  
**PPO:** Clipped surrogate objective with GAE (λ=0.95), actor-critic architecture, entropy bonus.  
Both trained for 600 episodes on a diverse synthetic patient cohort.

---

## Disclaimer

This project is a **B.Tech final year prototype** built for academic learning and portfolio demonstration. It uses:
- Public datasets (Pima Indians, Cleveland Heart)
- Synthetic longitudinal data for progression modeling
- Simulated RL environments

**It is not validated for clinical use and must never be used to inform real medical decisions.**

---

## Author

**Sadhana** — B.Tech AI, 3rd Year, SVNIT  
GitHub: [@devasadhu](https://github.com/devasadhu)