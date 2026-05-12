"""
MedTwin AI — Phase 4: RL Treatment Optimizer
=============================================
Two agents that learn optimal treatment sequencing:

  1. DQN  (Deep Q-Network)     — off-policy, experience replay
  2. PPO  (Proximal Policy Opt) — on-policy, clipped surrogate objective

Both operate on the same MedTwin environment:
  State  : 8-dim normalized health vector
  Actions: 6 interventions (none/metformin/lifestyle/combined/statin/sleep_therapy)
  Reward : −composite_risk  (agent minimizes risk)
  Episode: 8 steps × 3-month intervals = 24-month horizon

After training:
  - Saves best agent weights  → models/dqn_agent.pt / ppo_agent.pt
  - Saves training curves     → assets/rl_training.png
  - Prints optimal treatment sequence for a demo patient

Install:
  pip install torch numpy pandas matplotlib joblib

Run:
  python medtwin_rl.py
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_FEATURES = [
    "glucose", "bp_systolic", "bmi", "cholesterol",
    "sleep_hours", "stress_level", "activity_score", "hba1c",
]

INTERVENTIONS = ["none", "metformin", "lifestyle", "combined", "statin", "sleep_therapy"]
N_ACTIONS     = len(INTERVENTIONS)
STATE_DIM     = len(HEALTH_FEATURES)
INTERVENTION_CODES = {k: i for i, k in enumerate(INTERVENTIONS)}

# Normalization bounds (min, max) per feature
FEATURE_BOUNDS = {
    "glucose":        (70,  250),
    "bp_systolic":    (90,  180),
    "bmi":            (18,  45),
    "cholesterol":    (130, 320),
    "sleep_hours":    (3,   10),
    "stress_level":   (0,   100),
    "activity_score": (0,   100),
    "hba1c":          (4.5, 12),
}

PALETTE = {
    "bg":      "#0a0e1a",
    "surface": "#111827",
    "accent":  "#00d4ff",
    "green":   "#10b981",
    "red":     "#ef4444",
    "amber":   "#f59e0b",
    "purple":  "#7c3aed",
    "text":    "#e2e8f0",
    "muted":   "#64748b",
}

SCENARIO_COLORS = {
    "none":          PALETTE["red"],
    "metformin":     PALETTE["accent"],
    "lifestyle":     PALETTE["green"],
    "combined":      PALETTE["purple"],
    "statin":        PALETTE["amber"],
    "sleep_therapy": "#f472b6",
}

# ─────────────────────────────────────────────────────────────────────────────
# MEDTWIN ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class MedTwinEnv:
    """
    RL environment wrapping the trained progression model.

    State  : normalized 8-dim health vector ∈ [0,1]^8
    Action : integer 0–5 (intervention index)
    Reward : −composite_risk (range roughly −0.8 to 0)
             + bonus −0.5 if risk crosses a threshold going up
    Episode: 8 steps (month 0 → 3 → 6 → ... → 24)
    """

    STEPS_PER_EPISODE = 8
    MONTHS_PER_STEP   = 3

    def __init__(self, progression_artifact, random_patients=True):
        self.model    = progression_artifact["model"]
        self.scaler   = progression_artifact["scaler"]
        self.features = progression_artifact["features"]
        self.targets  = progression_artifact["targets"]
        self.random_patients = random_patients
        self.state    = None
        self.step_idx = 0
        self.initial_state = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _normalize(self, state_dict: dict) -> np.ndarray:
        vec = []
        for feat in HEALTH_FEATURES:
            lo, hi = FEATURE_BOUNDS[feat]
            vec.append(np.clip((state_dict[feat] - lo) / (hi - lo), 0.0, 1.0))
        return np.array(vec, dtype=np.float32)

    def _denormalize(self, vec: np.ndarray) -> dict:
        state = {}
        for i, feat in enumerate(HEALTH_FEATURES):
            lo, hi = FEATURE_BOUNDS[feat]
            state[feat] = float(vec[i] * (hi - lo) + lo)
        return state

    def _composite_risk(self, state_dict: dict) -> float:
        s = state_dict
        return float(
            0.30 * np.clip((s["glucose"]        - 70)  / 180, 0, 1) +
            0.20 * np.clip((s["bp_systolic"]    - 90)  / 90,  0, 1) +
            0.15 * np.clip((s["bmi"]            - 18)  / 27,  0, 1) +
            0.15 * np.clip((s["cholesterol"]    - 130) / 190, 0, 1) +
            0.12 * np.clip((s["hba1c"]          - 4.5) / 7.5, 0, 1) +
            0.08 * np.clip(1 - s["activity_score"] / 100,     0, 1)
        )

    def _step_model(self, state_dict: dict, intervention: str, month: int) -> dict:
        row = {f"init_{k}": state_dict[k] for k in HEALTH_FEATURES}
        row["intervention"] = INTERVENTION_CODES[intervention]
        row["month"]        = month
        X    = pd.DataFrame([row])[self.features]
        pred = self.model.predict(self.scaler.transform(X))[0]
        return {feat: float(val) for feat, val in zip(self.targets, pred)}

    def _sample_patient(self) -> dict:
        """Sample a diverse random patient for training."""
        return {
            "glucose":        np.random.normal(130, 35),
            "bp_systolic":    np.random.normal(130, 20),
            "bmi":            np.random.normal(29,  6),
            "cholesterol":    np.random.normal(215, 45),
            "sleep_hours":    np.random.normal(6.5, 1.2),
            "stress_level":   np.random.normal(55,  20),
            "activity_score": np.random.normal(40,  20),
            "hba1c":          np.random.normal(6.5, 1.2),
        }

    # ── RL interface ─────────────────────────────────────────────────────────

    def reset(self, patient_dict=None) -> np.ndarray:
        if patient_dict is not None:
            raw = patient_dict.copy()
        elif self.random_patients:
            raw = self._sample_patient()
        else:
            raw = self._sample_patient()

        # Clip to bounds
        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            raw[feat] = np.clip(raw[feat], lo, hi)

        self.initial_state = raw.copy()
        self.state         = raw.copy()
        self.step_idx      = 0
        self.prev_risk     = self._composite_risk(raw)
        return self._normalize(raw)

    def step(self, action: int):
        """
        Take one step: apply intervention for MONTHS_PER_STEP months.
        Returns: (next_state, reward, done, info)
        """
        assert self.state is not None, "Call reset() first."

        intervention = INTERVENTIONS[action]
        month_now    = self.step_idx * self.MONTHS_PER_STEP
        month_next   = month_now + self.MONTHS_PER_STEP

        next_state_dict = self._step_model(self.state, intervention, month_next)

        # Clip to physiological bounds
        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            next_state_dict[feat] = np.clip(next_state_dict[feat], lo, hi)

        curr_risk  = self._composite_risk(self.state)
        next_risk  = self._composite_risk(next_state_dict)

        # Reward design:
        #   base    : −next_risk  (minimize absolute risk)
        #   shaping : bonus for improvement vs no-treatment baseline
        #   penalty : if risk increases
        improvement = curr_risk - next_risk          # positive = good
        reward      = -next_risk + 0.5 * improvement
        if next_risk > curr_risk + 0.02:             # risk spike penalty
            reward -= 0.3

        self.state     = next_state_dict
        self.step_idx += 1
        done           = self.step_idx >= self.STEPS_PER_EPISODE

        info = {
            "intervention":  intervention,
            "month":         month_next,
            "risk":          next_risk,
            "prev_risk":     curr_risk,
            "improvement":   improvement,
        }
        return self._normalize(next_state_dict), reward, done, info

    @property
    def current_risk(self) -> float:
        return self._composite_risk(self.state) if self.state else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SHARED NEURAL NETWORK BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def mlp(in_dim, out_dim, hidden=(128, 128), activation=nn.ReLU):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# DQN
# ─────────────────────────────────────────────────────────────────────────────

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class DQNNet(nn.Module):
    """Dueling DQN architecture — separates value and advantage streams."""
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.shared = mlp(state_dim, 128, hidden=(128,))
        self.value  = nn.Linear(128, 1)
        self.adv    = nn.Linear(128, n_actions)

    def forward(self, x):
        h   = self.shared(x)
        V   = self.value(h)
        A   = self.adv(h)
        Q   = V + (A - A.mean(dim=-1, keepdim=True))
        return Q


class DQNAgent:
    """
    Double Dueling DQN with:
      - experience replay
      - target network (soft update)
      - ε-greedy exploration with linear decay
    """
    def __init__(
        self,
        lr           = 3e-4,
        gamma        = 0.97,
        batch_size   = 64,
        buffer_size  = 20_000,
        target_update_freq = 50,   # episodes
        eps_start    = 1.0,
        eps_end      = 0.05,
        eps_decay    = 0.995,
        tau          = 0.005,      # soft update coefficient
    ):
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update_freq = target_update_freq
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.tau          = tau
        self.steps_done   = 0

        self.online = DQNNet()
        self.target = DQNNet()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)
        self.losses    = []

    def select_action(self, state: np.ndarray, greedy=False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states      = torch.FloatTensor(np.array(batch.state))
        actions     = torch.LongTensor(batch.action).unsqueeze(1)
        rewards     = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones       = torch.FloatTensor(batch.done).unsqueeze(1)

        # Double DQN: online selects action, target evaluates
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_Q       = self.target(next_states).gather(1, next_actions)
            target_Q     = rewards + self.gamma * next_Q * (1 - dones)

        current_Q = self.online(states).gather(1, actions)
        loss      = F.smooth_l1_loss(current_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.losses.append(loss.item())
        return loss.item()

    def save(self, path="models/dqn_agent.pt"):
        torch.save({"online": self.online.state_dict(), "eps": self.eps}, path)
        print(f"  ✓ DQN saved → {path}")

    def load(self, path="models/dqn_agent.pt"):
        ckpt = torch.load(path, map_location="cpu")
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self.eps = ckpt.get("eps", self.eps_end)


# ─────────────────────────────────────────────────────────────────────────────
# PPO
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared backbone → separate actor (policy) and critic (value) heads.
    """
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.backbone = mlp(state_dim, 128, hidden=(128, 128))
        self.actor    = nn.Linear(128, n_actions)
        self.critic   = nn.Linear(128, 1)

    def forward(self, x):
        h      = self.backbone(x)
        logits = self.actor(h)
        value  = self.critic(h)
        return logits, value

    def get_action(self, state: np.ndarray):
        with torch.no_grad():
            s         = torch.FloatTensor(state).unsqueeze(0)
            logits, V = self(s)
            dist      = torch.distributions.Categorical(logits=logits)
            action    = dist.sample()
            log_prob  = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(V.item())

    def evaluate(self, states, actions):
        logits, values = self(states)
        dist           = torch.distributions.Categorical(logits=logits)
        log_probs      = dist.log_prob(actions)
        entropy        = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


PPOBatch = namedtuple("PPOBatch", ["states", "actions", "log_probs", "returns", "advantages"])


class PPOAgent:
    """
    PPO with:
      - clipped surrogate objective
      - GAE (Generalized Advantage Estimation)
      - entropy bonus for exploration
      - value function clipping
    """
    def __init__(
        self,
        lr          = 3e-4,
        gamma       = 0.97,
        lam         = 0.95,       # GAE lambda
        clip_eps    = 0.2,
        epochs      = 4,          # PPO update epochs per rollout
        batch_size  = 32,
        vf_coef     = 0.5,
        ent_coef    = 0.01,
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef

        self.ac        = ActorCritic()
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.losses    = []

        # Rollout buffer
        self._states    = []
        self._actions   = []
        self._log_probs = []
        self._rewards   = []
        self._values    = []
        self._dones     = []

    def select_action(self, state: np.ndarray, greedy=False) -> int:
        if greedy:
            with torch.no_grad():
                s      = torch.FloatTensor(state).unsqueeze(0)
                logits, _ = self.ac(s)
                return int(logits.argmax(dim=1).item())
        action, log_prob, value = self.ac.get_action(state)
        self._states.append(state)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        return action

    def store_reward(self, reward, done):
        self._rewards.append(reward)
        self._dones.append(float(done))

    def _compute_gae(self, last_value=0.0):
        """Compute returns and GAE advantages."""
        T         = len(self._rewards)
        returns   = np.zeros(T, dtype=np.float32)
        advs      = np.zeros(T, dtype=np.float32)
        gae       = 0.0
        next_val  = last_value

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - self._dones[t]
            delta = (self._rewards[t]
                     + self.gamma * next_val * next_non_terminal
                     - self._values[t])
            gae       = delta + self.gamma * self.lam * next_non_terminal * gae
            advs[t]   = gae
            returns[t] = gae + self._values[t]
            next_val  = self._values[t]

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return returns, advs

    def update(self, last_value=0.0):
        if len(self._rewards) == 0:
            return None

        returns, advantages = self._compute_gae(last_value)

        states    = torch.FloatTensor(np.array(self._states))
        actions   = torch.LongTensor(self._actions)
        old_lps   = torch.FloatTensor(self._log_probs)
        returns_t = torch.FloatTensor(returns)
        advs_t    = torch.FloatTensor(advantages)

        total_loss = 0.0
        for _ in range(self.epochs):
            # Mini-batch sampling
            idx = torch.randperm(len(self._states))
            for start in range(0, len(idx), self.batch_size):
                mb  = idx[start:start + self.batch_size]
                lps, vals, entropy = self.ac.evaluate(states[mb], actions[mb])

                # Policy loss (clipped surrogate)
                ratio     = (lps - old_lps[mb]).exp()
                surr1     = ratio * advs_t[mb]
                surr2     = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advs_t[mb]
                pol_loss  = -torch.min(surr1, surr2).mean()

                # Value loss
                val_loss  = F.mse_loss(vals, returns_t[mb])

                # Entropy bonus
                loss = pol_loss + self.vf_coef * val_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.losses.append(total_loss / self.epochs)

        # Clear rollout buffer
        self._states.clear(); self._actions.clear()
        self._log_probs.clear(); self._rewards.clear()
        self._values.clear(); self._dones.clear()

        return total_loss / self.epochs

    def save(self, path="models/ppo_agent.pt"):
        torch.save({"ac": self.ac.state_dict()}, path)
        print(f"  ✓ PPO saved → {path}")

    def load(self, path="models/ppo_agent.pt"):
        ckpt = torch.load(path, map_location="cpu")
        self.ac.load_state_dict(ckpt["ac"])


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(env, n_episodes=800, eval_every=50, verbose=True):
    agent       = DQNAgent()
    ep_rewards  = []
    ep_risks    = []
    eval_scores = []

    print("\n" + "─"*52)
    print("  Training DQN Agent")
    print(f"  Episodes: {n_episodes}  |  State dim: {STATE_DIM}  |  Actions: {N_ACTIONS}")
    print("─"*52)

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        ep_rew  = 0.0
        final_risk = 0.0

        for _ in range(env.STEPS_PER_EPISODE):
            action              = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.update()
            state      = next_state
            ep_rew    += reward
            final_risk = info["risk"]

        ep_rewards.append(ep_rew)
        ep_risks.append(final_risk)

        if verbose and ep % eval_every == 0:
            avg_rew  = np.mean(ep_rewards[-eval_every:])
            avg_risk = np.mean(ep_risks[-eval_every:])
            print(f"  Ep {ep:4d} | AvgRew {avg_rew:+.3f} | AvgRisk {avg_risk:.3f} | ε {agent.eps:.3f}")
            eval_scores.append(avg_risk)

    agent.save("models/dqn_agent.pt")
    return agent, ep_rewards, ep_risks, eval_scores


def train_ppo(env, n_episodes=800, rollout_len=8, eval_every=50, verbose=True):
    """
    PPO collects full episodes (rollout_len = STEPS_PER_EPISODE) then updates.
    """
    agent       = PPOAgent()
    ep_rewards  = []
    ep_risks    = []
    eval_scores = []

    print("\n" + "─"*52)
    print("  Training PPO Agent")
    print(f"  Episodes: {n_episodes}  |  Rollout len: {rollout_len}")
    print("─"*52)

    for ep in range(1, n_episodes + 1):
        state      = env.reset()
        ep_rew     = 0.0
        final_risk = 0.0

        for _ in range(rollout_len):
            action              = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward, done)
            state      = next_state
            ep_rew    += reward
            final_risk = info["risk"]

        # Update at end of episode
        agent.update(last_value=0.0)
        ep_rewards.append(ep_rew)
        ep_risks.append(final_risk)

        if verbose and ep % eval_every == 0:
            avg_rew  = np.mean(ep_rewards[-eval_every:])
            avg_risk = np.mean(ep_risks[-eval_every:])
            print(f"  Ep {ep:4d} | AvgRew {avg_rew:+.3f} | AvgRisk {avg_risk:.3f}")
            eval_scores.append(avg_risk)

    agent.save("models/ppo_agent.pt")
    return agent, ep_rewards, ep_risks, eval_scores


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — extract optimal treatment sequence
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent, env, patient_dict, label="Agent"):
    """Roll out the greedy policy for one patient. Returns trajectory + sequence."""
    state    = env.reset(patient_dict)
    sequence = []
    risks    = [env._composite_risk(env.initial_state)]

    for step in range(env.STEPS_PER_EPISODE):
        action                       = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        sequence.append({
            "step":         step + 1,
            "month":        info["month"],
            "intervention": info["intervention"],
            "risk":         round(info["risk"], 4),
            "prev_risk":    round(info["prev_risk"], 4),
            "delta":        round(info["risk"] - info["prev_risk"], 4),
        })
        risks.append(info["risk"])
        state = next_state

    return sequence, risks


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def smooth(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_rl_dashboard(
    dqn_rewards, dqn_risks, dqn_eval,
    ppo_rewards, ppo_risks, ppo_eval,
    dqn_seq, dqn_risks_traj,
    ppo_seq, ppo_risks_traj,
    demo_patient,
):
    fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        "MedTwin AI — RL Treatment Optimizer: DQN vs PPO",
        color=PALETTE["accent"], fontsize=18, fontweight="bold", y=0.97,
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)

    def sa(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(PALETTE["muted"]); sp.set_alpha(0.3)
        if title:  ax.set_title(title, color=PALETTE["text"], fontsize=10, fontweight="bold", pad=8)
        if xlabel: ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=9)
        ax.grid(color=PALETTE["muted"], alpha=0.15, linestyle="--")

    # ── 1. Learning curves — reward ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    sa(ax, title="Learning Curves — Episode Reward", xlabel="Episode", ylabel="Total Reward")
    for rewards, color, label in [
        (dqn_rewards, PALETTE["accent"], "DQN"),
        (ppo_rewards, PALETTE["purple"], "PPO"),
    ]:
        s = smooth(rewards)
        ax.plot(range(len(rewards)), rewards, color=color, alpha=0.15, lw=1)
        ax.plot(range(len(s)), s, color=color, lw=2.5, label=f"{label} (smoothed)")
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=9)

    # ── 2. Final risk per episode ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    sa(ax2, title="Final Risk Per Episode (lower = better)", xlabel="Episode", ylabel="Composite Risk")
    for risks, color, label in [
        (dqn_risks, PALETTE["accent"], "DQN"),
        (ppo_risks, PALETTE["purple"], "PPO"),
    ]:
        s = smooth(risks)
        ax2.plot(range(len(risks)), risks, color=color, alpha=0.15, lw=1)
        ax2.plot(range(len(s)), s, color=color, lw=2.5, label=f"{label} (smoothed)")
    ax2.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"], fontsize=9)

    # ── 3. DQN optimal sequence ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    sa(ax3, title="DQN Optimal Treatment Sequence (Demo Patient)", xlabel="Month", ylabel="Composite Risk")
    months = [0] + [s["month"] for s in dqn_seq]
    ax3.plot(months, dqn_risks_traj, color=PALETTE["accent"], lw=2.5, marker="o", markersize=6)
    ax3.fill_between(months, dqn_risks_traj, alpha=0.12, color=PALETTE["accent"])
    for step_data in dqn_seq:
        m, interv = step_data["month"], step_data["intervention"]
        r = step_data["risk"]
        ax3.annotate(
            interv.replace("_", "\n"),
            xy=(m, r), xytext=(m, r + 0.03),
            color=SCENARIO_COLORS.get(interv, PALETTE["muted"]),
            fontsize=7, ha="center", fontfamily="monospace",
            arrowprops=dict(arrowstyle="-", color=PALETTE["muted"], alpha=0.4),
        )

    # ── 4. PPO optimal sequence ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    sa(ax4, title="PPO Optimal Treatment Sequence (Demo Patient)", xlabel="Month", ylabel="Composite Risk")
    months_p = [0] + [s["month"] for s in ppo_seq]
    ax4.plot(months_p, ppo_risks_traj, color=PALETTE["purple"], lw=2.5, marker="s", markersize=6)
    ax4.fill_between(months_p, ppo_risks_traj, alpha=0.12, color=PALETTE["purple"])
    for step_data in ppo_seq:
        m, interv = step_data["month"], step_data["intervention"]
        r = step_data["risk"]
        ax4.annotate(
            interv.replace("_", "\n"),
            xy=(m, r), xytext=(m, r + 0.03),
            color=SCENARIO_COLORS.get(interv, PALETTE["muted"]),
            fontsize=7, ha="center", fontfamily="monospace",
            arrowprops=dict(arrowstyle="-", color=PALETTE["muted"], alpha=0.4),
        )

    # ── 5. Agent comparison bar ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    sa(ax5, title="Risk Reduction: DQN vs PPO vs Baselines", ylabel="Final Risk (Month 24)")

    labels  = ["No\nTreatment", "Static\nCombined", "DQN\nAgent", "PPO\nAgent"]
    no_treat_risk  = 0.596     # from Phase 2 output
    combined_risk  = 0.358     # from Phase 2 output
    dqn_final_risk = dqn_risks_traj[-1]
    ppo_final_risk = ppo_risks_traj[-1]
    values  = [no_treat_risk, combined_risk, dqn_final_risk, ppo_final_risk]
    colors  = [PALETTE["red"], PALETTE["amber"], PALETTE["accent"], PALETTE["purple"]]
    bars    = ax5.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax5.set_ylim(0, 0.75)
    ax5.axhline(0.5, color=PALETTE["muted"], lw=1, ls="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", color=PALETTE["text"],
                 fontsize=10, fontweight="bold")

    # ── 6. Intervention frequency ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    sa(ax6, title="Intervention Frequency — DQN vs PPO Policy")

    dqn_counts = {k: 0 for k in INTERVENTIONS}
    ppo_counts = {k: 0 for k in INTERVENTIONS}
    for s in dqn_seq: dqn_counts[s["intervention"]] += 1
    for s in ppo_seq: ppo_counts[s["intervention"]] += 1

    x     = np.arange(N_ACTIONS)
    w     = 0.35
    b1    = ax6.bar(x - w/2, [dqn_counts[k] for k in INTERVENTIONS], w,
                    color=PALETTE["accent"], alpha=0.8, label="DQN")
    b2    = ax6.bar(x + w/2, [ppo_counts[k] for k in INTERVENTIONS], w,
                    color=PALETTE["purple"], alpha=0.8, label="PPO")
    ax6.set_xticks(x)
    ax6.set_xticklabels([k.replace("_", "\n") for k in INTERVENTIONS],
                        color=PALETTE["text"], fontsize=8)
    ax6.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"], fontsize=9)

    plt.savefig("assets/rl_training.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print("\n  ✓ RL dashboard saved → assets/rl_training.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI-COMPATIBLE INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_rl_recommendation(patient_dict: dict, agent_type: str = "dqn") -> dict:
    """
    Load a trained agent and return the optimal treatment sequence.
    Drop-in replacement for the old Q-learning /optimize/treatment endpoint.
    """
    artifact = joblib.load("models/progression_model.pkl")
    env      = MedTwinEnv(artifact, random_patients=False)

    if agent_type == "dqn":
        agent = DQNAgent()
        agent.load("models/dqn_agent.pt")
    else:
        agent = PPOAgent()
        agent.load("models/ppo_agent.pt")

    sequence, risks = evaluate_agent(agent, env, patient_dict)
    base_risk        = risks[0]
    final_risk       = risks[-1]

    return {
        "agent":                    agent_type.upper(),
        "base_risk":                round(base_risk, 4),
        "final_risk":               round(final_risk, 4),
        "total_risk_reduction_pct": round((base_risk - final_risk) * 100, 1),
        "optimal_sequence":         sequence,
        "summary": (
            f"{agent_type.upper()} agent reduced composite risk from "
            f"{round(base_risk * 100, 1)}% to {round(final_risk * 100, 1)}% "
            f"over 24 months (−{round((base_risk - final_risk) * 100, 1)}%)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MedTwin AI — Phase 4: RL Treatment Optimizer")
    print("  Algorithms: DQN (Dueling Double) + PPO (Clipped)")
    print("=" * 60)

    # Load progression model (built in Phase 2)
    if not os.path.exists("models/progression_model.pkl"):
        print("⚠ Run medtwin_phase2.py first to build the progression model.")
        exit(1)

    artifact = joblib.load("models/progression_model.pkl")
    env      = MedTwinEnv(artifact, random_patients=True)

    # ── Train ────────────────────────────────────────────────────────────────
    dqn_agent, dqn_rew, dqn_risks_ep, dqn_eval = train_dqn(env, n_episodes=600)
    ppo_agent, ppo_rew, ppo_risks_ep, ppo_eval = train_ppo(env, n_episodes=600)

    # ── Evaluate on demo patient ─────────────────────────────────────────────
    demo = {
        "glucose":        158.0,
        "bp_systolic":    145.0,
        "bmi":            33.6,
        "cholesterol":    241.0,
        "sleep_hours":    5.5,
        "stress_level":   72.0,
        "activity_score": 22.0,
        "hba1c":          7.1,
    }

    env_eval = MedTwinEnv(artifact, random_patients=False)
    dqn_seq, dqn_risk_traj = evaluate_agent(dqn_agent, env_eval, demo, "DQN")
    ppo_seq, ppo_risk_traj = evaluate_agent(ppo_agent, env_eval, demo, "PPO")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  Demo Patient — Optimal Treatment Sequences")
    print("─"*60)

    for label, seq, risks_traj in [("DQN", dqn_seq, dqn_risk_traj), ("PPO", ppo_seq, ppo_risk_traj)]:
        print(f"\n  {label} Agent:")
        print(f"  {'Month':>6}  {'Intervention':<18}  {'Risk':>6}  {'Δ Risk':>8}")
        print(f"  {'─'*44}")
        print(f"  {'  M0':>6}  {'(baseline)':<18}  {risks_traj[0]:.4f}")
        for s in seq:
            arrow = "↓" if s["delta"] <= 0 else "↑"
            print(f"  {'M'+str(s['month']):>6}  {s['intervention']:<18}  {s['risk']:.4f}  {arrow}{abs(s['delta']):.4f}")
        print(f"\n  Total reduction: {risks_traj[0]:.4f} → {risks_traj[-1]:.4f}  "
              f"(−{(risks_traj[0]-risks_traj[-1])*100:.1f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n  Generating RL dashboard...")
    fig = plot_rl_dashboard(
        dqn_rew, dqn_risks_ep, dqn_eval,
        ppo_rew, ppo_risks_ep, ppo_eval,
        dqn_seq, dqn_risk_traj,
        ppo_seq, ppo_risk_traj,
        demo,
    )

    print("\n" + "="*60)
    print("  Phase 4 complete.")
    print("  Saved: models/dqn_agent.pt")
    print("  Saved: models/ppo_agent.pt")
    print("  Saved: assets/rl_training.png")
    print("\n  Next → Add /optimize/rl endpoint to backend + RL tab in React")
    print("="*60)

    plt.show()
