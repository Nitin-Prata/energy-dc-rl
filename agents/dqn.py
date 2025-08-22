def save(self, path: str):
    """Save agent weights to file."""
    torch.save({
        'q_network': self.q_network.state_dict(),
        'transition_model': self.transition_model.state_dict()
    }, path)

def load(self, path: str):
    """Load agent weights from file."""
    checkpoint = torch.load(path)
    self.q_network.load_state_dict(checkpoint['q_network'])
    self.transition_model.load_state_dict(checkpoint['transition_model'])

# agents/dqn_agent.py
"""
DQNAgent (Double DQN). Uses PyTorch. Handles Gymnasium & Gym obs/actions robustly.
Assumes discrete action_space (env.action_space.n). If action space is continuous,
you should switch to DDPG/SAC. For this project we use discrete: number of active servers.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from agents.replay_buffer import ReplayBuffer

# Simple MLP Q-network (dueling optional)
class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, n_actions:int, hidden=(256,256), dueling:bool=False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            # feature trunk
            layers = []
            in_dim = obs_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            self.feature = nn.Sequential(*layers)
            # dueling heads
            self.value_head = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128,1))
            self.adv_head = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, n_actions))
        else:
            layers = []
            in_dim = obs_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers += [nn.Linear(in_dim, n_actions)]
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.dueling:
            feat = self.feature(x)
            val = self.value_head(feat)
            adv = self.adv_head(feat)
            q = val + (adv - adv.mean(dim=1, keepdim=True))
            return q
        else:
            return self.model(x)

class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 200000,
        device: str = None,
        tau: float = 1.0,    # hard update frequency multiplier (if integer use step mod)
        target_update_freq: int = 1000,
        dueling: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.q_net = QNetwork(obs_dim, n_actions, dueling=dueling).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, dueling=dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_capacity)
        self.target_update_freq = target_update_freq
        self.update_count = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        # For continuous Box action space, output a numpy array in [0.0, 1.0]
        if np.random.rand() < epsilon:
            action = np.random.uniform(0.0, 1.0)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                qvals = self.q_net(state_v).cpu().numpy()[0]
                action = np.clip(qvals.mean(), 0.0, 1.0)
        return np.array([action], dtype=np.float32)

    def push(self, s,a,r,ns,d):
        self.replay.push(s,a,r,ns,d)

    def update(self) -> Tuple[float, int]:
        if len(self.replay) < self.batch_size:
            return None, 0
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s_v = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns_v = torch.tensor(ns, dtype=torch.float32).to(self.device)
        a_v = torch.tensor(a, dtype=torch.long).unsqueeze(1).to(self.device)
        r_v = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
        d_v = torch.tensor(d.astype(float), dtype=torch.float32).unsqueeze(1).to(self.device)

        # current Q
        q_values = self.q_net(s_v).gather(1, a_v)

        # Double DQN target: actions from q_net, values from target_net
        with torch.no_grad():
            next_q_values_online = self.q_net(ns_v)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(ns_v).gather(1, next_actions)
            target = r_v + self.gamma * next_q_target * (1 - d_v)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item()), 1
