# agents/replay_buffer.py
"""
Simple replay buffer (uniform sampling).
Small, reliable, and easy to understand. Replace by
PrioritizedReplayBuffer later for better sample efficiency.
"""
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store numpy arrays (copy to avoid mutability issues)
        self.buffer.append((np.array(state, copy=True),
                            np.array(action, copy=True),
                            float(reward),
                            np.array(next_state, copy=True),
                            bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
