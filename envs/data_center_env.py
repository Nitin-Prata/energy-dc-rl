import numpy as np
import gymnasium as gym
from gymnasium import spaces



class DataCenterEnv(gym.Env):
    """
    Custom Data Center Environment (Gymnasium version).
    Simulates energy, cost, carbon, and completion time for RL-based autoscaling.
    Features:
        - Dynamic electricity pricing
        - Renewable energy share
        - Carbon intensity tracking
        - Realistic reward shaping
    Compatible with Stable-Baselines3 and Gymnasium API.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, seed=None):
        super(DataCenterEnv, self).__init__()
        if seed is not None:
            np.random.seed(seed)

        # Example action & observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Internal variables
        self.state = None
        self.current_step = 0
        self.max_steps = 200
        self.render_mode = render_mode
        self.total_energy = 0.0
        self.total_completion_time = 0.0
        # New: dynamic pricing, renewables, carbon
        self.dynamic_pricing = np.random.uniform(0.08, 0.20, self.max_steps)  # $/kWh
        self.renewable_share = np.random.uniform(0.1, 0.7, self.max_steps)    # fraction
        self.carbon_intensity = np.random.uniform(0.2, 0.7, self.max_steps)   # kg CO2/kWh
        self.total_carbon = 0.0
        self.total_cost = 0.0

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
        Returns:
            state (np.ndarray): Initial observation.
            info (dict): Initial info dict.
        """
        super().reset(seed=seed)

        # Reset state variables
        self.state = np.random.rand(self.observation_space.shape[0])
        self.current_step = 0
        self.total_energy = 0.0
        self.total_completion_time = 0.0

        info = {"total_energy": self.total_energy, "total_completion_time": self.total_completion_time}
        return self.state, info

    def step(self, action):
        """
        Executes one step in the environment.
        Args:
            action (float or np.ndarray): Action taken by agent.
        Returns:
            state (np.ndarray): Next observation.
            reward (float): Reward for this step.
            terminated (bool): Whether episode ended.
            truncated (bool): Whether max steps reached.
            info (dict): Step metrics (energy, cost, carbon, etc).
        """
        self.current_step += 1

        # Simulate environment dynamics
        self.state = np.clip(self.state + np.random.randn(*self.state.shape) * 0.01, 0, 1)

        # Make energy and completion time highly sensitive to action
        # Best action is near 0.2 (low energy, fast completion)
        energy_used = 100 + 900 * float(np.abs(action - 0.2))**2  # Quadratic penalty, min at 0.2
        completion_time = 100 + 400 * float(np.abs(action - 0.2))**2  # Quadratic penalty, min at 0.2

        self.total_energy += energy_used
        self.total_completion_time += completion_time

        # New: dynamic pricing, renewables, carbon
        price = self.dynamic_pricing[self.current_step-1]
        renewable = self.renewable_share[self.current_step-1]
        carbon_intensity = self.carbon_intensity[self.current_step-1]
        cost = energy_used * price
        carbon = energy_used * carbon_intensity * (1 - renewable)
        self.total_cost += cost
        self.total_carbon += carbon

        # Sharper reward: strongly reward low energy and fast completion
        reward = -energy_used - 0.5 * completion_time

        # Termination condition
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "step_energy": energy_used,
            "step_cost": cost,
            "step_carbon": carbon,
            "total_energy": self.total_energy,
            "total_cost": self.total_cost,
            "total_carbon": self.total_carbon,
            "total_completion_time": self.total_completion_time
        }
        return self.state, reward, terminated, truncated, info

    def _baseline_energy_estimate(self):
        # Estimate baseline energy for this step (e.g., threshold or random policy)
        # This can be tuned for your baseline
        # For now, assume a fixed value or simple heuristic
        return 10.0  # TODO: Replace with actual baseline logic if available
    def render(self):
        """Optional visualization."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        """Clean up if needed."""
        pass
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        """Clean up if needed."""
        pass
