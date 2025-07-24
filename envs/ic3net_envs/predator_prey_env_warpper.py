from argparse import Namespace
import numpy as np
from .predator_prey_env import PredatorPreyEnv


class PredatorPreyEnvWarpper(PredatorPreyEnv):
    def __init__(self):
        super(PredatorPreyEnvWarpper, self).__init__()

    def step(
        self, action: np.ndarray | list
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
        obs, reward, done, info = super().step(action)
        obs = np.argmax(obs, axis=-1).astype(float)
        obs /= self.vocab_size  # normalize
        state = obs.flatten()
        obs = obs.reshape(obs.shape[0], -1)
        self.episode_step += 1
        if isinstance(reward, np.ndarray):
            reward = reward.sum()
        if self.episode_step == self.episode_limit:
            done = True
        return state, obs, reward, done, done

    def get_available_actions(self) -> np.ndarray:
        return np.ones(shape=(self.npredator, self.naction), dtype=int)

    def multi_agent_init(self, args: Namespace):
        super().multi_agent_init(args)
        self.episode_limit = args.episode_max_steps
        self.obs_shape = (2 * self.vision + 1) * (2 * self.vision + 1)
        self.state_shape = self.npredator * self.obs_shape
        self.n_agents = self.npredator
        self.n_actions = self.naction
        self.win_counted = False

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = super().reset()
        obs = np.argmax(obs, axis=-1).astype(float)
        obs /= self.vocab_size  # normalize
        state = obs.flatten()
        obs = obs.reshape(self.npredator, -1)
        self.episode_step = 0
        return state, obs

    def get_sight(self):
        return self.vision * 2 + 1
