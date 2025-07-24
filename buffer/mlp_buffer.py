import numpy as np


class MLPReplayBuffer:
    def __init__(
        self,
        max_size: int,
        state_dim: list[int],
        obs_dim: list[int],
        act_dim: list[int],
    ):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.obs = np.zeros(shape=[self.max_size] + obs_dim, dtype=np.float32)
        self.acts = np.zeros(shape=[self.max_size] + act_dim, dtype=np.float32)
        self.state = np.zeros(shape=[self.max_size] + state_dim, dtype=np.float32)

        self.rewards = np.zeros(shape=[self.max_size], dtype=np.float32)
        self.terminated = np.zeros(shape=[self.max_size], dtype=np.bool_)
        self.next_obs = np.zeros(shape=[self.max_size] + obs_dim, dtype=np.float32)

        self._current_size = 0
        self._current_idx = 0

    def sample_batch(self, batch_size: int):
        batch_idx = np.random.randint(self._current_size, size=batch_size)
        state = self.state[batch_idx]
        obs = self.obs[batch_idx]
        acts = self.acts[batch_idx]
        rewards = self.rewards[batch_idx]
        terminated = self.terminated[batch_idx]
        next_obs = self.next_obs[batch_idx]

        return state, obs, acts, rewards, next_obs, terminated

    def append(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        act: np.ndarray,
        reward: float | np.ndarray,
        next_obs: np.ndarray,
        terminated: bool,
    ):
        if self._current_size < self.max_size:
            self._current_size += 1

        if isinstance(reward, np.ndarray):
            reward = reward.sum()
        self.state[self._current_idx] = state
        self.obs[self._current_idx] = obs
        self.acts[self._current_idx] = act
        self.rewards[self._current_idx] = reward
        self.terminated[self._current_idx] = terminated
        self.next_obs[self._current_idx] = next_obs

        self._current_idx = (self._current_idx + 1) % self.max_size

    def size(self):
        return self._current_size

    def __len__(self):
        return self._current_size

    def save(self, path: str):
        other = np.array([self._current_size, self._current_idx], dtype=np.int32)
        np.savez(
            path,
            state=self.state,
            obs=self.obs,
            action=self.acts,
            reward=self.rewards,
            terminated=self.terminated,
            next_obs=self.next_obs,
            other=other,
        )

    def load(self, path: str):
        data = np.load(path)
        other = data["other"]
        if int(other[0]) > self.max_size:
            print("loading from a bigger size rpm!")

        self._current_size = min(int(other[0]), self.max_size)
        self._current_idx = min(int(other[1]), self.max_size - 1)

        self.state[: self._current_idx] = data["state"][: self._current_size]
        self.obs[: self._current_size] = data["obs"][: self._current_size]
        self.acts[: self._current_size] = data["action"][: self._current_size]
        self.rewards[: self._current_size] = data["reward"][: self._current_size]
        self.terminated[: self._current_size] = data["terminated"][: self._current_size]
        self.next_obs[: self._current_size] = data["next_obs"][: self._current_size]
        print("[load rpm]memory load from {}".format(path))
