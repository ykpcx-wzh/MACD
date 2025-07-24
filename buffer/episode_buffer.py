from typing import Tuple, Any

import numpy as np
import random
from collections import deque

from numpy import ndarray, dtype, floating


class EpisodeExperience:
    def __init__(self, episode_length: int) -> None:
        self.max_len = episode_length

        self.episode_state = []
        self.episode_actions = []
        self.episode_reward = []
        self.episode_terminated = []
        self.episode_obs = []
        self.episode_available_actions = []
        self.episode_filled = []

    @property
    def count(self):
        return len(self.episode_state)

    def add(self, state, actions, reward, terminated, obs, available_actions, filled):
        assert self.count < self.max_len
        self.episode_state.append(state)
        self.episode_actions.append(actions)
        self.episode_reward.append(reward)
        self.episode_terminated.append(terminated)
        self.episode_obs.append(obs)
        self.episode_available_actions.append(available_actions)
        self.episode_filled.append(filled)

    def get_data(
        self,
    ) -> tuple[
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
    ]:
        assert self.count == self.max_len
        return (
            np.array(self.episode_state),
            np.array(self.episode_actions),
            np.array(self.episode_reward),
            np.array(self.episode_terminated),
            np.array(self.episode_obs),
            np.array(self.episode_available_actions),
            np.array(self.episode_filled),
        )


class EpisodeReplayBuffer(object):
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)

    def add(self, episode_experience):
        self.buffer.append(episode_experience)

    @property
    def count(self):
        return len(self.buffer)

    def sample_batch(self, batch_size: int):

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        (
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        ) = ([], [], [], [], [], [], [])
        for episode in batch:
            s, a, r, t, obs, available_actions, filled = episode.get_data()
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            t_batch.append(t)
            obs_batch.append(obs)
            available_actions_batch.append(available_actions)
            filled_batch.append(filled)

        s_batch = np.array(s_batch)
        filled_batch = np.array(filled_batch)
        r_batch = np.array(r_batch)
        t_batch = np.array(t_batch)
        a_batch = np.array(a_batch).astype(np.int64)
        obs_batch = np.array(obs_batch)
        available_actions_batch = np.array(available_actions_batch).astype(np.int64)

        return (
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )
