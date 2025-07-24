import torch
import os
import numpy as np
from src.algos.qmix import QMIX
from src.utils import AvailableActionsSampler


class QMixAgent:
    def __init__(
        self,
        algorithm: QMIX,
        exploration_start: float,
        min_exploration: float,
        exploration_decay: float,
        update_target_interval: int,
    ) -> None:
        self.alg = algorithm
        self.global_step = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay

        self.target_update_count = 0
        self.update_target_interval = update_target_interval

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        agent_model_path = os.path.join(save_dir, "agent.pt")
        qmixer_model_path = os.path.join(save_dir, "qmixer.pt")

        torch.save(self.alg.agent_model.state_dict(), agent_model_path)
        torch.save(self.alg.qmixer_model.state_dict(), qmixer_model_path)

        print("Saved agent and qmixer model to {}".format(save_dir))

    def reset_agent(self, batch_size: int = 1) -> None:
        self.alg.init_hidden_states(batch_size)

    def sample(self, obs: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            actions = AvailableActionsSampler(available_actions).sample()

        self.exploration = max(
            self.min_exploration, self.exploration - self.exploration_decay
        )

        return actions

    def predict(self, obs: np.ndarray, available_actions: np.ndarray) -> np.ndarray:

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.int32, device=self.device
        )

        agents_q, self.alg.hidden_states = self.alg.predict_local_q(
            obs, self.alg.hidden_states
        )

        unavailable_actions_mask = (available_actions == 0).to(torch.float32)
        agents_q -= unavailable_actions_mask * 1e8
        actions = torch.argmax(agents_q, dim=-1).detach().cpu().numpy()
        return actions

    def learn(
        self,
        state_batch: np.ndarray,
        actions_batch: np.ndarray,
        reward_batch: np.ndarray,
        terminated_batch: np.ndarray,
        obs_batch: np.ndarray,
        available_actions_batch: np.ndarray,
        filled_batch: np.ndarray,
    ) -> tuple[float, float]:

        if self.global_step % self.update_target_interval == 0:
            self.alg.sync_target()
            self.target_update_count += 1

        self.global_step += 1

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(
            actions_batch, dtype=torch.int64, device=self.device
        )
        reward_batch = torch.tensor(
            reward_batch, dtype=torch.float32, device=self.device
        )
        terminated_batch = torch.tensor(
            terminated_batch, dtype=torch.float32, device=self.device
        )
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        available_actions_batch = torch.tensor(
            available_actions_batch, dtype=torch.int32, device=self.device
        )
        filled_batch = torch.tensor(
            filled_batch, dtype=torch.float32, device=self.device
        )

        mean_loss, mean_td_error = self.alg.learn(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return mean_loss, mean_td_error
