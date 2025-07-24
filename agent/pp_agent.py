import argparse
import gymnasium.spaces
import torch
import numpy as np

from algos.mlp_qmix import MlpQmixAlgo


class PPAgent:
    def __init__(
        self,
        algo: MlpQmixAlgo,
        agent_action_space: gymnasium.spaces.Discrete,
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.target_update_count = 0
        self.device = device
        self.alg = algo
        self.agent_action_space = agent_action_space
        self.args = args
        self.exploration_start: float = args.exploration_start
        self.exploration_end: float = args.exploration_end
        self.exploration_decay: float = args.exploration_decay
        self.update_target_interval: int = args.update_target_interval

        self.exploration: float = self.exploration_start

        # 同步网络
        self.global_step = 0

    # 训练时，根据贪心策略选择动作
    def sample(self, obs: np.ndarray) -> np.ndarray:
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs)
        else:
            actions = []
            for _ in range(self.args.n_agent):
                actions.append(self.agent_action_space.sample())
            actions = np.array(actions)

        self.exploration = max(
            self.exploration_end, self.exploration - self.exploration_decay
        )

        return actions

    # 测试时，完全按照网络输出选择动作
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        agents_q = self.alg.predict_local_q(obs)
        actions = torch.argmax(agents_q, dim=-1).detach().cpu().numpy()
        return actions

    def train(self):
        """
        set agent to train mode
        """
        self.alg.train()

    def eval(self):
        """
        set agent to eval mode
        """
        self.alg.eval()

    def learn(
        self,
        batch_state: np.ndarray,
        batch_obs: np.ndarray,
        batch_rewards: np.ndarray,
        batch_actions: np.ndarray,
        batch_terminated: np.ndarray,
        batch_next_obs: np.ndarray,
    ) -> tuple[float, float]:
        """

        Args:
            batch_state (np.ndarray):      [batch_size, n_agent, state_length]
            batch_obs (np.ndarray):        [batch_size, n_agent, obs_length]
            batch_rewards: (np.ndarray)    [batch_size, n_agent]
            batch_actions (np.ndarray):    [batch_size, n_agent]
            batch_terminated (np.ndarray): [batch_size, n_agent]
            batch_next_obs (np.ndarray):   [batch_size, n_agent, obs_length]

        Returns:

        """
        if self.global_step % self.update_target_interval == 0:
            self.alg.hard_sync_target()
            self.target_update_count = 0

        self.global_step += 1
        state_batch_tensor = torch.tensor(
            batch_state, dtype=torch.float32, device=self.device
        )
        obs_batch_tensor = torch.tensor(
            batch_obs, dtype=torch.float32, device=self.device
        )
        reward_batch_tensor = torch.tensor(
            batch_rewards, dtype=torch.float32, device=self.device
        )
        action_batch_tensor = torch.tensor(
            batch_actions, dtype=torch.int64, device=self.device
        )
        next_ob_batch_tensor = torch.tensor(
            batch_next_obs, dtype=torch.float32, device=self.device
        )
        terminated_batch_tensor = torch.tensor(
            batch_terminated, dtype=torch.float32, device=self.device
        )

        loss, mean_td_error = self.alg.learn(
            state_batch_tensor,
            obs_batch_tensor,
            action_batch_tensor,
            reward_batch_tensor,
            terminated_batch_tensor,
            next_ob_batch_tensor,
        )

        return loss, mean_td_error
