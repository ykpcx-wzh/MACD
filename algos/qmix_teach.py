from copy import deepcopy

import numpy as np
import torch

from model.qmixer import QMixerModel
from model.rnn_model import RNNModel, RNNSelfAttnModel, RNNDreamerModel
from model.vq_vae import LinearVQVAE
from model.myhash import MyHash, DreamerHash

from loguru import logger


class QMIXTeach:
    def __init__(
            self,
            agent_model_student: RNNModel,
            agent_model_teacher: RNNSelfAttnModel,
            qmixer_model: QMixerModel,
            qmixer_model_student: QMixerModel,
            dreamer_count_model: RNNDreamerModel,
            vq_vae_model: LinearVQVAE,
            double_q: bool = True,
            gamma: float = 0.99,
            lr: float = 0.0005,
            clip_grad_norm: None | float = None,
            _config: dict = None,
    ) -> None:
        """QMIX algorithm
        Args:
            agent_model_student (nn.Module): agents' local q network for decision-making.
            agent_model_teacher (nn.Module): agents' local q network for decision-making.
            qmixer_model (nn.Module): A mixing network which takes local q values as input
                to construct a global Q network.
            double_q (bool): Double-DQN.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
            clip_grad_norm (None, or float): clipped value of gradients' global norm.
        """
        # checks
        self.student_hidden_states = None
        self.target_student_hidden_states = None
        self.teacher_hidden_states = None
        self.target_teacher_hidden_states = None
        self.teacher_hidden_states_rnd = None
        self.dreamer_hidden_states = None
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # =====================================
        # 初始化学生网络和学生Qmix网络
        # =====================================
        # 学生RNN网络
        self.agent_model_student = agent_model_student.to(self.device)
        self.target_agent_model_student = deepcopy(agent_model_student).to(self.device)
        # 学生QMIX网络
        self.qmixer_model_student = qmixer_model_student.to(self.device)
        self.target_qmixer_model_student = deepcopy(self.qmixer_model_student).to(
            self.device
        )

        # =============================================
        # 初始化教师教师AttentionRNN网络网络和教师Qmix网络
        # =============================================
        # 教师AttentionRNN网络
        self.agent_model_teacher = agent_model_teacher.to(self.device)
        self.target_agent_model_teacher = deepcopy(self.agent_model_teacher).to(
            self.device
        )

        # 教师RND网络
        self.agent_model_teacher_rnd = deepcopy(self.agent_model_teacher).to(
            self.device
        )
        self.agent_model_teacher_rnd.init_weight()
        # 教师QMIX网络
        self.qmixer_model = qmixer_model.to(self.device)
        self.target_qmixer_model = deepcopy(self.qmixer_model).to(self.device)

        # VQ-vae 网络
        self.vq_vae = vq_vae_model.to(self.device)
        self.myhash = MyHash(_config["vq_vae_num_embedding"])

        # Dreamer Count 网络
        self.dreamer_model = dreamer_count_model.to(self.device)
        self.dreamer_hash = DreamerHash(n_categorical=_config["n_categorical"], n_classes=_config["n_classes"])

        self.n_agents = _config["n_agents"]
        self.double_q = double_q
        self.gamma = gamma
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        self.tau = 0.01

        # ========================================
        # 初始化优化器，为不同优化器配置对应的参数
        # ========================================

        # 用于训练的教师网络，更新教师AttentionRNN和QMIX
        self.params_tea = list(self.agent_model_teacher.parameters()) + list(
            self.qmixer_model.parameters()
        )
        self.optimizer = torch.optim.RMSprop(
            params=self.params_tea, lr=self.lr, eps=0.00001
        )

        # 用于蒸馏的学生网络，更新学生的RNN
        self.params_kd = list(self.agent_model_student.parameters())
        self.optimizer_kd = torch.optim.RMSprop(
            params=self.params_kd, lr=self.lr, eps=0.00001
        )

        # 用于训练的学生网络，更新学生的RNN和QMIX
        self.params_student = list(self.agent_model_student.parameters()) + list(
            self.qmixer_model_student.parameters()
        )
        self.optimizer_student = torch.optim.RMSprop(
            params=self.params_student, lr=self.lr, eps=0.00001
        )

        # 蒸馏RND网络
        self.params_kd_rnd = list(self.agent_model_teacher_rnd.parameters())
        self.optimizer_kd_rnd = torch.optim.Adam(
            params=self.params_kd_rnd, lr=self.lr, eps=0.00001
        )

        # VQ-VAE
        self.params_vq_vae = list(self.vq_vae.parameters())
        self.optimizer_vq_vae = torch.optim.Adam(
            params=self.params_vq_vae, lr=self.lr, eps=0.00001
        )

        # dreamer
        self.params_dreamer = list(self.dreamer_model.parameters())
        self.optimizer_dreamer = torch.optim.Adam(
            params=self.params_dreamer, lr=self.lr, eps=0.00001
        )

    def train(self):
        self.agent_model_student.train()
        self.agent_model_teacher.train()
        self.qmixer_model.train()
        self.qmixer_model_student.train()
        self.agent_model_teacher_rnd.train()
        self.vq_vae.train()
        self.dreamer_model.train()

    def eval(self):
        self.agent_model_student.eval()
        self.agent_model_teacher.eval()
        self.qmixer_model.eval()
        self.qmixer_model_student.eval()
        self.agent_model_teacher_rnd.eval()
        self.vq_vae.eval()
        self.dreamer_model.eval()

    def init_hidden_states(self, batch_size: int) -> None:
        self.teacher_hidden_states = (
            self.agent_model_teacher.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .to(self.device)
        )
        self.target_teacher_hidden_states = (
            self.target_agent_model_teacher.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .to(self.device)
        )
        self.student_hidden_states = (
            self.agent_model_student.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .to(self.device)
        )
        self.target_student_hidden_states = (
            self.target_agent_model_student.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .to(self.device)
        )
        self.teacher_hidden_states_rnd = (
            self.agent_model_teacher_rnd.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .to(self.device)
        )

        self.dreamer_hidden_states = (
            self.dreamer_model.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, 1, -1)
            .to(self.device)
        )

    def predict_local_q(
            self, obs: torch.Tensor, hidden_state: torch.Tensor, agent_type: str = "student"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs (torch.Tensor): (n_agents, obs_shape)
            hidden_state (torch.Tensor): (n_agents, rnn_hidden_dim)
            agent_type (str): 'student' or 'teacher' or 'rnd'
        Returns:
            self.agent_model(obs, hidden_state)
        """
        obs = obs.unsqueeze(0)
        if agent_type == "student":
            return self.agent_model_student(obs, hidden_state)
        elif agent_type == "teacher":
            return self.agent_model_teacher(obs, hidden_state)
        elif agent_type == "rnd":
            return self.agent_model_teacher_rnd(obs, hidden_state)
        else:
            raise ValueError

    def sync_target(self) -> None:

        # for target_param, ori_param in zip(
        #         self.target_agent_model_teacher.parameters(),
        #         self.agent_model_teacher.parameters(),
        # ):
        #     target_param.data.copy_(
        #         self.tau * ori_param + (1 - self.tau) * target_param
        #     )
        #
        # for target_param, ori_param in zip(
        #         self.target_qmixer_model.parameters(), self.qmixer_model.parameters()
        # ):
        #     target_param.data.copy_(
        #         self.tau * ori_param + (1 - self.tau) * target_param
        #     )

        self.target_agent_model_teacher.load_state_dict(self.agent_model_teacher.state_dict())
        self.target_qmixer_model.load_state_dict(self.qmixer_model.state_dict())
        logger.info("Update teacher target network!!!!")

    def sync_target_for_student(self) -> None:
        # for target_param, ori_param in zip(
        #         self.target_qmixer_model_student.parameters(),
        #         self.qmixer_model_student.parameters(),
        # ):
        #     target_param.data.copy_(
        #         self.tau * ori_param + (1 - self.tau) * target_param
        #     )
        # for target_param, ori_param in zip(
        #         self.target_agent_model_student.parameters(),
        #         self.agent_model_student.parameters(),
        # ):
        #     target_param.data.copy_(
        #         self.tau * ori_param + (1 - self.tau) * target_param
        #     )
        self.target_agent_model_student.load_state_dict(self.agent_model_student.state_dict())
        self.target_qmixer_model_student.load_state_dict(self.qmixer_model_student.state_dict())
        logger.info("Update student target network!!!!")

    def student_learn(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.student_hidden_states = self.agent_model_student(
                obs, self.student_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_student_hidden_states = (
                self.target_agent_model_student(obs, self.target_student_hidden_states)
            )
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch
        ).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(dim=3)[
                0
            ]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model_student(
            chosen_action_local_qs, state_batch[:, :-1, :]
        )

        target_global_max_qs = self.target_qmixer_model_student(
            target_local_max_qs, state_batch[:, 1:, :]
        )

        target = reward_batch + self.gamma * mask * target_global_max_qs

        # ===================================
        # 根据TD误差更新学生的RNN和QMIX
        # ====================================
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise-TD
        self.optimizer_student.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_student, self.clip_grad_norm)
        self.optimizer_student.step()

        return loss.item(), mean_td_error.item()

    def student_learn_with_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            factor: torch.Tensor,
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.student_hidden_states = self.agent_model_student.forward(
                obs, self.student_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_student_hidden_states = (
                self.target_agent_model_student.forward(
                    obs, self.target_student_hidden_states
                )
            )
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch
        ).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(dim=3)[
                0
            ]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model_student.forward(
            chosen_action_local_qs, state_batch[:, :-1, :]
        )

        target_global_max_qs = self.target_qmixer_model_student.forward(
            target_local_max_qs, state_batch[:, 1:, :]
        )

        target = reward_batch + self.gamma * mask * target_global_max_qs

        # ===================================
        # 根据TD误差更新学生的RNN和QMIX
        # ====================================
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask

        chosen_action_factor = torch.gather(factor, dim=3, index=actions_batch).squeeze(
            -1
        )  # [B x T - 1 x N]
        avg_factor = chosen_action_factor.mean(dim=-1)  # [B x T]

        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error ** 2 * avg_factor.unsqueeze(-1)).sum() / mask.sum()

        # Optimise-TD
        self.optimizer_student.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_student, self.clip_grad_norm)
        self.optimizer_student.step()

        return loss.item(), mean_td_error.item()

    def student_learn_with_vq_vae_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            factor: torch.Tensor,
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.student_hidden_states = self.agent_model_student.forward(
                obs, self.student_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_student_hidden_states = (
                self.target_agent_model_student.forward(
                    obs, self.target_student_hidden_states
                )
            )
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch
        ).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(dim=3)[
                0
            ]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model_student.forward(
            chosen_action_local_qs, state_batch[:, :-1, :]
        )

        target_global_max_qs = self.target_qmixer_model_student.forward(
            target_local_max_qs, state_batch[:, 1:, :]
        )

        target = reward_batch + self.gamma * mask * target_global_max_qs

        # ===================================
        # 根据TD误差更新学生的RNN和QMIX
        # ====================================
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask

        # chosen_action_factor = torch.gather(factor, dim=3, index=actions_batch).squeeze(
        #     -1
        # )  # [B x T - 1 x N]
        # avg_factor = chosen_action_factor.mean(dim=-1)  # [B x T]

        factor = factor[:, :-1]

        mean_td_error = masked_td_error.sum() / mask.sum()
        # loss = (masked_td_error ** 2 * factor.unsqueeze(-1)).sum() / mask.sum()
        loss = (masked_td_error ** 2 * factor.unsqueeze(-1)).mean()

        # Optimise-TD
        self.optimizer_student.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_student, self.clip_grad_norm)
        self.optimizer_student.step()

        return loss.item(), mean_td_error.item()

    def learn_only_teacher(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Args:
            state_batch (torch.Tensor):             (batch_size, episode_length, state_shape)
            actions_batch (torch.Tensor):           (batch_size, episode_length, n_agents)
            reward_batch (torch.Tensor):            (batch_size, episode_length, 1)
            terminated_batch (torch.Tensor):        (batch_size, episode_length, 1)
            obs_batch (torch.Tensor):               (batch_size, episode_length, n_agents, obs_shape)
            available_actions_batch (torch.Tensor): (batch_size, episode_length, n_agents, n_actions)
            filled_batch (torch.Tensor):            (batch_size, episode_length, 1)
        Returns:
            loss (float): train loss
            td_error (float): train TD error
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            local_q, self.teacher_hidden_states = self.agent_model_teacher(obs, self.teacher_hidden_states)
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_teacher_hidden_states = (
                self.target_agent_model_teacher(obs, self.target_teacher_hidden_states))
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(target_local_qs, 3, cur_max_actions).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(dim=3)[0]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model(chosen_action_local_qs, state_batch[:, :-1, :])

        target_global_max_qs = self.target_qmixer_model(target_local_max_qs, state_batch[:, 1:, :])

        target = reward_batch + self.gamma * (1 - terminated_batch) * target_global_max_qs

        # ===================================
        # 根据TD误差更新教师的AttentionRNN和QMIX
        # ====================================
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise-TD
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_tea, self.clip_grad_norm)
        self.optimizer.step()

        return loss.item(), mean_td_error.item()

    def kd_student(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_student_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_student_q, self.student_hidden_states = self.agent_model_student(
                obs, self.student_hidden_states
            )
            agent_student_q = agent_student_q.reshape(batch_size, self.n_agents, -1)
            agent_student_qs.append(agent_student_q)

        agent_student_qs = torch.stack(agent_student_qs, dim=1)

        # Knowledge Diss error
        kd_error = (agent_student_qs - local_qs.clone().detach())[:, :-1]
        expanded_mask_kd = mask.unsqueeze(-1).expand_as(kd_error)
        masked_kd_error = kd_error * expanded_mask_kd

        mean_kd_error = masked_kd_error.sum() / expanded_mask_kd.sum()

        loss_kd = (masked_kd_error ** 2).sum() / expanded_mask_kd.sum()

        # optimise knowledge diss
        self.optimizer_kd.zero_grad()
        loss_kd.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd, self.clip_grad_norm)
        self.optimizer_kd.step()

        return loss_kd.item(), mean_kd_error.item()

    def kd_student_with_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            discount_factor: torch.Tensor,  # [B x T - 1 x N x A]
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            local_q, self.teacher_hidden_states = self.agent_model_teacher.forward(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_student_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_student_q, self.student_hidden_states = (
                self.agent_model_student.forward(obs, self.student_hidden_states)
            )
            agent_student_q = agent_student_q.reshape(batch_size, self.n_agents, -1)
            agent_student_qs.append(agent_student_q)

        agent_student_qs = torch.stack(agent_student_qs, dim=1)

        # 知识蒸馏的误差
        kd_error = (agent_student_qs - local_qs.detach())[:, :-1]
        # factored_kd_error = kd_error * discount_factor

        mask_kd = mask.unsqueeze(-1).expand(*kd_error.shape)
        masked_kd_error = kd_error * mask_kd

        mean_kd_error = masked_kd_error.sum() / mask_kd.sum()

        discount_factor = discount_factor.mean(dim=2)
        masked_kd_error = masked_kd_error.mean(dim=2)

        loss_kd = (masked_kd_error ** 2 * discount_factor).sum() / mask_kd.mean(
            dim=2
        ).sum()

        # optimise knowledge diss
        self.optimizer_kd.zero_grad()
        loss_kd.backward()
        # means = [param.cpu().detach().mean().item() for param in self.agent_model_student.parameters()]
        # maxes = [param.cpu().detach().max().item() for param in self.agent_model_student.parameters()]
        # print(f'KD STU mean grad is: {sum(means) / len(means)} | {max(means)} | {min(means)}, max grad is {sum(maxes) / len(maxes)} | {min(maxes)} | {max(maxes)}')
        # for name, param in self.agent_model_student.named_parameters():
        #     print(name, param.grad.max())
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd, self.clip_grad_norm)
        self.optimizer_kd.step()

        return loss_kd.item(), mean_kd_error.item()

    def kd_student_with_vq_vae_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            discount_factor: torch.Tensor,  # [B x T]
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            local_q, self.teacher_hidden_states = self.agent_model_teacher.forward(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_student_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_student_q, self.student_hidden_states = (
                self.agent_model_student.forward(obs, self.student_hidden_states)
            )
            agent_student_q = agent_student_q.reshape(batch_size, self.n_agents, -1)
            agent_student_qs.append(agent_student_q)

        agent_student_qs = torch.stack(agent_student_qs, dim=1)

        # 知识蒸馏的误差
        kd_error = (agent_student_qs - local_qs.detach())[:, :-1]
        # factored_kd_error = kd_error * discount_factor

        mask_kd = mask.unsqueeze(-1).expand(*kd_error.shape)
        masked_kd_error = kd_error * mask_kd

        mean_kd_error = masked_kd_error.sum() / mask_kd.sum()

        masked_kd_error = masked_kd_error.mean(dim=(2, 3))
        discount_factor = discount_factor[:, :-1]

        loss_kd = (masked_kd_error ** 2 * discount_factor).sum() / mask_kd.mean(
            dim=(2, 3)
        ).sum()

        # optimise knowledge diss
        self.optimizer_kd.zero_grad()
        loss_kd.backward()
        # means = [param.cpu().detach().mean().item() for param in self.agent_model_student.parameters()]
        # maxes = [param.cpu().detach().max().item() for param in self.agent_model_student.parameters()]
        # print(f'KD STU mean grad is: {sum(means) / len(means)} | {max(means)} | {min(means)}, max grad is {sum(maxes) / len(maxes)} | {min(maxes)} | {max(maxes)}')
        # for name, param in self.agent_model_student.named_parameters():
        #     print(name, param.grad.max())
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd, self.clip_grad_norm)
        self.optimizer_kd.step()

        return loss_kd.item(), mean_kd_error.item()

    def kd_student_with_dreamer_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            discount_factor: torch.Tensor,  # [B x T]
    ) -> tuple[float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)
        mask_factor = (discount_factor > 0).to(dtype=torch.uint8)[:, :-1]

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            local_q, self.teacher_hidden_states = self.agent_model_teacher.forward(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_student_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_student_q, self.student_hidden_states = (
                self.agent_model_student.forward(obs, self.student_hidden_states)
            )
            agent_student_q = agent_student_q.reshape(batch_size, self.n_agents, -1)
            agent_student_qs.append(agent_student_q)

        agent_student_qs = torch.stack(agent_student_qs, dim=1)

        # 知识蒸馏的误差
        kd_error = (agent_student_qs - local_qs.detach())[:, :-1]
        # factored_kd_error = kd_error * discount_factor

        mask_kd = (mask * mask_factor.unsqueeze(-1)).unsqueeze(-1).expand(*kd_error.shape)
        masked_kd_error = kd_error * mask_kd

        mean_kd_error = masked_kd_error.sum() / mask_kd.sum()

        masked_kd_error = torch.abs(masked_kd_error).mean(dim=(2, 3))
        discount_factor = discount_factor[:, :-1]

        # loss_kd = (masked_kd_error ** 2 * discount_factor).sum() / mask_kd.mean(dim=(2, 3)).sum()

        loss_kd = (masked_kd_error ** 2 * discount_factor).mean()

        # optimise knowledge diss
        self.optimizer_kd.zero_grad()
        loss_kd.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd, self.clip_grad_norm)
        self.optimizer_kd.step()

        return loss_kd.item(), mean_kd_error.item()

    def kd_rnd(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float]:
        """
        蒸馏RND网络
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_rnd_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_rnd_q, self.teacher_hidden_states_rnd = self.agent_model_teacher_rnd(
                obs, self.teacher_hidden_states_rnd
            )
            agent_rnd_q = agent_rnd_q.reshape(batch_size, self.n_agents, -1)
            agent_rnd_qs.append(agent_rnd_q)

        agent_rnd_qs = torch.stack(agent_rnd_qs, dim=1)

        # Knowledge Diss error
        kd_error = (agent_rnd_qs - local_qs.clone().detach())[:, :-1]
        mask_kd = mask.unsqueeze(-1).expand_as(kd_error)
        masked_kd_error = kd_error * mask_kd

        mean_kd_error = masked_kd_error.sum() / mask_kd.sum()

        loss_kd = (masked_kd_error ** 2).sum() / mask_kd.sum()

        # optimise knowledge diss
        self.optimizer_kd_rnd.zero_grad()
        loss_kd.backward()
        # means = [param.cpu().detach().mean().item() for param in self.agent_model_teacher_rnd.parameters()] maxes =
        # [param.cpu().detach().max().item() for param in self.agent_model_teacher_rnd.parameters()] print(f'KD RND
        # mean grad is: {sum(means) / len(means)} | {min(means)} | {max(means)}, max grad is {sum(maxes) / len(
        # maxes)} | {min(maxes)} | {max(maxes)}')
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd_rnd, self.clip_grad_norm)
        self.optimizer_kd_rnd.step()

        return loss_kd.item(), mean_kd_error.item()

    def vq_vae_learn(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float, float]:

        recons, _input, vq_loss = self.vq_vae.forward(state_batch)
        losses = self.vq_vae.loss_function(recons, _input, vq_loss)

        loss = losses["loss"]
        self.optimizer_vq_vae.zero_grad()
        loss.backward()
        self.optimizer_vq_vae.step()

        return (
            losses["Reconstruction_Loss"].detach().cpu().item(),
            losses["VQ_Loss"].detach().cpu().item(),
            loss.item(),
        )

    def dreamer_learn(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float]:

        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        dreamer_q_tots = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            local_q, self.teacher_hidden_states = self.agent_model_teacher.forward(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            dreamer_q_tot, self.dreamer_hidden_states = self.dreamer_model.forward(
                obs, self.dreamer_hidden_states
            )
            dreamer_q_tots.append(dreamer_q_tot.reshape(batch_size, -1))

        local_qs = torch.stack(local_qs, dim=1)
        dreamer_q_tots = torch.stack(dreamer_q_tots, dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch
        ).squeeze(3)

        chosen_action_global_qs = self.qmixer_model.forward(
            chosen_action_local_qs, state_batch[:, :-1, :]
        )

        # print(dreamer_q_tots.shape, chosen_action_global_qs.shape)
        # 计算蒸馏损失
        kd_error = dreamer_q_tots[:, :-1] - chosen_action_global_qs.detach()
        masked_kd_error = kd_error * mask

        loss = masked_kd_error.pow(2).sum() / mask.sum()

        self.optimizer_dreamer.zero_grad()
        loss.backward()
        self.optimizer_dreamer.step()
        return loss.detach().item()

    def get_dreamer_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            _type: str
    ) -> tuple[np.ndarray, float, float]:
        batch_size = state_batch.shape[0]
        self.init_hidden_states(batch_size)

        probs = self.dreamer_model.get_probs(obs_batch, self.dreamer_hidden_states)

        mask, indexes = self.dreamer_hash.get_index(probs, _type)
        hash_area = sum(np.array(self.dreamer_hash.hash) > 0) / len(self.dreamer_hash.hash)
        mask_area = mask.mean()

        return mask, hash_area, mask_area

    def get_dreamer_factor_with_teacher_baseline(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor
    ) -> tuple[np.ndarray, float, float]:
        batch_size = state_batch.shape[0]
        self.init_hidden_states(batch_size)

        probs = self.dreamer_model.get_probs(obs_batch, self.dreamer_hidden_states)

        mask = self.dreamer_hash.get_index_with_teacher_baseline(probs)
        hash_area = sum(np.array(self.dreamer_hash.hash) > 0) / len(self.dreamer_hash.hash)
        mask_area = mask.mean()

        return mask, hash_area, mask_area

    def kd_rnd_with_qENC(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float]:
        """
        蒸馏RND网络
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        tea_q_encs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states, q_encode_tea = (
                self.agent_model_teacher.forward(obs, self.teacher_hidden_states, True)
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            q_encode_tea = q_encode_tea.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)
            tea_q_encs.append(q_encode_tea)

        local_qs = torch.stack(local_qs, dim=1)
        tea_q_encs = torch.stack(tea_q_encs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_rnd_qs = []
        agent_rnd_q_encs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_rnd_q, self.teacher_hidden_states_rnd, q_encode_rnd = (
                self.agent_model_teacher_rnd.forward(
                    obs, self.teacher_hidden_states_rnd, True
                )
            )
            agent_rnd_q = agent_rnd_q.reshape(batch_size, self.n_agents, -1)
            q_encode_rnd = q_encode_rnd.reshape(batch_size, self.n_agents, -1)
            agent_rnd_qs.append(agent_rnd_q)
            agent_rnd_q_encs.append(q_encode_rnd)

        agent_rnd_qs = torch.stack(agent_rnd_qs, dim=1)
        agent_rnd_q_encs = torch.stack(agent_rnd_q_encs, dim=1)

        # ---------------  对于Q值的LOSS --------------------------- #
        kd_error = (agent_rnd_qs - local_qs.clone().detach())[:, :-1]
        mask_kd = mask.unsqueeze(-1).expand_as(kd_error)
        masked_kd_error = kd_error * mask_kd

        mean_kd_error = masked_kd_error.sum() / mask_kd.sum()

        loss_kd = (masked_kd_error ** 2).sum() / mask_kd.sum()

        # ----------------------- 对于Q Encoded 的LOSS --------------------------- #
        kd_error_q_enc = (agent_rnd_q_encs - tea_q_encs.detach())[:, :-1]
        mask_for_q_enc = mask.unsqueeze(-1).expand_as(kd_error_q_enc)
        masked_kd_error_q_enc = mask_for_q_enc * kd_error_q_enc

        mean_kd_q_enc_error = masked_kd_error_q_enc.sum() / mask_for_q_enc.sum()
        loss_q_enc = (masked_kd_error_q_enc ** 2).sum() / mask_for_q_enc.sum()

        loss_all = loss_kd + loss_q_enc

        # 优化
        self.optimizer_kd_rnd.zero_grad()
        loss_all.backward()
        # means = [param.cpu().detach().mean().item() for param in self.agent_model_teacher_rnd.parameters()]
        # maxes = [param.cpu().detach().max().item() for param in self.agent_model_teacher_rnd.parameters()]
        # print(f'KD RND mean grad is: {sum(means) / len(means)} | {min(means)} | {max(means)}, max grad is {sum(maxes) / len(maxes)} | {min(maxes)} | {max(maxes)}')
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd_rnd, self.clip_grad_norm)
        self.optimizer_kd_rnd.step()

        return loss_kd.detach().item(), loss_q_enc.detach().item()

    @torch.no_grad()
    def get_vq_vae_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[np.ndarray, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        indexes = self.vq_vae.get_codebook_index(state_batch)
        hash_count = self.myhash.count(indexes)

        big = sum(np.array(self.myhash.hash) > 0) / len(self.myhash.hash)
        # print(indexes.shape, hash_count.min(), hash_count.max(), hash_count.mean(), min(self.myhash.hash), max(self.myhash.hash), sum(self.myhash.hash) / len(self.myhash.hash), big)

        hash_factor = 1 / hash_count.reshape((batch_size, episode_len))

        return hash_factor, big

    @torch.no_grad()
    def get_count_factor(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> np.ndarray:
        """
        获取伪计数系数
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_rnd_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_rnd_q, self.teacher_hidden_states_rnd = self.agent_model_teacher_rnd(
                obs, self.teacher_hidden_states_rnd
            )
            agent_rnd_q = agent_rnd_q.reshape(batch_size, self.n_agents, -1)
            agent_rnd_qs.append(agent_rnd_q)

        agent_rnd_qs = torch.stack(agent_rnd_qs, dim=1)

        # Knowledge Diss error
        kd_error = (agent_rnd_qs - local_qs.clone().detach())[
                   :, :-1
                   ]  # [B x (T - 1) x N x A]
        masked_kd_error = kd_error * mask.unsqueeze(-1).expand(*kd_error.shape)

        # normed_factor = 3 * masked_kd_error ** 2  # [B x (T - 1) x N x A]
        # normed_factor = torch.clamp(normed_factor, 0.01, 3)
        normed_factor = torch.tanh(masked_kd_error ** 2 * 2)  # [B x (T - 1) x N x A]

        return normed_factor.cpu().numpy()

    @torch.no_grad()
    def get_rnd_count_factor_cosine_similarity(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> float:
        """
        通过余弦相似度计算两个Q矩阵的相似度，从而获取伪计数系数
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        agent_rnd_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_rnd_q, self.teacher_hidden_states_rnd = self.agent_model_teacher_rnd(
                obs, self.teacher_hidden_states_rnd
            )
            agent_rnd_q = agent_rnd_q.reshape(batch_size, self.n_agents, -1)
            agent_rnd_qs.append(agent_rnd_q)

        agent_rnd_qs = torch.stack(agent_rnd_qs, dim=1)

        local_qs = local_qs - torch.min(local_qs)
        agent_rnd_qs = agent_rnd_qs - torch.min(agent_rnd_qs)
        cos_sim = torch.cosine_similarity(agent_rnd_qs.flatten().unsqueeze(0), local_qs.flatten().unsqueeze(0))

        return cos_sim.cpu().numpy()

    def kd_stu_with_factor_train_stu(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
    ) -> tuple[float, float, float, np.ndarray, float]:

        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :, :]
        actions_batch = actions_batch[:, :, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :, :]
        filled_batch = filled_batch[:, :, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        # ===================================
        # 获取伪计数的系数
        # ==================================
        local_tea_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_tea_qs.append(local_q)

        local_tea_qs = torch.stack(local_tea_qs, dim=1)

        agent_rnd_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_rnd_q, self.teacher_hidden_states_rnd = self.agent_model_teacher_rnd(
                obs, self.teacher_hidden_states_rnd
            )
            agent_rnd_q = agent_rnd_q.reshape(batch_size, self.n_agents, -1)
            agent_rnd_qs.append(agent_rnd_q)

        agent_rnd_qs = torch.stack(agent_rnd_qs, dim=1)

        # 获取系数
        factor_kd_error = (
                agent_rnd_qs.clone().detach() - local_tea_qs.clone().detach()
        )  # [B x T x N x A]
        factor_masked_kd_error = (
            factor_kd_error  # * mask.unsqueeze(-1).expand(*factor_kd_error.shape)
        )
        normed_abs_factor_4d = torch.tanh(torch.abs(factor_masked_kd_error) * 1)
        # abs_kd_error = torch.abs(factor_masked_kd_error.sum(dim=2)) / factor_masked_kd_error.shape[2] # [B x T x A]
        # normed_count_factor = torch.tanh(abs_kd_error * 4)

        # 蒸馏训练RND
        kd_rnd_error = agent_rnd_qs - local_tea_qs.clone().detach()
        kd_rnd_error = (kd_rnd_error ** 2).sum(axis=(2, 3))
        mask_rnd_kd = mask.squeeze(-1)
        masked_rnd_kd_error = kd_rnd_error * mask_rnd_kd
        loss_rnd_kd = masked_rnd_kd_error.sum() / mask_rnd_kd.sum()

        # optimise knowledge diss
        self.optimizer_kd_rnd.zero_grad()
        loss_rnd_kd.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_kd_rnd, self.clip_grad_norm)
        self.optimizer_kd_rnd.step()

        # ===================================
        # 蒸馏学生网络的损失
        # ===================================
        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.teacher_hidden_states = self.agent_model_teacher(
                obs, self.teacher_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

        local_qs = torch.stack(local_qs, dim=1)

        agent_student_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            agent_student_q, self.student_hidden_states = self.agent_model_student(
                obs, self.student_hidden_states
            )
            agent_student_q = agent_student_q.reshape(batch_size, self.n_agents, -1)
            agent_student_qs.append(agent_student_q)

        agent_student_qs = torch.stack(agent_student_qs, dim=1)

        # Knowledge Diss error
        kd_stu_error = agent_student_qs - local_qs.clone().detach()
        kd_stu_error_3d = kd_stu_error.mean(dim=2)  # [B x T x A]
        loss_kd_stu = (kd_stu_error ** 2).mean()

        # ===================================
        # 学生网络和环境交互产生的损失
        # ===================================

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.student_hidden_states = self.agent_model_student(
                obs, self.student_hidden_states
            )
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_student_hidden_states = (
                self.target_agent_model_student(obs, self.target_student_hidden_states)
            )
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :, :, :], dim=3, index=actions_batch
        ).squeeze(
            3
        )  # [B x T x N]
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, :, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, :].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(dim=3)[
                0
            ]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model_student(
            chosen_action_local_qs, state_batch[:, :, :]
        )  # [B x T x N] -> [B x T x 1]

        target_global_max_qs = self.target_qmixer_model_student(
            target_local_max_qs, state_batch[:, :, :]
        )

        target = reward_batch + self.gamma * mask * target_global_max_qs

        # ===================================
        # 根据TD误差更新学生的RNN和QMIX
        # ====================================
        td_error = target.detach() - chosen_action_global_qs  # [B x T]
        masked_td_error = td_error * mask
        loss_td = (masked_td_error ** 2).mean()

        factor_for_td = torch.gather(
            normed_abs_factor_4d, dim=3, index=actions_batch
        ).squeeze(
            3
        )  # [B x T x N]
        factor_for_td = factor_for_td.mean(dim=-1)  # [B x T]

        factor_for_kd = normed_abs_factor_4d.mean(dim=2)  # [B x T x A]

        kd_error_ = kd_stu_error_3d
        td_error_ = masked_td_error.squeeze(-1)
        factored_kd_loss = ((kd_error_ * factor_for_kd) ** 2).sum() / (
                kd_error_.nbytes / kd_error_.element_size()
        )
        factored_td_loss = ((td_error_ * (1 - factor_for_td)) ** 2).sum() / mask.sum()
        # error_all = kd_error_ ** 2 * factor_for_td + td_error_ ** 2 * (1 - factor_for_td)
        loss_all = factored_kd_loss + factored_td_loss
        # Optimise-TD
        self.optimizer_student.zero_grad()
        loss_all.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_student, self.clip_grad_norm)
        self.optimizer_student.step()

        return (
            loss_kd_stu.item(),
            loss_td.item(),
            loss_all.item(),
            normed_abs_factor_4d.cpu().numpy(),
            loss_rnd_kd.item(),
        )

    def kd_student_with_mask_td_stu_with_mask(
            self,
            state_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            reward_batch: torch.Tensor,
            terminated_batch: torch.Tensor,
            obs_batch: torch.Tensor,
            available_actions_batch: torch.Tensor,
            filled_batch: torch.Tensor,
            mask_dreamer: torch.Tensor,  # [B x T]
    ) -> tuple[float, float, float]:
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self.init_hidden_states(batch_size)

        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        mask_dreamer = mask_dreamer.unsqueeze(-1)[:, :-1]
        reward_batch = reward_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch) * mask_dreamer
        mask_for_td = (1 - filled_batch) * (1 - terminated_batch) * (1 - mask_dreamer)

        # 教师网络得到的Q值
        tea_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            tea_local_q, self.teacher_hidden_states = self.agent_model_teacher.forward(
                obs, self.teacher_hidden_states
            )
            tea_local_q = tea_local_q.reshape(batch_size, self.n_agents, -1)
            tea_local_qs.append(tea_local_q)

        tea_local_qs = torch.stack(tea_local_qs, dim=1)

        # ===================================
        # 根据蒸馏误差更新学生的RNN
        # ====================================
        stu_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            stu_local_q, self.student_hidden_states = (
                self.agent_model_student.forward(obs, self.student_hidden_states)
            )
            stu_local_q = stu_local_q.reshape(batch_size, self.n_agents, -1)
            stu_local_qs.append(stu_local_q)

        stu_local_qs = torch.stack(stu_local_qs, dim=1)

        # 知识蒸馏的误差
        kd_error = (stu_local_qs - tea_local_qs.detach())[:, :-1]
        mean_kd_error = (kd_error ** 2).mean(dim=(2, 3))  # [BxTxNxA]->[BxT]
        masked_kd_error = mean_kd_error * mask.squeeze(dim=-1)

        target_stu_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])

            target_stu_local_q, self.target_student_hidden_states = (
                self.target_agent_model_student.forward(
                    obs, self.target_student_hidden_states
                )
            )
            target_stu_local_q = target_stu_local_q.view(batch_size, self.n_agents, -1)
            target_stu_local_qs.append(target_stu_local_q)

        target_stu_local_qs = torch.stack(target_stu_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            stu_local_qs[:, :-1, :, :], dim=3, index=actions_batch
        ).squeeze(3)
        # mask unavailable actions
        target_stu_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = stu_local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_stu_local_qs, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_local_max_qs = target_stu_local_qs.max(dim=3)[
                0
            ]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model_student.forward(
            chosen_action_local_qs, state_batch[:, :-1, :]
        )

        target_global_max_qs = self.target_qmixer_model_student.forward(
            target_local_max_qs, state_batch[:, 1:, :]
        )

        target = reward_batch + self.gamma * target_global_max_qs

        # ===================================
        # 根据TD误差更新学生的RNN和QMIX
        # ====================================
        td_error = (target.detach() - chosen_action_global_qs) * mask_for_td
        masked_td_error = td_error ** 2
        masked_td_error = masked_td_error.squeeze(-1)

        all_error = masked_kd_error + masked_td_error
        loss = all_error.mean()
        # Optimise-TD
        self.optimizer_student.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params_student, self.clip_grad_norm)
        self.optimizer_student.step()

        return masked_kd_error.mean().detach().cpu().numpy(), masked_td_error.mean().detach().cpu().numpy(), loss.item()
