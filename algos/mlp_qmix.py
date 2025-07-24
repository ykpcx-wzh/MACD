import argparse

import torch
from copy import deepcopy
from model.mlp_model import MLPModel
from model.qmixer import QMixerModel


class MlpQmixAlgo:
    def __init__(
        self,
        agent_model: MLPModel,
        qmixer_model: QMixerModel,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.args = args
        self.double_q = args.double_q
        self.gamma = args.gamma
        self.lr = args.lr
        self.clip_grad_norm = args.clip_grad_norm

        self.device = device
        self.agent_model = agent_model.to(self.device)
        self.target_agent_model = deepcopy(self.agent_model).to(self.device)
        self.qmixer_model = qmixer_model.to(self.device)
        self.target_qmixer_model = deepcopy(self.qmixer_model).to(self.device)

        self.param = list(agent_model.parameters()) + list(qmixer_model.parameters())
        self.optimizer = torch.optim.Adam(self.param, lr=self.lr, eps=1e-5)

    def hard_sync_target(self) -> None:
        self.target_qmixer_model.load_state_dict(self.qmixer_model.state_dict())
        self.target_agent_model.load_state_dict(self.agent_model.state_dict())

    def soft_sync_target(self, tau: float = 1.0) -> None:
        models_pair = [
            [self.target_agent_model, self.agent_model],
            [self.target_qmixer_model, self.qmixer_model],
        ]
        for pair in models_pair:
            target, source = pair
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.data.copy_(
                    tau * source_param.data + (1 - tau) * target_param.data
                )

    def train(self):
        self.agent_model.train()
        self.qmixer_model.train()
        self.target_qmixer_model.train()
        self.target_agent_model.train()

    def eval(self):
        self.agent_model.eval()
        self.qmixer_model.eval()
        self.target_qmixer_model.eval()
        self.target_agent_model.eval()

    def predict_local_q(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent_model(obs)

    def learn(
        self,
        state_batch: torch.Tensor,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        terminated_batch: torch.Tensor,
        next_obs_batch: torch.Tensor,
    ) -> tuple[float, float]:
        """

        Args:
            state_batch: [batch_size, state_size]
            obs_batch:  [batch_size, n_agent, obs_size]
            action_batch:
            reward_batch:
            terminated_batch:
            next_obs_batch:

        Returns:

        """
        batch_size = state_batch.shape[0]
        mask = 1 - terminated_batch
        losses = []
        mean_td_errors = []

        for _ in range(self.args.num_epochs_per_episode):
            local_qs = self.agent_model(obs_batch)

            target_local_qs = self.target_agent_model(next_obs_batch)

            chosen_action_local_qs = torch.gather(
                local_qs, index=action_batch.unsqueeze(-1), dim=2
            )

            if self.double_q:
                local_qs_detach = local_qs.clone().detach()
                current_max_actions = local_qs_detach.max(dim=2, keepdim=True)[
                    1
                ]  # 返回索引
                target_local_max_qs = torch.gather(
                    target_local_qs, dim=2, index=current_max_actions
                )
            else:
                target_local_max_qs = target_local_qs.max(dim=2)[0]

            chosen_action_global_qs = self.qmixer_model(
                chosen_action_local_qs, state_batch
            )
            target_global_max_qs = self.target_qmixer_model(
                target_local_max_qs, state_batch
            )

            target = (
                reward_batch
                + self.gamma * (1 - terminated_batch) * target_global_max_qs
            )

            # ==================================
            # TD误差
            # ==================================
            td_error = target.detach() - chosen_action_global_qs
            masked_td_error = td_error * mask
            mean_td_error = masked_td_error.sum() / mask.sum()
            loss = (mean_td_error**2).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.param, self.clip_grad_norm)
            self.optimizer.step()

            losses.append(loss.item())
            mean_td_errors.append(mean_td_error.item())

        return sum(losses) / len(losses), sum(mean_td_errors) / len(losses)
