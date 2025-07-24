import os
from typing import Literal

from loguru import logger

import numpy as np
import torch
from algos.qmix_teach import QMIXTeach
from utils import AvailableActionsSampler


class QMixTeachAgent:
    def __init__(
            self,
            algorithm: QMIXTeach,
            exploration_start: float,
            min_exploration: float,
            exploration_decay: float,
            update_target_interval: int,
            config: dict,
    ) -> None:
        self.alg = algorithm
        self.global_step = 0
        self.global_step_student = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay

        self.target_update_count = 0
        self.update_target_interval = update_target_interval

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.saved_teacher = False
        self.loaded_teacher = False
        self.config = config

    def save_best_teacher(self, save_dir: str, win_rate: float, steps: int) -> None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        agent_model_teacher_path = os.path.join(save_dir, "best_teacher.pt")
        qmixer_model_path = os.path.join(save_dir, "best_qmix.pt")
        dreamer_path = os.path.join(save_dir, "best_dreamer.pt")

        torch.save(self.alg.agent_model_teacher.state_dict(), agent_model_teacher_path)
        torch.save(self.alg.qmixer_model.state_dict(), qmixer_model_path)
        torch.save(self.alg.dreamer_model.state_dict(), dreamer_path)

        with open(os.path.join(save_dir, "best_info.txt"), 'a+') as f:
            f.write(f"在{steps}步保存此模型，测试胜率为:{win_rate}\n")

        logger.info(
            f"Saved teacher agent, qmixer and dreamer model to {save_dir}, steps is {steps}, win_rate is {win_rate}")

    def save_last_teacher(self, save_dir: str, win_rate: float, steps: int) -> None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        agent_model_teacher_path = os.path.join(save_dir, "last_teacher.pt")
        qmixer_model_path = os.path.join(save_dir, "last_qmix.pt")
        dreamer_path = os.path.join(save_dir, "last_dreamer.pt")

        torch.save(self.alg.agent_model_teacher.state_dict(), agent_model_teacher_path)
        torch.save(self.alg.qmixer_model.state_dict(), qmixer_model_path)
        torch.save(self.alg.dreamer_model.state_dict(), dreamer_path)

        with open(os.path.join(save_dir, "last_info.txt"), 'a+') as f:
            f.write(f"在{steps}步保存此模型，测试胜率为:{win_rate}\n")

        logger.debug(
            f"Saved last teacher agent, qmixer and dreamer model to {save_dir}, steps is {steps}, win_rate is {win_rate}")

    def load_teacher_and_dreamer(self, weight_path: str, weight_type: Literal['best', 'last', 'pymarl']):
        if not os.path.exists(weight_path):
            return False

        if weight_type == 'best':
            teacher_name = "best_teacher.pt"
            qmix_name = "best_qmix.pt"
            dreamer_name = "best_dreamer.pt"
        elif weight_type == 'last':
            teacher_name = "last_teacher.pt"
            qmix_name = "last_qmix.pt"
            dreamer_name = "last_dreamer.pt"
        elif weight_type == "pymarl":
            teacher_name = "agent.th"
            qmix_name = "mixer.th"
            dreamer_name = "dreamer.th"
        else:
            raise ValueError

        info = weight_type

        res1 = self.alg.agent_model_teacher.load_state_dict(
            torch.load(os.path.join(weight_path, teacher_name), weights_only=True))
        res2 = self.alg.qmixer_model.load_state_dict(
            torch.load(os.path.join(weight_path, qmix_name), weights_only=True))
        res3 = self.alg.dreamer_model.load_state_dict(
            torch.load(os.path.join(weight_path, dreamer_name), weights_only=True)
        )
        if (len(res1.missing_keys) == 0 and len(res1.unexpected_keys) == 0 and len(res2.missing_keys) == 0
                and len(res2.unexpected_keys) == 0 and len(res3.missing_keys) == 0 and len(res3.unexpected_keys) == 0):
            logger.info(f"successfully loaded {info} teacher_agent, teacher_qmix and dreamer_agent!")
            return True
        else:
            return False

    def reset_agent(self, batch_size: int = 1) -> None:
        self.alg.init_hidden_states(batch_size)
        # self.alg.init_hidden_states_student(batch_size)

    def sample(self, obs: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions, agent_type="teacher")
        else:
            actions = AvailableActionsSampler(available_actions).sample()

        self.exploration = max(
            self.min_exploration, self.exploration - self.exploration_decay
        )

        return actions

    def sample_stu(self, obs: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions, agent_type="student")
        else:
            actions = AvailableActionsSampler(available_actions).sample()

        # self.exploration = max(self.min_exploration, self.exploration - self.exploration_decay)

        return actions

    def predict(
            self,
            obs: np.ndarray,
            available_actions: np.ndarray,
            agent_type: str = "student",
    ) -> np.ndarray:

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.int32, device=self.device
        )

        if agent_type in ["student", "teacher", "rnd"]:
            agents_q, self.alg.student_hidden_states = self.alg.predict_local_q(
                obs, self.alg.student_hidden_states, agent_type
            )
        else:
            raise ValueError

        unavailable_actions_mask = (available_actions == 0).to(torch.float32)
        agents_q -= unavailable_actions_mask * 1e8
        actions = torch.argmax(agents_q, dim=-1).detach().cpu().numpy()
        return actions

    def student_learn(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float]:
        """
        Args:
            state_batch (np.ndarray):             (batch_size, episode_length, state_shape)
            actions_batch (np.ndarray):           (batch_size, episode_length, action_shape)
            reward_batch (np.ndarray):            (batch_size, episode_length, 1)
            terminated_batch (np.ndarray):        (batch_size, episode_length, 1)
            obs_batch (np.ndarray):               (batch_size, episode_length, n_agent, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, episode_length, n_agent, n_actions)
            filled_batch (np.ndarray):            (batch_size, episode_length, 1)
        returns:
            mean_loss (float):
            mean_td_error (float):
            kd_loss (float):
        """
        if self.global_step_student % self.update_target_interval == 0:
            self.alg.sync_target_for_student()
            self.target_update_count += 1

        self.global_step_student += 1

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

        mean_loss, mean_td_error = self.alg.student_learn(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return mean_loss, mean_td_error

    def student_learn_with_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            factor: np.ndarray,
    ) -> tuple[float, float]:
        """
        Args:
            factor:
            state_batch (np.ndarray):             (batch_size, episode_length, state_shape)
            actions_batch (np.ndarray):           (batch_size, episode_length, action_shape)
            reward_batch (np.ndarray):            (batch_size, episode_length, 1)
            terminated_batch (np.ndarray):        (batch_size, episode_length, 1)
            obs_batch (np.ndarray):               (batch_size, episode_length, n_agent, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, episode_length, n_agent, n_actions)
            filled_batch (np.ndarray):            (batch_size, episode_length, 1)
        returns:
            mean_loss (float):
            mean_td_error (float):
            kd_loss (float):
        """
        if self.global_step_student % self.update_target_interval == 0:
            self.alg.sync_target_for_student()
            self.target_update_count += 1

        self.global_step_student += 1

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

        factor_batch = torch.tensor(factor, dtype=torch.float32, device=self.device)

        mean_loss, mean_td_error = self.alg.student_learn_with_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            factor_batch,
        )

        return mean_loss, mean_td_error

    def student_learn_with_vq_vae_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            factor: np.ndarray,
    ) -> tuple[float, float]:
        """
        Args:
            factor:
            state_batch (np.ndarray):             (batch_size, episode_length, state_shape)
            actions_batch (np.ndarray):           (batch_size, episode_length, action_shape)
            reward_batch (np.ndarray):            (batch_size, episode_length, 1)
            terminated_batch (np.ndarray):        (batch_size, episode_length, 1)
            obs_batch (np.ndarray):               (batch_size, episode_length, n_agent, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, episode_length, n_agent, n_actions)
            filled_batch (np.ndarray):            (batch_size, episode_length, 1)
        returns:
            mean_loss (float):
            mean_td_error (float):
            kd_loss (float):
        """
        if self.global_step_student % self.update_target_interval == 0:
            self.alg.sync_target_for_student()
            self.target_update_count += 1

        self.global_step_student += 1

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

        factor_batch = torch.tensor(factor, dtype=torch.float32, device=self.device)

        mean_loss, mean_td_error = self.alg.student_learn_with_vq_vae_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            factor_batch,
        )

        return mean_loss, mean_td_error

    def learn_only_teacher(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float]:
        """
        Args:
            state_batch (np.ndarray):             (batch_size, episode_length, state_shape)
            actions_batch (np.ndarray):           (batch_size, episode_length, action_shape)
            reward_batch (np.ndarray):            (batch_size, episode_length, 1)
            terminated_batch (np.ndarray):        (batch_size, episode_length, 1)
            obs_batch (np.ndarray):               (batch_size, episode_length, n_agent, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, episode_length, n_agent, n_actions)
            filled_batch (np.ndarray):            (batch_size, episode_length, 1)
        returns:
            mean_loss (float):
            mean_td_error (float):
            kd_loss (float):
        """
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

        return self.alg.learn_only_teacher(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

    def kd_student(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float]:

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

        mean_kd_loss, mean_kd_error = self.alg.kd_student(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return mean_kd_loss, mean_kd_error

    def kd_rnd(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float]:

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

        mean_kd_loss, mean_kd_error = self.alg.kd_rnd(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return mean_kd_loss, mean_kd_error

    def vq_vae_learn(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float, float]:

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

        recons_loss, vq_loss, loss = self.alg.vq_vae_learn(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return recons_loss, vq_loss, loss

    def dreamer_learn(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float]:

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

        return self.alg.dreamer_learn(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

    def kd_rnd_with_qENC(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float]:

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

        mean_kd_loss, mean_kd_error = self.alg.kd_rnd_with_qENC(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return mean_kd_loss, mean_kd_error

    def get_count_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> np.ndarray:

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

        batch_count_factor = self.alg.get_count_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return batch_count_factor

    def get_rnd_cos_sim_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> float:

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

        scatter_count_factor = self.alg.get_rnd_count_factor_cosine_similarity(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        return scatter_count_factor

    def get_vq_vae_count_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[np.ndarray, float]:

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

        return self.alg.get_vq_vae_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

    def get_dreamer_count_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            _type: str
    ):

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

        return self.alg.get_dreamer_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            _type,
        )

    def get_dreamer_count_factor_with_teacher_baseline(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray
    ):

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

        return self.alg.get_dreamer_factor_with_teacher_baseline(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch
        )

    def kd_student_with_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            count_factor: np.ndarray,
    ) -> tuple[float, float]:

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
        count_factor = torch.tensor(
            count_factor, dtype=torch.float32, device=self.device
        )

        mean_kd_loss, mean_kd_error = self.alg.kd_student_with_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            count_factor,
        )

        return mean_kd_loss, mean_kd_error

    def kd_student_with_vq_vae_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            count_factor: np.ndarray,
    ) -> tuple[float, float]:

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
        count_factor = torch.tensor(
            count_factor, dtype=torch.float32, device=self.device
        )

        return self.alg.kd_student_with_vq_vae_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            count_factor,
        )

    def kd_student_with_dreamer_factor(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            count_factor: np.ndarray,
    ) -> tuple[float, float]:

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
        count_factor = torch.tensor(
            count_factor, dtype=torch.float32, device=self.device
        )

        return self.alg.kd_student_with_dreamer_factor(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            count_factor,
        )

    def kd_stu_with_factor_train_stu(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray, float]:

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

        loss_kd, loss_td, loss_all, normed_count_factor, loss_rnd_kd = (
            self.alg.kd_stu_with_factor_train_stu(
                state_batch,
                actions_batch,
                reward_batch,
                terminated_batch,
                obs_batch,
                available_actions_batch,
                filled_batch,
            )
        )

        return loss_kd, loss_td, loss_all, normed_count_factor, loss_rnd_kd

    def kd_student_with_mask_td_stu_with_mask(
            self,
            state_batch: np.ndarray,
            actions_batch: np.ndarray,
            reward_batch: np.ndarray,
            terminated_batch: np.ndarray,
            obs_batch: np.ndarray,
            available_actions_batch: np.ndarray,
            filled_batch: np.ndarray,
            mask: np.ndarray,
    ) -> tuple[float, float, float]:

        if self.global_step_student % self.update_target_interval == 0:
            self.alg.sync_target_for_student()
            self.target_update_count += 1

        self.global_step_student += 1

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

        mask = torch.tensor(mask, dtype=torch.int32, device=self.device)

        return self.alg.kd_student_with_mask_td_stu_with_mask(
            state_batch,
            actions_batch,
            reward_batch,
            terminated_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            mask,
        )
