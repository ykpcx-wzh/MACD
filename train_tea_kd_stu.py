import argparse
import os.path
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import swanlab
from loguru import logger
import pandas as pd

from agent.qmix_teach_agent import QMixTeachAgent
from algos.qmix_teach import QMIXTeach
from buffer.episode_buffer import EpisodeExperience, EpisodeReplayBuffer
from envs import REGISTRY
from envs.find_one_goal.find_one_goal import FindOneGoal
from envs.ic3net_envs.predator_prey_env_warpper import PredatorPreyEnvWarpper
from envs.multiagentenv import MultiAgentEnv
from model.qmixer import QMixerModel
from model.vdn import VDNMixer
from model.qatten import QattenMixer
from model.nmixer import QMixerModel as NQMixerModel
from model.rnn_model import (RNNModel, RNNTransModel, RNNSelfAttnModel, RNNDreamerModel, RNNLinearSelfAttnModel,
                             RNNLinearFC3SelfAttnModel, RNNMultiHeadAttnModel, CADPATTRNNAgent, ATTRNNAgent)
from model.vq_vae import LinearVQVAE


@torch.no_grad()
def collect_teacher_rollout(
        env: MultiAgentEnv | PredatorPreyEnvWarpper | FindOneGoal,
        agent: QMixTeachAgent,
        teacher_rpm: EpisodeReplayBuffer,
        config: dict,
):
    episode_limit = config["episode_limit"]
    agent.reset_agent()
    agent.alg.eval()

    # ------------------------------------------------
    # 从一次游戏中收集一条完整的经验
    # --------------------------------------------------
    state, obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    episode_experience = EpisodeExperience(episode_limit)
    while True:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        next_state, next_obs, reward, terminated, truncated = env.step(actions)
        episode_reward += reward
        episode_step += 1
        episode_experience.add(
            state, actions, [reward], [terminated], obs, available_actions, [0]
        )
        state = next_state
        obs = next_obs

        if terminated or truncated:
            break

    teacher_steps = episode_step

    # ===================================================================
    # 每条经验里面存储的步数信息是固定长度的，如果游戏提前结束，那么剩下的用零填充
    # ====================================================================
    state_zero = np.zeros_like(state, dtype=state.dtype)
    actions_zero = np.zeros_like(actions, dtype=actions.dtype)
    obs_zero = np.zeros_like(obs, dtype=obs.dtype)
    available_actions_zero = np.zeros_like(
        available_actions, dtype=available_actions.dtype
    )
    reward_zero = 0
    terminated_zero = True
    for _ in range(episode_step, episode_limit):
        episode_experience.add(
            state_zero,
            actions_zero,
            [reward_zero],
            [terminated_zero],
            obs_zero,
            available_actions_zero,
            [1],
        )
    teacher_rpm.add(episode_experience)

    return teacher_steps


@torch.no_grad()
def collect_student_rollout(
        env: MultiAgentEnv | FindOneGoal | PredatorPreyEnvWarpper,
        agent: QMixTeachAgent,
        student_rpm: EpisodeReplayBuffer,
        config: dict,
):
    agent.alg.eval()
    agent.reset_agent()
    # ------------------------------------------------
    # 从一次游戏中收集一条完整的经验
    # --------------------------------------------------
    episode_limit = config["episode_limit"]
    state, obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    episode_experience = EpisodeExperience(episode_limit)

    while True:
        available_actions = env.get_available_actions()
        actions = agent.sample_stu(obs, available_actions)
        next_state, next_obs, reward, terminated, truncated = env.step(actions)
        episode_reward += reward
        episode_step += 1
        episode_experience.add(
            state, actions, [reward], [terminated], obs, available_actions, [0]
        )
        state = next_state
        obs = next_obs

        if terminated or truncated:
            break

    # ===================================================================
    # 每条经验里面存储的步数信息是固定长度的，如果游戏提前结束，那么剩下的用零填充
    # ====================================================================
    state_zero = np.zeros_like(state, dtype=state.dtype)
    actions_zero = np.zeros_like(actions, dtype=actions.dtype)
    obs_zero = np.zeros_like(obs, dtype=obs.dtype)
    available_actions_zero = np.zeros_like(
        available_actions, dtype=available_actions.dtype
    )
    reward_zero = 0
    terminated_zero = True
    for _ in range(episode_step, episode_limit):
        episode_experience.add(
            state_zero,
            actions_zero,
            [reward_zero],
            [terminated_zero],
            obs_zero,
            available_actions_zero,
            [1],
        )
    student_rpm.add(episode_experience)

    return episode_step


def collect_rollout(
        env: MultiAgentEnv | FindOneGoal | PredatorPreyEnvWarpper,
        agent: QMixTeachAgent,
        student_rpm: EpisodeReplayBuffer,
        teacher_rpm: EpisodeReplayBuffer,
        config: dict,
) -> int:
    train_mode = config["train_mode"]
    if train_mode == "train_tea" or train_mode == "train_tea_and_dreamer":
        return collect_teacher_rollout(env, agent, teacher_rpm, config)
    elif train_mode == "train_tea_kd_stu" or train_mode == "train_tea_kd_stu_rnd":
        return collect_teacher_rollout(env, agent, teacher_rpm, config)
    elif (
            train_mode == "train_tea_kd_stu_train_stu"
            or train_mode == "train_tea_kd_stu_train_stu_rnd"
            or train_mode == "train_tea_kd_stu_train_stu_rnd_TEST"
            or train_mode == "train_tea_kd_stu_train_stu_vq_vae"
            or train_mode == "train_tea_kd_stu_train_stu_dreamer_count"
            or train_mode == "pretrained_teacher_kd_td_stu_with_dreamer"
            or train_mode == "train_tea__train_kd_stu_with_RND"
    ):
        collect_student_rollout(env, agent, student_rpm, config)
        return collect_teacher_rollout(env, agent, teacher_rpm, config)
    else:
        raise NotImplementedError


def kd_student(
        teacher_rpm: EpisodeReplayBuffer, qmix_agent: QMixTeachAgent, config: dict
):
    qmix_agent.alg.train()
    data = teacher_rpm.sample_batch(config["batch_size"])
    loss, kd_error = qmix_agent.kd_student(*data)
    return loss, kd_error


def train_tea_kd_stu_train_stu(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> dict:
    qmix_agent.alg.train()
    tea_loss, kd_loss, stu_td_loss = [0.0] * 3
    if current_steps < config["teacher_train_steps"]:
        tea_data = teacher_rpm.sample_batch(config["batch_size"])
        tea_loss, _ = qmix_agent.learn_only_teacher(*tea_data)
    else:
        stu_data = student_rpm.sample_batch(config["batch_size"])
        kd_loss, _ = qmix_agent.kd_student(*stu_data)
        stu_td_loss, _ = qmix_agent.student_learn(*stu_data)

    return {
        "teacher_loss": tea_loss,
        "student_kd_loss": kd_loss,
        "student_td_loss": stu_td_loss,
    }


def train_only_teacher_td(
        teacher_rpm: EpisodeReplayBuffer, qmix_agent: QMixTeachAgent, config: dict
) -> dict:
    qmix_agent.alg.train()
    tea_data = teacher_rpm.sample_batch(config["batch_size"])
    loss, td_error = qmix_agent.learn_only_teacher(*tea_data)
    return {"teacher_loss": loss, "teacher_error": td_error}


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> tuple[np.ndarray, float, float, float, float, float, float, float]:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    data = teacher_rpm.sample_batch(config["batch_size"])
    teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
    if current_steps < 100_0000:
        kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
        big = 0.0
    else:
        kd_count_factor, big = qmix_agent.get_vq_vae_count_factor(*data)

    student_kd_loss, _ = qmix_agent.kd_student_with_vq_vae_factor(*data, kd_count_factor)

    stu_data = student_rpm.sample_batch(config["batch_size"])
    student_td_loss, _ = qmix_agent.student_learn(*stu_data)
    recons_loss, vq_loss, vq_all_loss = qmix_agent.vq_vae_learn(*data)

    return (
        kd_count_factor,
        teacher_td_loss,
        student_kd_loss,
        student_td_loss,
        recons_loss,
        vq_loss,
        vq_all_loss,
        big,
    )


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae_first_train_tea_then_stu(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
    teacher_td_loss, student_kd_loss, student_td_loss, recons_loss, vq_loss, vq_all_loss = [0.0] * 6

    if current_steps < 300_0000:
        data = teacher_rpm.sample_batch(config["batch_size"])
        teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
        recons_loss, vq_loss, vq_all_loss = qmix_agent.vq_vae_learn(*data)
    else:
        stu_data = student_rpm.sample_batch(config["batch_size"])

        kd_count_factor, big = qmix_agent.get_vq_vae_count_factor(*stu_data)

        student_kd_loss, _ = qmix_agent.kd_student_with_vq_vae_factor(*stu_data, kd_count_factor)
        student_td_loss, _ = qmix_agent.student_learn_with_vq_vae_factor(*stu_data, np.ones(
            shape=(config["batch_size"], config["episode_limit"])))

    return (
        kd_count_factor,
        teacher_td_loss,
        student_kd_loss,
        student_td_loss,
        recons_loss,
        vq_loss,
        vq_all_loss,
    )


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_first_train_tea_then_stu(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> dict:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
    teacher_td_loss, student_kd_loss, student_td_loss, dreamer_loss, hash_area, mask_area = [0.0] * 6

    if current_steps < 200_0000:
        if config["skip_teacher_training"]:
            return {}
        data = teacher_rpm.sample_batch(config["batch_size"])
        teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
        dreamer_loss = qmix_agent.dreamer_learn(*data)
    else:

        stu_data = student_rpm.sample_batch(config["batch_size"])

        kd_count_factor, hash_area, mask_area = qmix_agent.get_dreamer_count_factor(*stu_data)

        if mask_area > 0:
            student_kd_loss, _ = qmix_agent.kd_student_with_dreamer_factor(*stu_data, kd_count_factor)
            student_td_loss, _ = qmix_agent.student_learn_with_vq_vae_factor(*stu_data, np.ones(
                shape=(config["batch_size"], config["episode_limit"])))

    return {
        "kd_count_factor_max": kd_count_factor.max(),
        "kd_count_factor_mean": kd_count_factor.mean(),
        "teacher_td_loss": teacher_td_loss,
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "dreamer_loss": dreamer_loss,
        "hash_area": hash_area,
        "mask_area": mask_area,
    }


def train_tea_and_dreamer(
        teacher_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    """

    Args:
        teacher_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()

    data = teacher_rpm.sample_batch(config["batch_size"])
    teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
    dreamer_loss = qmix_agent.dreamer_learn(*data)

    return {
        "teacher_td_loss": teacher_td_loss,
        "dreamer_loss": dreamer_loss,
    }


def pretrained_teacher_kd_td_stu_with_dreamer(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()
    student_kd_loss, student_td_loss, all_loss = 0.0, 0.0, 0.0

    stu_data = student_rpm.sample_batch(config["batch_size"])
    dreamer_mask, hash_area, mask_area = qmix_agent.get_dreamer_count_factor(*stu_data, config["hash_type"])

    if mask_area > 0:
        student_kd_loss, student_td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(
            *stu_data, dreamer_mask)

    return {
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "all_loss": all_loss,
        "hash_area": hash_area,
        "mask_area": mask_area,
    }

def pretrained_teacher_kd_td_stu(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()

    stu_data = student_rpm.sample_batch(config["batch_size"])
    kd_loss, kd_error = qmix_agent.kd_student(*stu_data)
    td_loss, td_error = qmix_agent.student_learn(*stu_data)

    return {
        "student_kd_loss": kd_loss,
        "student_td_loss": td_loss,
    }


def pretrained_teacher_kd_stu(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()

    stu_data = student_rpm.sample_batch(config["batch_size"])
    kd_loss, kd_error = qmix_agent.kd_student(*stu_data)

    return {
        "student_kd_loss": kd_loss,
    }


def pretrained_teacher_kd_td_stu_with_dreamer_and_restored_mask_area(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: dict,
) -> dict:
    qmix_agent.alg.train()
    student_kd_loss, student_td_loss, all_loss = 0.0, 0.0, 0.0

    stu_data = student_rpm.sample_batch(config["batch_size"])
    # dreamer_mask, hash_area, mask_area = qmix_agent.get_dreamer_count_factor(*stu_data, config["hash_type"])
    index = current_steps // config["episode_max_steps"]
    mask_area =  config["log_detail"].at[index, "mask_area"]

    dreamer_mask = torch.distributions.Bernoulli(
        torch.ones(config["batch_size"], config["episode_limit"]) * mask_area).sample().numpy()

    if mask_area > 0:
        student_kd_loss, student_td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(
            *stu_data, dreamer_mask)

    return {
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "all_loss": all_loss,
        "mask_area": mask_area,
    }


def pretrained_teacher_kd_td_stu_with_static_factor(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()

    mask_area_factor = config["mask_area_factor"]

    stu_data = student_rpm.sample_batch(config["batch_size"])
    dreamer_mask = torch.distributions.Bernoulli(
        torch.ones(config["batch_size"], config["episode_limit"]) * mask_area_factor).sample().numpy()
    student_kd_loss, student_td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(
        *stu_data, dreamer_mask)

    return {
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "all_loss": all_loss,
        "mask_area": np.mean(dreamer_mask),
    }


# 这是上面这个方法的对照组
def pretrained_teacher_kd_td_stu_without_dreamer(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()

    stu_data = student_rpm.sample_batch(config["batch_size"])

    student_kd_loss, _ = qmix_agent.kd_student(*stu_data)
    student_td_loss, _ = qmix_agent.student_learn(*stu_data)

    return {
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
    }


def pretrained_teacher_kd_td_stu_with_dreamer_with_teacher_baseline(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict:
    qmix_agent.alg.train()
    student_kd_loss, student_td_loss, all_loss = 0.0, 0.0, 0.0

    stu_data = student_rpm.sample_batch(config["batch_size"])
    dreamer_mask, hash_area, mask_area = qmix_agent.get_dreamer_count_factor_with_teacher_baseline(*stu_data)

    if mask_area > 0:
        student_kd_loss, student_td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(
            *stu_data, dreamer_mask)

    return {
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "all_loss": all_loss,
        "hash_area": hash_area,
        "mask_area": mask_area,
    }


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_DuiZhaoZu(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> dict:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
    teacher_td_loss, student_kd_loss, student_td_loss, dreamer_loss, hash_area, mask_area = [0.0] * 6

    if current_steps < 100_0000:
        data = teacher_rpm.sample_batch(config["batch_size"])
        teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
        dreamer_loss = qmix_agent.dreamer_learn(*data)
    else:

        stu_data = student_rpm.sample_batch(config["batch_size"])

        student_kd_loss, _ = qmix_agent.kd_student_with_dreamer_factor(*stu_data, np.ones(
            shape=(config["batch_size"], config["episode_limit"])))
        student_td_loss, _ = qmix_agent.student_learn_with_vq_vae_factor(*stu_data, np.ones(
            shape=(config["batch_size"], config["episode_limit"])))

    return {
        "kd_count_factor_max": kd_count_factor.max(),
        "kd_count_factor_mean": kd_count_factor.mean(),
        "teacher_td_loss": teacher_td_loss,
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "dreamer_loss": dreamer_loss,
        "hash_area": hash_area,
        "mask_area": mask_area,
    }


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_first_train_tea_then_stu_mask4KdTd(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> dict:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
    teacher_td_loss, kd_loss, td_loss, dreamer_loss, hash_area, mask_area, all_loss = [0.0] * 7

    if current_steps < config["teacher_train_steps"]:
        if config["skip_teacher_training"]:
            return {}
        data = teacher_rpm.sample_batch(config["batch_size"])
        teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
        dreamer_loss = qmix_agent.dreamer_learn(*data)
    else:

        stu_data = student_rpm.sample_batch(config["batch_size"])

        kd_count_factor, hash_area, mask_area = qmix_agent.get_dreamer_count_factor(*stu_data)
        mask = kd_count_factor > 0
        if mask_area > 0:
            kd_loss, td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(*stu_data, mask)

    return {
        "kd_count_factor_max": kd_count_factor.max(),
        "kd_count_factor_mean": kd_count_factor.mean(),
        "teacher_td_loss": teacher_td_loss,
        "student_kd_loss": kd_loss,
        "student_td_loss": td_loss,
        "student_all_loss": all_loss,
        "dreamer_loss": dreamer_loss,
        "hash_area": hash_area,
        "mask_area": mask_area,
    }


def teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae_test_no_factor(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        current_steps: int,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """

    Args:
        current_steps:
        teacher_rpm:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    kd_count_factor = np.ones(shape=(config["batch_size"], config["episode_limit"]))
    teacher_td_loss, student_kd_loss, student_td_loss, recons_loss, vq_loss, vq_all_loss = [0.0] * 6

    if current_steps < 100_0000:
        data = teacher_rpm.sample_batch(config["batch_size"])
        teacher_td_loss, _ = qmix_agent.learn_only_teacher(*data)
        recons_loss, vq_loss, vq_all_loss = qmix_agent.vq_vae_learn(*data)
    else:
        stu_data = student_rpm.sample_batch(config["batch_size"])

        student_kd_loss, _ = qmix_agent.kd_student_with_vq_vae_factor(*stu_data, kd_count_factor)
        student_td_loss, _ = qmix_agent.student_learn_with_vq_vae_factor(*stu_data, kd_count_factor)

    return (
        kd_count_factor,
        teacher_td_loss,
        student_kd_loss,
        student_td_loss,
        recons_loss,
        vq_loss,
        vq_all_loss,
    )


def kd_td_student_with_factor_and_train_stu_devided_no_factor_td_stu(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> tuple[np.ndarray, float, float, float]:
    """

    Args:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    stu_data = student_rpm.sample_batch(config["batch_size"])

    count_factor = qmix_agent.get_count_factor(*stu_data)
    kd_stu_loss, _ = qmix_agent.kd_student_with_factor(*stu_data, count_factor)

    td_stu_loss, _ = qmix_agent.student_learn_with_factor(*stu_data, np.ones_like(count_factor))

    kd_rnd_loss, _ = qmix_agent.kd_rnd(*stu_data)

    return count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss


def kd_td_stu_with_rnd_cosine_factor(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> dict[str, float]:
    """

    Args:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    stu_data = student_rpm.sample_batch(config["batch_size"])

    count_factor = qmix_agent.get_rnd_cos_sim_factor(*stu_data)
    kd_stu_loss, _ = qmix_agent.kd_student_with_factor(*stu_data, (1 - count_factor))

    td_stu_loss, _ = qmix_agent.student_learn_with_factor(*stu_data, count_factor)

    kd_rnd_loss, _ = qmix_agent.kd_rnd(*stu_data)

    info = {
        "student_kd_loss": kd_stu_loss,
        "student_td_loss": td_stu_loss,
        "kd_rnd_loss": kd_rnd_loss,
        "count_factor_min": count_factor.min(),
        "count_factor_max": count_factor.max(),
        "count_factor_mean": count_factor.mean(),
    }

    return info


def kd_td_stu_kd_rnd_with_qENC(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
) -> tuple[np.ndarray, float, float, float, float]:
    """

    Args:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss, kd_qENC_loss
    """
    qmix_agent.alg.train()
    (
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
    ) = student_rpm.sample_batch(config["batch_size"])

    count_factor = qmix_agent.get_count_factor(  # [B x T - 1 x N x A]
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
    )

    kd_stu_loss, _ = qmix_agent.kd_student_with_factor(
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
        count_factor,
    )

    td_stu_loss, _ = qmix_agent.student_learn_with_factor(
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
        np.ones_like(count_factor),
    )

    kd_rnd_loss, kd_qENC_loss = qmix_agent.kd_rnd_with_qENC(
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
    )

    return count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss, kd_qENC_loss


def kd_td_student_with_factor_and_train_stu_test(
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
        test_mode: str,  # ["pure_kd", "pure_td"]
) -> tuple[np.ndarray, float, float, float]:
    """

    Args:
        test_mode:
        student_rpm:
        qmix_agent:
        config:

    Returns:
        count_factor, kd_stu_loss, td_stu_loss, kd_rnd_loss
    """
    qmix_agent.alg.train()
    (
        s_batch,
        a_batch,
        r_batch,
        t_batch,
        obs_batch,
        available_actions_batch,
        filled_batch,
    ) = student_rpm.sample_batch(config["batch_size"])

    if test_mode == "pure_kd":
        count_factor = qmix_agent.get_count_factor(  # [B x T - 1 x N x A]
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        kd_stu_loss, _ = qmix_agent.kd_student_with_factor(
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            np.ones_like(count_factor),
        )

        td_stu_loss, _ = qmix_agent.student_learn_with_factor(
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            np.zeros_like(count_factor),
        )

    elif test_mode == "pure_td":
        count_factor = qmix_agent.get_count_factor(  # [B x T - 1 x N x A]
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
        )

        kd_stu_loss, _ = qmix_agent.kd_student_with_factor(
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            np.zeros_like(count_factor),
        )

        td_stu_loss, _ = qmix_agent.student_learn_with_factor(
            s_batch,
            a_batch,
            r_batch,
            t_batch,
            obs_batch,
            available_actions_batch,
            filled_batch,
            np.ones_like(count_factor),
        )

    else:
        raise ValueError(f"test_mode {test_mode} is not supported.")

    return count_factor, kd_stu_loss, td_stu_loss, 0.0


def train_tea__train_kd_stu_with_RND__cosine_sim_factor(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        config: dict,
):
    qmix_agent.alg.train()
    tea_data = teacher_rpm.sample_batch(config["batch_size"])
    teacher_td_loss, _ = qmix_agent.learn_only_teacher(*tea_data)

    stu_data = student_rpm.sample_batch(config["batch_size"])
    scatter_count_factor = qmix_agent.get_rnd_cos_sim_factor(*stu_data)

    dreamer_mask = torch.distributions.Bernoulli(
        torch.ones(config["batch_size"], config["episode_limit"]) * scatter_count_factor).sample().numpy()

    student_kd_loss, student_td_loss, all_loss = qmix_agent.kd_student_with_mask_td_stu_with_mask(
        *stu_data, dreamer_mask)

    loss_rnd, _ = qmix_agent.kd_rnd(*stu_data)

    return {
        "loss_rnd": loss_rnd,
        "scatter_count_factor": scatter_count_factor,
        "teacher_td_loss": teacher_td_loss,
        "student_kd_loss": student_kd_loss,
        "student_td_loss": student_td_loss,
        "all_loss": all_loss,
        "mask_area": np.mean(dreamer_mask),
    }


def train_once(
        teacher_rpm: EpisodeReplayBuffer,
        student_rpm: EpisodeReplayBuffer,
        qmix_agent: QMixTeachAgent,
        kd_times: int,
        config: dict,
        current_steps: int = 0,
):
    log_dict = {}
    if config["train_mode"] == "train_tea":
        teacher_info = train_only_teacher_td(
            teacher_rpm, qmix_agent, config
        )
        log_dict.update(teacher_info)

    elif config["train_mode"] == "train_tea_and_dreamer":
        log_info = train_tea_and_dreamer(teacher_rpm, qmix_agent, config)
        log_dict.update(log_info)

    elif config["train_mode"] == "pretrained_teacher_kd_td_stu_with_dreamer":
        if config["sub_train_mode"] == "ablation":
            log_info = pretrained_teacher_kd_td_stu_without_dreamer(student_rpm, qmix_agent, config)
        elif config["sub_train_mode"] == "default":
            log_info = pretrained_teacher_kd_td_stu_with_dreamer(student_rpm, qmix_agent, config)
        elif config["sub_train_mode"] == "static_factor":
            log_info = pretrained_teacher_kd_td_stu_with_static_factor(student_rpm, qmix_agent, config)
        elif config["sub_train_mode"] == "teacher_baseline":
            log_info = pretrained_teacher_kd_td_stu_with_dreamer_with_teacher_baseline(student_rpm, qmix_agent, config)
        elif config["sub_train_mode"] == "use_stored_mask_area":
            log_info = pretrained_teacher_kd_td_stu_with_dreamer_and_restored_mask_area(student_rpm, qmix_agent, config, current_steps)
        elif config["sub_train_mode"] == "just_kd_and_td":
            log_info = pretrained_teacher_kd_td_stu(student_rpm, qmix_agent, config)
        elif config["sub_train_mode"] == "just_kd":
            log_info = pretrained_teacher_kd_stu(student_rpm, qmix_agent, config)
        else:
            raise NotImplementedError
        log_dict.update(log_info)

    elif config["train_mode"] == "train_tea_kd_stu":
        teacher_info = train_only_teacher_td(
            teacher_rpm, qmix_agent, config
        )
        if kd_times == 0:
            kd_loss, kd_error = 0.0, 0.0
        else:
            kd_losses, kd_errors = [], []
            for _ in range(kd_times):
                kd_loss, kd_error = kd_student(teacher_rpm, qmix_agent, config)
                kd_losses.append(kd_loss)
                kd_errors.append(kd_error)
            kd_loss, kd_error = np.mean(kd_losses), np.mean(kd_errors)

        log_dict.update(teacher_info)
        log_dict.update(
            {
                "kd_loss": kd_loss,
                "kd_error": kd_error,
            }
        )

    elif config["train_mode"] == "train_tea_kd_stu_train_stu":
        log_info = train_tea_kd_stu_train_stu(teacher_rpm, student_rpm, qmix_agent, config, current_steps)
        log_dict.update(log_info)

    elif config["train_mode"] == "train_tea_kd_stu_rnd":
        raise NotImplementedError

    elif config["train_mode"] == "train_tea__train_kd_stu_with_RND":
        if config["sub_train_mode"] == "cosine_sim_factor":
            log_info = train_tea__train_kd_stu_with_RND__cosine_sim_factor(
                teacher_rpm, student_rpm, qmix_agent, config
            )
            log_dict.update(log_info)

        else:
            raise NotImplementedError

    elif config["train_mode"] == "train_tea_kd_stu_train_stu_rnd":
        if config["sub_train_mode"] == "kd_td_stu_kd_rnd_with_qENC":
            teacher_info = train_only_teacher_td(student_rpm, qmix_agent, config)
            count_factor, student_kd_loss, student_td_loss, rnd_kd_loss, qENC_loss = (
                kd_td_stu_kd_rnd_with_qENC(student_rpm, qmix_agent, config)
            )
            log_dict.update({"qENC_loss": qENC_loss, **teacher_info})
        elif (
                config["sub_train_mode"]
                == "kd_td_student_with_factor_and_train_stu_devided_no_factor_td_stu"
        ):
            teacher_info = train_only_teacher_td(student_rpm, qmix_agent, config)
            count_factor, student_kd_loss, student_td_loss, rnd_kd_loss = (
                kd_td_student_with_factor_and_train_stu_devided_no_factor_td_stu(
                    student_rpm, qmix_agent, config
                )
            )
            log_dict.update(**teacher_info)
        else:
            raise NotImplementedError

        log_dict.update(
            {
                "rnd_loss": rnd_kd_loss,
                "kd_stu_loss": student_kd_loss,
                "td_stu_loss": student_td_loss,
                "mean_count_factor_min": count_factor.min(),
                "mean_count_factor_max": count_factor.max(),
                "mean_count_factor_mean": count_factor.mean(),
            }
        )

    elif config["train_mode"] == "train_tea_kd_stu_train_stu_vq_vae":
        big = 0.0
        if config["sub_train_mode"] == "default":
            (
                count_factor,
                teacher_td_loss,
                student_kd_loss,
                student_td_loss,
                recons_loss,
                vq_loss,
                vq_all_loss,
                big,
            ) = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        elif config["sub_train_mode"] == "first_train_tea_then_stu":
            (
                count_factor,
                teacher_td_loss,
                student_kd_loss,
                student_td_loss,
                recons_loss,
                vq_loss,
                vq_all_loss,
            ) = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae_first_train_tea_then_stu(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        elif config["sub_train_mode"] == "train_without_factor":
            (
                count_factor,
                teacher_td_loss,
                student_kd_loss,
                student_td_loss,
                recons_loss,
                vq_loss,
                vq_all_loss,
            ) = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_vq_vae_test_no_factor(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        else:
            raise NotImplementedError

        log_dict.update(
            {
                "teacher_loss": teacher_td_loss,
                "recons_loss": recons_loss,
                "vq_loss": vq_loss,
                "vq_all_loss": vq_all_loss,
                "kd_stu_loss": student_kd_loss,
                "td_stu_loss": student_td_loss,
                "mean_count_factor_min": count_factor.min(),
                "mean_count_factor_max": count_factor.max(),
                "mean_count_factor_mean": count_factor.mean(),
                "big_factor": big
            }
        )
    elif config["train_mode"] == "train_tea_kd_stu_train_stu_rnd_TEST":
        teacher_loss, teacher_error = train_only_teacher_td(
            teacher_rpm, qmix_agent, config
        )
        if kd_times == 0:
            mean_count_factor_min, mean_count_factor_max, mean_count_factor_mean = (
                1.0,
                1.0,
                1.0,
            )
            kd_stu_loss_, td_stu_loss_, kd_rnd_loss_ = 0.0, 0.0, 0.0
        else:
            kd_stu_losses, td_stu_losses, kd_rnd_losses, loss_stu_alls = [], [], [], []
            count_factor_mins, count_factor_maxs, count_factor_means = [], [], []
            for _ in range(kd_times):
                count_factor, kd_stu_loss, td_stu_loss, kd_rnd_los = (
                    kd_td_student_with_factor_and_train_stu_test(
                        student_rpm, qmix_agent, config, test_mode="pure_td"
                    )
                )

                kd_stu_losses.append(kd_stu_loss)
                td_stu_losses.append(td_stu_loss)
                kd_rnd_losses.append(kd_rnd_los)
                count_factor_mins.append(count_factor.min())
                count_factor_maxs.append(count_factor.max())
                count_factor_means.append(count_factor.mean())
            kd_stu_loss_, td_stu_loss_, kd_rnd_loss_ = (
                np.mean(kd_stu_losses),
                np.mean(td_stu_losses),
                np.mean(kd_rnd_losses),
            )
            mean_count_factor_min = np.mean(count_factor_mins)
            mean_count_factor_max = np.mean(count_factor_maxs)
            mean_count_factor_mean = np.mean(count_factor_means)

        log_dict.update(
            {
                "teacher_loss": teacher_loss,
                "rnd_loss": kd_rnd_loss_,
                "kd_stu_loss": kd_stu_loss_,
                "td_stu_loss": td_stu_loss_,
                "mean_count_factor_min": mean_count_factor_min,
                "mean_count_factor_max": mean_count_factor_max,
                "mean_count_factor_mean": mean_count_factor_mean,
            }
        )

    elif config["train_mode"] == "train_tea_kd_stu_train_stu_dreamer_count":
        if config["sub_train_mode"] == "first_train_tea_then_stu":
            log_info = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_first_train_tea_then_stu(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        elif config["sub_train_mode"] == "DuiZhaoZu":
            log_info = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_DuiZhaoZu(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        elif config["sub_train_mode"] == "first_train_tea_then_stu_with_mask4TdKd":
            log_info = teaRPM_for_tea_kd_stu_stuRPM_for_td_stu_kd_dreamer_first_train_tea_then_stu_mask4KdTd(
                teacher_rpm, student_rpm, qmix_agent, config, current_steps
            )
        else:
            raise NotImplementedError

        log_dict.update(**log_info)

    else:
        raise ValueError(f"No mode in train ONCE {config['train_mode']}")

    return log_dict


@torch.no_grad()
def run_evaluate_episode(
        env: MultiAgentEnv,
        agent: QMixTeachAgent,
        agent_type: str,
        eval_times: int = 3,
):
    agent.alg.agent_model_teacher.eval()
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []

    if agent_type not in ["teacher", "student", "rnd"]:
        raise ValueError

    for _ in range(eval_times):
        agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        state, obs = env.reset()

        while True:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions, agent_type)
            state, obs, reward, terminated, truncated = env.step(actions)
            episode_step += 1
            episode_reward += reward

            if terminated or truncated:
                break

        is_win = env.win_counted
        eval_reward_buffer.append(episode_reward)
        eval_steps_buffer.append(episode_step)
        eval_is_win_buffer.append(is_win)

    return (
        np.mean(eval_reward_buffer),
        np.mean(eval_steps_buffer),
        np.mean(eval_is_win_buffer),
    )


def eval_once(
        env: MultiAgentEnv,
        agent: QMixTeachAgent,
        eval_times: int = 32,
        config: dict = None,
):
    log_dict = {}
    agent.alg.eval()
    if config["train_mode"] == "train_tea" or config["train_mode"] == "train_tea_and_dreamer":
        eval_reward_teacher, eval_steps_teacher, eval_is_win_teacher = (
            run_evaluate_episode(env, agent, "teacher", eval_times)
        )
        log_dict.update(
            {
                "eval_reward_teacher": eval_reward_teacher,
                "eval_is_win_teacher": eval_is_win_teacher,
            }
        )
    elif (
            config["train_mode"] == "train_tea_kd_stu"
            or config["train_mode"] == "train_tea_kd_stu_rnd"
    ):
        eval_reward_teacher, eval_steps_teacher, eval_is_win_teacher = (
            run_evaluate_episode(env, agent, "teacher", eval_times)
        )
        eval_reward_student, eval_steps_student, eval_is_win_student = (
            run_evaluate_episode(env, agent, "student", eval_times)
        )
        log_dict.update(
            {
                "eval_reward_teacher": eval_reward_teacher,
                "eval_is_win_teacher": eval_is_win_teacher,
                "eval_reward_student": eval_reward_student,
                "eval_is_win_student": eval_is_win_student,
            }
        )
    elif (
            config["train_mode"] == "train_tea_kd_stu_train_stu"
            or config["train_mode"] == "train_tea_kd_stu_train_stu_rnd_TEST"
            or config["train_mode"] == "train_tea_kd_stu_train_stu_vq_vae"
            or config["train_mode"] == "train_tea_kd_stu_train_stu_dreamer_count"
            or config["train_mode"] == "pretrained_teacher_kd_td_stu_with_dreamer"
            or config["train_mode"] == "train_tea__train_kd_stu_with_RND"
    ):
        eval_reward_teacher, eval_steps_teacher, eval_is_win_teacher = (
            run_evaluate_episode(env, agent, "teacher", eval_times)
        )
        eval_reward_student, eval_steps_student, eval_is_win_student = (
            run_evaluate_episode(env, agent, "student", eval_times)
        )
        log_dict.update(
            {
                "eval_reward_teacher": eval_reward_teacher,
                "eval_steps_teacher": eval_steps_teacher,
                "eval_is_win_teacher": eval_is_win_teacher,
                "eval_reward_student": eval_reward_student,
                "eval_is_win_student": eval_is_win_student,
            }
        )
    elif config["train_mode"] == "train_tea_kd_stu_train_stu_rnd":
        eval_reward_teacher, eval_steps_teacher, eval_is_win_teacher = (
            run_evaluate_episode(env, agent, "teacher", eval_times)
        )
        eval_reward_student, eval_steps_student, eval_is_win_student = (
            run_evaluate_episode(env, agent, "student", eval_times)
        )
        eval_reward_rnd, eval_steps_student, eval_is_win_rnd = run_evaluate_episode(
            env, agent, "rnd", eval_times
        )
        log_dict.update(
            {
                "eval_reward_teacher": eval_reward_teacher,
                "eval_is_win_teacher": eval_is_win_teacher,
                "eval_reward_student": eval_reward_student,
                "eval_is_win_student": eval_is_win_student,
                "eval_reward_rnd": eval_reward_rnd,
                "eval_is_win_rnd": eval_is_win_rnd,
            }
        )
    else:
        raise ValueError(f"No mode {config['train_mode']}")

    return log_dict


def seed_everything(seed: int = 999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_current_run_dir(root_dir: str, run_name: str):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    dir_names = os.listdir(root_dir)
    run_names = [name for name in dir_names if name.startswith("run")]
    run_ids = [int(name.split("_")[1]) for name in run_names]
    my_id = max(run_ids) + 1 if len(run_ids) > 0 else 1

    my_dir_name = os.path.join(root_dir, f"run_{my_id}_{run_name}")
    os.makedirs(my_dir_name, exist_ok=True)

    return my_dir_name


def generate_teacher_baseline(env, agent, rpm, config: dict, episode_num: int) -> np.ndarray:
    agent.alg.eval()
    # ------------------------------------------------
    # 从一次游戏中收集一条完整的经验
    # --------------------------------------------------
    episode_limit = config["episode_limit"]
    for i in range(episode_num):
        agent.reset_agent()
        state, obs = env.reset()
        episode_reward = 0.0
        episode_step = 0
        episode_experience = EpisodeExperience(episode_limit)

        while True:
            available_actions = env.get_available_actions()
            actions = agent.sample_stu(obs, available_actions)
            next_state, next_obs, reward, terminated, truncated = env.step(actions)
            episode_reward += reward
            episode_step += 1
            episode_experience.add(
                state, actions, [reward], [terminated], obs, available_actions, [0]
            )
            state = next_state
            obs = next_obs

            if terminated or truncated:
                break

        # ===================================================================
        # 每条经验里面存储的步数信息是固定长度的，如果游戏提前结束，那么剩下的用零填充
        # ====================================================================
        state_zero = np.zeros_like(state, dtype=state.dtype)
        actions_zero = np.zeros_like(actions, dtype=actions.dtype)
        obs_zero = np.zeros_like(obs, dtype=obs.dtype)
        available_actions_zero = np.zeros_like(
            available_actions, dtype=available_actions.dtype
        )
        reward_zero = 0
        terminated_zero = True
        for _ in range(episode_step, episode_limit):
            episode_experience.add(
                state_zero,
                actions_zero,
                [reward_zero],
                [terminated_zero],
                obs_zero,
                available_actions_zero,
                [1],
            )
        rpm.add(episode_experience)

    stu_data = rpm.sample_batch(episode_num)
    agent.get_dreamer_count_factor(*stu_data, config["hash_type"])
    hash_list = deepcopy(agent.alg.dreamer_hash.hash)
    np_hash = np.array(hash_list, dtype=int)
    _range = np.sum(np_hash)
    normalized = np_hash / _range * config["tea_steps"]
    steps = normalized.astype(dtype=int)

    return steps


def check_git_repo() -> None:
    from git.repo import Repo
    repo = Repo(".")
    if repo.is_dirty():
        raise RuntimeError("GIT 仓库有未提交的文件，请提交后重新运行代码！")
    else:
        logger.info(f"GIT仓库的所有修改都已提交，程序正常运行")


def main(_paser: argparse.ArgumentParser):
    env_name = ""
    for index, name in enumerate(sys.argv):
        if name == "--env":
            env_name = sys.argv[index + 1]
            break

    if env_name == "":
        raise ValueError("No --env given")
    elif env_name == "pp":
        env = PredatorPreyEnvWarpper()
        env.init_args(_paser)
        args = _paser.parse_args()
        env.multi_agent_init(args)
    elif env_name == "smac":
        args = _paser.parse_args()
        env = REGISTRY["sc2"](
            map_name=args.scenario, difficulty=args.difficulty, sight=args.sight
        )
    elif env_name == "smac_v2":
        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }
        args = _paser.parse_args()
        env = REGISTRY["sc2_v2"](
            map_name=args.scenario, difficulty=args.difficulty, capability_config=distribution_config
        )
    elif env_name == "find_one_goal":
        args = _paser.parse_args()
        env = FindOneGoal(
            map_size=args.map_size,
            agent_num=args.n_agent,
            seed=args.seed,
            sight=args.sight,
            episode_limit=args.episode_max_steps,
        )
    else:
        raise ValueError

    config = vars(args)
    config["episode_limit"] = env.episode_limit
    config["obs_shape"] = env.obs_shape
    config["state_shape"] = env.state_shape
    config["n_agents"] = env.n_agents
    config["n_actions"] = env.n_actions

    if config["env"] == "smac":
        run_name = f'{config["agent_type"]}_{config["scenario"]}_seed_{config["seed"]}_sight_{env.get_sight()}'
    elif config["env"] == "pp":
        run_name = f'{config["agent_type"]}_pp_seed_{config["seed"]}'
    elif config["env"] == "find_one_goal":
        run_name = f'{config["agent_type"]}_FindOneGoal_seed_{config["seed"]}_sight_{env.get_sight()}'
    elif config["env"] == "smac_v2_10_8":
        run_name = f'smac_v2_new_{config["agent_type"]}_{config["scenario"]}_seed_{config["seed"]}'
    else:
        raise ValueError

    seed_everything(config["seed"])

    current_run_dir = generate_current_run_dir("runs", run_name)
    config["current_run_dir"] = current_run_dir
    logger.info(f"本次实验日志记录在{current_run_dir}，预训练模型位置为{config['restored_weight_dir']}")

    teacher_rpm = EpisodeReplayBuffer(config["replay_buffer_size"])
    student_rpm = EpisodeReplayBuffer(config["replay_buffer_size"])
    agent_model_student = RNNModel(
        config["obs_shape"], config["n_actions"], config["rnn_hidden_dim"]
    )

    if config["agent_type"] == "trans_rnn":
        agent_model_teacher = RNNTransModel(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    elif config["agent_type"] == "rnn":
        agent_model_teacher = RNNModel(
            config["obs_shape"], config["n_actions"], config["rnn_hidden_dim"]
        )
    elif config["agent_type"] == "self_attn_rnn":
        agent_model_teacher = RNNSelfAttnModel(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    elif config["agent_type"] == "self_linear_attn_rnn":
        agent_model_teacher = RNNLinearSelfAttnModel(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    elif config["agent_type"] == "self_linear_fc3_attn_rnn":
        agent_model_teacher = RNNLinearFC3SelfAttnModel(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    elif config["agent_type"] == "multi_head_attn_rnn":
        agent_model_teacher = RNNMultiHeadAttnModel(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    elif config["agent_type"] == "cadp_attn_rnn":
        agent_model_teacher = CADPATTRNNAgent(
            config["obs_shape"],
            argparse.Namespace(**config),
        )
    elif config["agent_type"] == "att_rnn":
        agent_model_teacher = ATTRNNAgent(
            config["obs_shape"],
            config["n_actions"],
            config["rnn_hidden_dim"],
            config["attention_heads"],
        )
    else:
        raise ValueError(f"没有这个agent类型{config['agent_type']}")

    if config["mixer_type"] == "qmix":

        mixer_model_teacher = QMixerModel(
            config["n_agents"],
            config["state_shape"],
            config["mixing_embed_dim"],
            config["hypernet_layers"],
            config["hypernet_embed_dim"],
        )

        mixer_model_student = QMixerModel(
            config["n_agents"],
            config["state_shape"],
            config["mixing_embed_dim"],
            config["hypernet_layers"],
            config["hypernet_embed_dim"],
        )

    elif config["mixer_type"] == "vdn":

        mixer_model_teacher = VDNMixer()
        mixer_model_student = VDNMixer()
    
    elif config["mixer_type"] == "nqmix":

        mixer_model_teacher = NQMixerModel(
            config["n_agents"],
            config["state_shape"],
            config["mixing_embed_dim"],
            config["hypernet_layers"],
            config["hypernet_embed_dim"],
        )

        mixer_model_student = NQMixerModel(
            config["n_agents"],
            config["state_shape"],
            config["mixing_embed_dim"],
            config["hypernet_layers"],
            config["hypernet_embed_dim"],
        )

    elif config["mixer_type"] == "qatten":
        mixer_model_teacher = QattenMixer(
            config["n_agents"],
            config["state_shape"],
        )
        mixer_model_student = QattenMixer(
            config["n_agents"],
            config["state_shape"],
        )

    else:
        raise NotImplementedError(f"没这个Mixer{config['mixer_type']}")



    dreamer_model = RNNDreamerModel(input_shape=config["n_agents"] * config["obs_shape"], n_actions=config["n_actions"],
                                    rnn_hidden_dim=config["rnn_hidden_dim"], n_categorical=config["n_categorical"],
                                    n_classes=config["n_classes"])

    vq_vae_model = LinearVQVAE(
        in_features=config["state_shape"],
        embedding_dim=config["vq_vae_embedding_dim"],
        num_embeddings=config["vq_vae_num_embedding"],
        hidden_dims=[64, 32],
    )

    algorithm = QMIXTeach(
        agent_model_student,
        agent_model_teacher,
        mixer_model_teacher,
        mixer_model_student,
        dreamer_model,
        vq_vae_model,
        config["double_q"],
        config["gamma"],
        config["lr"],
        config["clip_grad_norm"],
        config,
    )

    qmix_agent = QMixTeachAgent(
        algorithm,
        config["exploration_start"],
        config["min_exploration"],
        config["exploration_decay"],
        config["update_target_interval"],
        config,
    )

    # 加载预训练模型
    if config["train_mode"] == "pretrained_teacher_kd_td_stu_with_dreamer":
        res = qmix_agent.load_teacher_and_dreamer(config["restored_weight_dir"], 'last')
        if not res:
            raise RuntimeError(
                f"在pretrained_teacher_kd_td_stu_with_dreamer训练模式下，未能成功加载预训练模型，程序退出！！")

        if config["sub_train_mode"] == "use_stored_mask_area":
            df = pd.read_csv(os.path.join(config["restored_weight_dir"], 'log_detail.csv'), index_col=0)
            config['log_detail'] = df
            logger.info(f"成功加载log信息！！")

    while teacher_rpm.count < config["memory_warmup_size"]:
        collect_rollout(
            env,
            qmix_agent,
            student_rpm=student_rpm,
            teacher_rpm=teacher_rpm,
            config=config,
        )

    # 为每个环境配置对应的swanlab project
    if config["env"] == "smac":
        prefix = config["scenario"]
    elif config["env"] == "find_one_goal":
        prefix = f"{config['n_agent']}_agent"
    else:
        raise NotImplementedError

    run = swanlab.init(
        project=config["env"] + "-MACD",  #_" + prefix,
        config=config,
        experiment_name=run_name,
        mode=config["swanlab_mode"],
    )

    kd_count = 0
    total_steps = 0
    last_test_step = -1e10
    best_teacher_win_rate = -1.0

    if (config["train_mode"] == "pretrained_teacher_kd_td_stu_with_dreamer" and
            config["sub_train_mode"] == "teacher_baseline"):
        hash_rpm = EpisodeReplayBuffer(config["replay_buffer_size"])
        episode_num = 100
        hash_baseline = generate_teacher_baseline(env, qmix_agent, hash_rpm, config, episode_num)
        qmix_agent.alg.dreamer_hash.teacher_baseline_hash = hash_baseline.tolist()
        logger.info(f"Hash Baseline Info: {hash_baseline.max()} {hash_baseline.min()} {hash_baseline.mean()}"
                    f" {len(hash_baseline)}")

    while total_steps < config["training_steps"]:
        one_episode_steps = collect_rollout(
            env,
            qmix_agent,
            student_rpm=student_rpm,
            teacher_rpm=teacher_rpm,
            config=config,
        )
        log_dict = train_once(
            teacher_rpm,
            student_rpm,
            qmix_agent,
            kd_times=1,
            config=config,
            current_steps=total_steps,
        )
        kd_count += 1
        total_steps += one_episode_steps
        run.log(log_dict, step=total_steps)
        run.log(
            {
                "train_step_in_an_episode": one_episode_steps,
                "exploration": qmix_agent.exploration,
                "replay_buffer_size": teacher_rpm.count,
                "target_update_count": qmix_agent.target_update_count,
            },
            step=total_steps,
        )

        # 进行eval测试环境
        if total_steps - last_test_step > config["test_steps"]:
            last_test_step = total_steps
            eval_log_dict = eval_once(env, qmix_agent, config=config)
            run.log(eval_log_dict, step=total_steps)

            logger_dict = dict(steps=total_steps, **eval_log_dict)
            win_rate = logger_dict.get("eval_is_win_teacher")
            if win_rate > best_teacher_win_rate:
                qmix_agent.save_best_teacher(config["current_run_dir"], win_rate, total_steps)
                best_teacher_win_rate = win_rate

            qmix_agent.save_last_teacher(config["current_run_dir"], win_rate, total_steps)

            logger.info(logger_dict)

            if config['train_mode'] == 'pretrained_teacher_kd_td_stu_with_dreamer' and config['sub_train_mode'] == 'default':
                path = os.path.join(config['current_run_dir'], f'step_{total_steps}.npy')
                qmix_agent.alg.dreamer_hash.to_npy(path)
                logger.info(f"成功保存hash热力图到{path}")

    run.finish()
    logger.info("swanlab finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_tea_kd_stu.py",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Name of the map",
        default="MMM2",
    )
    parser.add_argument("--replay_buffer_size", "-r", type=int, default=2000)
    parser.add_argument("--mixing_embed_dim", "-m", type=int, default=32)
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--memory_warmup_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration_start", type=float, default=1.0)
    parser.add_argument("--min_exploration", type=float, default=0.05)
    parser.add_argument("--exploration_decay", type=float, default=2e-6)
    parser.add_argument("--update_target_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--training_steps", type=int, default=300_0000)
    parser.add_argument("--test_steps", type=int, default=2e5)
    parser.add_argument("--clip_grad_norm", type=int, default=10)
    parser.add_argument("--hypernet_layers", type=int, default=2)
    parser.add_argument("--hypernet_embed_dim", type=int, default=64)
    parser.add_argument("--double_q", type=bool, default=True)
    parser.add_argument("--difficulty", type=str, default="7")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--swanlab_mode", type=str, default="online"
    )
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--student_self_train_interval", type=int, default=1)
    parser.add_argument("--relationship_nums", type=int, default=4)
    parser.add_argument("--tar_hidden_dim", type=int, default=256)

    parser.add_argument("--agent_type",type=str,default="rnn")
    parser.add_argument("--mixer_type",type=str,default="qmix")
    parser.add_argument("--env", type=str, choices=["smac", "pp", "find_one_goal", "smac_v2"], default="smac")
    """
        find one goal 环境参数设置
    """
    parser.add_argument("--map_size", type=int, default=10)
    parser.add_argument("--n_agent", type=int, default=2)
    parser.add_argument("--episode_max_steps", type=int, default=120)
    parser.add_argument("--sight", type=int, choices=list(range(1, 10)), default=1)

    """
        训练方式设置
    """
    parser.add_argument("--train_mode", type=str, default="train_tea")
    parser.add_argument("--kd_freq", type=float, default=1)
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
    )
    parser.add_argument("--sub_train_mode", type=str, default="default")
    parser.add_argument(
        "--vq_vae_embedding_dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--vq_vae_num_embedding",
        type=int,
        default=512,
    )
    parser.add_argument("--n_categorical", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--restored_weight_dir", type=str, default="saved_model/fog_teacher_dreamer_4_8")
    parser.add_argument("--mask_area_factor", type=float, default=1.0)
    parser.add_argument("--hash_type", type=str, default="1/n*100")  # cut_off_at_1000 cut_off_at_100
    parser.add_argument("--tea_steps", type=int, default=50_0000)  # cut_off_at_1000 cut_off_at_100

    # check_git_repo()
    main(parser)
