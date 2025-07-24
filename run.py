import os
from multiprocessing import Pool
import random

smac_avail_maps = [
    "3m",
    "8m",
    "25m",
    "5m_vs_6m",
    "8m_vs_9m",
    "10m_vs_11m",
    "27m_vs_30m",
    "MMM",
    "MMM2",
    "2s3z",
    "3s5z",
    "3s5z_vs_3s6z",
    "3s_vs_3z",
    "3s_vs_4z",
    "3s_vs_5z",
    "1c3s5z",
    "2m_vs_1z",
    "corridor",
    "6h_vs_8z",
    "2s_vs_1sc",
    "so_many_baneling",
    "bane_vs_bane",
    "2c_vs_64zg",
]


def fog_cmds_exp_7_12_4():
    """
    qmix
    """
    seeds = range(3)
    map_size = 15
    n_agents = [4]
    sight = 2
    algos = ["self_attn_rnn"]
    mixers  = ["qmix"]
    train_modes = ["pretrained_teacher_kd_td_stu_with_dreamer"]
    swanlab_mode = "cloud" 
    kd_freq = [1]
    batch_size = 128
    training_steps = 155_0000
    episode_max_steps = 100

    exp_name = "about_hash"
    sub_train_mode = "default"
    restored_weight_dirs = ["saved_model/fog_4_agent/qmix/run_61_self_attn_rnn_FindOneGoal_seed_1_sight_2",
                            "saved_model/fog_4_agent/qmix/run_61_self_attn_rnn_FindOneGoal_seed_1_sight_2"]
    hash_types = ["1/n*100"]

    factors = [1]

    _cmds = []
    for seed in seeds:
        for n_agent in n_agents:
            for algo in algos:
                for mixer in mixers:
                    if algo == "rnn":
                        cmd = (
                            f"python train_tea_kd_stu.py --env find_one_goal --seed {seed} --agent_type {algo} "
                            f"--map_size {map_size} --n_agent {n_agent} --train_mode train_tea --swanlab_mode {swanlab_mode} "
                            f"--batch_size {batch_size} --training_steps {training_steps} --exp_name {exp_name} "
                            f"--episode_max_steps {episode_max_steps} --sight {sight} --mixer_type {mixer}"
                        )
                        _cmds.append(cmd)
                else:
                    for freq in kd_freq:
                        for train_mode in train_modes:
                            for weight in restored_weight_dirs:
                                for factor in factors:
                                    for hash_type in hash_types:
                                        for mixer in mixers:
                                            cmd = (
                                                f"python train_tea_kd_stu.py --env find_one_goal --seed {seed} --agent_type {algo} "
                                                f"--map_size {map_size} --n_agent {n_agent} --train_mode {train_mode} "
                                                f"--swanlab_mode {swanlab_mode} --kd_freq {freq} --batch_size {batch_size} "
                                                f"--training_steps {training_steps} --episode_max_steps {episode_max_steps} "
                                                f"--exp_name {exp_name} --sub_train_mode {sub_train_mode} "
                                                f"--restored_weight_dir {weight} --mask_area_factor {factor} "
                                                f"--hash_type {hash_type} --sight {sight} --mixer_type {mixer}"
                                            )
                                            _cmds.append(cmd)

    return _cmds


if __name__ == "__main__":
    paral_num = 3
    cmds =  fog_cmds_exp_7_12_4()
    random.shuffle(cmds)
    print(f"will execute {len(cmds)} commands:", *cmds, sep="\n")
    pool = Pool(processes=paral_num)
    pool.map_async(os.system, cmds)
    pool.close()
    pool.join()
