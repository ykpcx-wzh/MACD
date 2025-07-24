import pandas as pd
import wandb
import os

def download_fog():
    api = wandb.Api()
    entity, project = "lioner-kiss", "find_one_goal_10_11_4_agent"
    runs = api.runs(entity + "/" + project)
    run_num_dict = {}
    for index, run in enumerate(runs):
        mixer_type = 'qmix'
        n_agents = run.config["n_agents"]
        train_mode = run.config["train_mode"]
        sub_train_mode = run.config["sub_train_mode"]
        path = f'Find_One_Goal/{mixer_type}/agents_{n_agents}/{train_mode}/{sub_train_mode}'
        if path not in run_num_dict.keys():
            run_num_dict[path] = 0
        else:
            run_num_dict[path] += 1
        os.makedirs(path, exist_ok=True)
        history = run.scan_history(page_size=100_000)
        all_data = {}
        for row in history:
            for key in row.keys():
                if key not in all_data.keys():
                    all_data[key] = [row[key]]
                else:
                    all_data[key].append(row[key])
    
        pd_data = pd.DataFrame(all_data)
        pd_data.to_csv(path + f"/seed_{run_num_dict[path]}.csv", index=False)
        print(f'完成了第{index}个Run')

def download_smac():
    api = wandb.Api()
    entity, project = "lioner-kiss", "smac_6h_vs_8z"
    runs = api.runs(entity + "/" + project)
    run_num_dict = {}
    for index, run in enumerate(runs):
        map_name = run.config['scenario']
        mixer_type = run.config["mixer_type"]
        path = f'SMAC/{mixer_type}/{map_name}'
        if path not in run_num_dict.keys():
            run_num_dict[path] = 0
        else:
            run_num_dict[path] += 1
        os.makedirs(path, exist_ok=True)
        history = run.scan_history(page_size=100_000)
        all_data = {}
        for row in history:
            for key in row.keys():
                if key not in all_data.keys():
                    all_data[key] = [row[key]]
                else:
                    all_data[key].append(row[key])

        pd_data = pd.DataFrame(all_data)
        pd_data.to_csv(path + f"/seed_{run_num_dict[path]}.csv", index=False)
        print(f'完成了第{index}个Run')

if __name__ == "__main__":
    download_fog()
