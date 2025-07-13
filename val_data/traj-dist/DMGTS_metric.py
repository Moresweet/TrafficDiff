import timeit
import collections
import pandas as pd
import numpy as np

import pickle
import traj_dist.distance as tdist
from traj_dist.pydist.linecell import trajectory_set_grid
import timeit
import collections
import pandas as pd
import numpy as np
import torch


def filter_data(data):
    Q1 = np.percentile(data, 25)  # 第一四分位数
    Q3 = np.percentile(data, 75)  # 第三四分位数
    IQR = Q3 - Q1
    # 定义阈值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

# traj_list = pickle.load(open("./data/benchmark_trajectories.pkl", "rb"), encoding='latin1')[:100]
traj_sac_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/SAC_traj.pt')
traj_sac_list = [i.numpy().astype(np.float64) for i in traj_sac_list]
label_traj = torch.load('/home/moresweet/gitCloneZone/DMGTS/visualization/new/future.pt')
label_traj = [i.cpu().numpy().astype(np.float64) for i in label_traj][:len(traj_sac_list)]
traj_ddpg_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/DDPG_traj.pt')
traj_ddpg_list = [i.numpy().astype(np.float64) for i in traj_ddpg_list]
traj_ppo_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/PPO_traj.pt')
traj_ppo_list = [i.numpy().astype(np.float64) for i in traj_ppo_list]
traj_td3_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/TD3_traj.pt')
traj_td3_list = [i.numpy().astype(np.float64) for i in traj_td3_list]
traj_sac_dmgts_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/SAC_traj_dmgts.pt')
traj_sac_dmgts_list = [i.numpy().astype(np.float64) for i in traj_sac_dmgts_list]
traj_ppo_dmgts_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/PPO_traj_dmgts.pt')
traj_ppo_dmgts_list = [i.numpy().astype(np.float64) for i in traj_ppo_dmgts_list]
traj_ddpg_dmgts_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/DDPG_traj_dmgts.pt')
traj_ddpg_dmgts_list = [i.numpy().astype(np.float64) for i in traj_ddpg_dmgts_list]
traj_td3_dmgts_list = torch.load('/home/moresweet/gitCloneZone/carlaSimBench/data/TD3_traj_dmgts.pt')
traj_td3_dmgts_list = [i.numpy().astype(np.float64) for i in traj_td3_dmgts_list]
sspd_sac_record = []
dwt_sac_record = []
frechet_sac_record = []
sspd_ddpg_record = []
dwt_ddpg_record = []
frechet_ddpg_record = []
sspd_td3_record = []
dwt_td3_record = []
frechet_td3_record = []
sspd_ppo_record = []
dwt_ppo_record = []
frechet_ppo_record = []
sspd_sac_dmgts_record = []
dwt_sac_dmgts_record = []
frechet_sac_dmgts_record = []
sspd_ddpg_dmgts_record = []
dwt_ddpg_dmgts_record = []
frechet_ddpg_dmgts_record = []
sspd_td3_dmgts_record = []
dwt_td3_dmgts_record = []
frechet_td3_dmgts_record = []
sspd_ppo_dmgts_record = []
dwt_ppo_dmgts_record = []
frechet_ppo_dmgts_record = []
for idx, traj in enumerate(traj_sac_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_sac_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_sac_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_sac_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))
sspd_sac_record = filter_data(sspd_sac_record)
dwt_sac_record = filter_data(dwt_sac_record)
frechet_sac_record = filter_data(frechet_sac_record)
sspd_sac = np.mean(sspd_sac_record)
dwt_sac = np.mean(dwt_sac_record)
frechet_sac = np.mean(frechet_sac_record)

for idx, traj in enumerate(traj_ddpg_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_ddpg_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_ddpg_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_ddpg_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_ddpg_record = filter_data(sspd_ddpg_record)
dwt_ddpg_record = filter_data(dwt_ddpg_record)
frechet_ddpg_record = filter_data(frechet_ddpg_record)

sspd_ddpg = np.mean(sspd_ddpg_record)
dwt_ddpg = np.mean(dwt_ddpg_record)
frechet_ddpg = np.mean(frechet_ddpg_record)

for idx, traj in enumerate(traj_td3_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_td3_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_td3_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_td3_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_td3_record = filter_data(sspd_td3_record)
dwt_td3_record = filter_data(dwt_td3_record)
frechet_td3_record = filter_data(frechet_td3_record)

sspd_td3 = np.mean(sspd_td3_record)
dwt_td3 = np.mean(dwt_td3_record)
frechet_td3 = np.mean(frechet_td3_record)

for idx, traj in enumerate(traj_ppo_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_ppo_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_ppo_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_ppo_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_ppo_record = filter_data(sspd_ppo_record)
dwt_ppo_record = filter_data(dwt_ppo_record)
frechet_ppo_record = filter_data(frechet_ppo_record)

sspd_ppo = np.mean(sspd_ppo_record)
dwt_ppo = np.mean(dwt_ppo_record)
frechet_ppo = np.mean(frechet_ppo_record)

for idx, traj in enumerate(traj_ddpg_dmgts_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_ddpg_dmgts_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_ddpg_dmgts_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_ddpg_dmgts_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_ddpg_dmgts_record = filter_data(sspd_ddpg_dmgts_record)
dwt_ddpg_dmgts_record = filter_data(dwt_ddpg_dmgts_record)
frechet_ddpg_dmgts_record = filter_data(frechet_ddpg_dmgts_record)
sspd_ddpg_dmgts = np.mean(sspd_ddpg_dmgts_record)
dwt_ddpg_dmgts = np.mean(dwt_ddpg_dmgts_record)
frechet_ddpg_dmgts = np.mean(frechet_ddpg_dmgts_record)

for idx, traj in enumerate(traj_sac_dmgts_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_sac_dmgts_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_sac_dmgts_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_sac_dmgts_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_sac_dmgts_record = filter_data(sspd_sac_dmgts_record)
dwt_sac_dmgts_record = filter_data(dwt_sac_dmgts_record)
frechet_sac_dmgts_record = filter_data(frechet_sac_dmgts_record)
sspd_sac_dmgts = np.mean(sspd_sac_dmgts_record)
dwt_sac_dmgts = np.mean(dwt_sac_dmgts_record)
frechet_sac_dmgts = np.mean(frechet_sac_dmgts_record)

for idx, traj in enumerate(traj_td3_dmgts_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_td3_dmgts_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_td3_dmgts_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_td3_dmgts_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_td3_dmgts_record = filter_data(sspd_td3_dmgts_record)
dwt_td3_dmgts_record = filter_data(dwt_td3_dmgts_record)
frechet_td3_dmgts_record = filter_data(frechet_td3_dmgts_record)
sspd_td3_dmgts = np.mean(sspd_td3_dmgts_record)
dwt_td3_dmgts = np.mean(dwt_td3_dmgts_record)
frechet_td3_dmgts = np.mean(frechet_td3_dmgts_record)

for idx, traj in enumerate(traj_ppo_dmgts_list):
    valid_length = np.argmax(traj[:,0][::-1] != 0)
    if valid_length == 24:
        continue
    traj = traj[:24 - valid_length]
    sspd_ppo_dmgts_record.append(tdist.sspd(traj, label_traj[idx][:24 - valid_length]))
    dwt_ppo_dmgts_record.append(tdist.dtw(traj, label_traj[idx][:24-valid_length]))
    frechet_ppo_dmgts_record.append(tdist.frechet(traj, label_traj[idx][:24-valid_length]))

sspd_ppo_dmgts_record = filter_data(sspd_ppo_dmgts_record)
dwt_ppo_dmgts_record = filter_data(dwt_ppo_dmgts_record)
frechet_ppo_dmgts_record = filter_data(frechet_ppo_dmgts_record)
sspd_ppo_dmgts = np.mean(sspd_ppo_dmgts_record)
dwt_ppo_dmgts = np.mean(dwt_ppo_dmgts_record)
frechet_ppo_dmgts = np.mean(frechet_ppo_dmgts_record)

print_data = {'Algorithm': ['DDPG', 'SAC', 'PPO', 'TD3', 'DDPG_DMGTS', 'SAC_DMGTS', 'PPO_DMGTS', 'TD3_DMGTS'],
    'SSPD': [sspd_ddpg, sspd_sac, sspd_ppo, sspd_td3, sspd_ddpg_dmgts, sspd_sac_dmgts, sspd_ppo_dmgts, sspd_td3_dmgts],
    'Frechet': [frechet_ddpg, frechet_sac, frechet_ppo, frechet_td3, frechet_ddpg_dmgts, frechet_sac_dmgts, frechet_ppo_dmgts, frechet_td3_dmgts],
    'DTW': [dwt_ddpg, dwt_sac, dwt_ppo, dwt_td3, dwt_ddpg_dmgts, dwt_sac_dmgts, dwt_ppo_dmgts, dwt_td3_dmgts]}

df = pd.DataFrame(print_data)
print(df)

# time_dict = collections.defaultdict(dict)



