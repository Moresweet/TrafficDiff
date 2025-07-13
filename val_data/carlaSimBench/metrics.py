import torch
from scipy.stats import wasserstein_distance, multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_average_speeds(nbrs_count, sac_traj_vel_dmgts, ddpg_traj_vel_dmgts, ppo_traj_vel_dmgts, td3_traj_vel_dmgts,
                        mode):
    # 计算每个场景的平均速度
    def calculate_average_velocity(trajectories):
        return torch.stack([torch.mean(trajectory, dim=0) for trajectory in trajectories])

    sac_avg_vel = calculate_average_velocity(sac_traj_vel_dmgts)
    ddpg_avg_vel = calculate_average_velocity(ddpg_traj_vel_dmgts)
    ppo_avg_vel = calculate_average_velocity(ppo_traj_vel_dmgts)
    td3_avg_vel = calculate_average_velocity(td3_traj_vel_dmgts)

    def filter_and_interpolate(avg_velocities):
        valid_speeds = []
        for vel in avg_velocities:
            speed = torch.norm(vel).item()
            if mode == 'Speed':
                if speed >= 0.3:
                    valid_speeds.append(speed)
                else:
                    valid_speeds.append(None)  # 标记为无效
            else:
                if 3.14 >= speed >= -3.14:
                    valid_speeds.append(abs(speed))
                else:
                    valid_speeds.append(None)  # 标记为无效

        # 处理无效值
        for i in range(len(valid_speeds)):
            if valid_speeds[i] is None:
                if i > 0 and valid_speeds[i - 1] is not None:
                    valid_speeds[i] = valid_speeds[i - 1]  # 使用之前的有效值插入

        return valid_speeds

    # 对每种算法过滤和插值
    sac_valid_speeds = filter_and_interpolate(sac_avg_vel)
    ddpg_valid_speeds = filter_and_interpolate(ddpg_avg_vel)
    ppo_valid_speeds = filter_and_interpolate(ppo_avg_vel)
    td3_valid_speeds = filter_and_interpolate(td3_avg_vel)

    # 计算平均速度和分组
    def group_average_speed(nbrs_count, valid_speeds):
        grouped_avg_speed = {}
        for count, speed in zip(nbrs_count, valid_speeds):
            if speed is not None:  # 只考虑有效速度
                if count not in grouped_avg_speed:
                    grouped_avg_speed[count] = []
                grouped_avg_speed[count].append(speed)

        # 计算每组的平均速度和误差
        group_means = []
        group_max = []
        group_min = []
        groups = sorted(grouped_avg_speed.keys())

        for group in groups:
            speeds = grouped_avg_speed[group]
            group_means.append(np.mean(speeds))
            group_max.append(np.max(speeds))
            group_min.append(np.min(speeds))

        return groups, group_means, group_max, group_min

    # 对每种算法进行分组计算
    sac_groups, sac_means, sac_max, sac_min = group_average_speed(nbrs_count, sac_valid_speeds)
    ddpg_groups, ddpg_means, ddpg_max, ddpg_min = group_average_speed(nbrs_count, ddpg_valid_speeds)
    ppo_groups, ppo_means, ppo_max, ppo_min = group_average_speed(nbrs_count, ppo_valid_speeds)
    td3_groups, td3_means, td3_max, td3_min = group_average_speed(nbrs_count, td3_valid_speeds)

    # 绘制直方图
    def plot_average_speed(groups, means, maxs, mins, label, subplot_position):
        plt.subplot(2, 2, subplot_position)
        plt.errorbar(groups, means, yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
                     fmt='o', capsize=5)
        plt.xticks(groups)
        if mode == 'Speed':
            plt.xlabel('Number of Vehicles', fontsize=17, fontfamily='Times New Roman')
            plt.ylabel('Average Speed', fontsize=17, fontfamily='Times New Roman')
            plt.title(f'Average Speed ({label})', fontsize=17, fontfamily='Times New Roman')
        else:
            plt.xlabel('Number of Vehicles')
            plt.ylabel('Average Steer')
            plt.title(f'Average Steer ({label})', fontsize=17, fontfamily='Times New Roman')
        plt.grid(True)

    # 创建四张图
    plt.figure(figsize=(10, 6))

    # SAC 图
    plot_average_speed(sac_groups, sac_means, sac_max, sac_min, 'SAC', 1)

    # DDPG 图
    plot_average_speed(ddpg_groups, ddpg_means, ddpg_max, ddpg_min, 'DDPG', 2)

    # PPO 图
    plot_average_speed(ppo_groups, ppo_means, ppo_max, ppo_min, 'PPO', 3)

    # TD3 图
    plot_average_speed(td3_groups, td3_means, td3_max, td3_min, 'TD3', 4)

    plt.tight_layout()
    plt.savefig('./'+mode + '_metrics.png', dpi=600, bbox_inches='tight')
    plt.show()


def calculate_acceleration(label_traj):
    label_traj = label_traj.cpu()
    # 提取 x 和 y 坐标
    x = label_traj[:, :, 0]
    y = label_traj[:, :, 1]

    # 计算速度 v_x 和 v_y
    v_x = np.gradient(x, 0.2, axis=1)  # 对时间轴（第二维）进行差分
    v_y = np.gradient(y, 0.2, axis=1)

    # 计算加速度 acc_x 和 acc_y
    acc_x = np.gradient(v_x, 0.2, axis=1)
    acc_y = np.gradient(v_y, 0.2, axis=1)

    # 将加速度合并为一个数组，shape 为 (B, 24, 2)
    acceleration = np.stack((acc_x, acc_y), axis=-1)

    return acceleration


def kl_divergence_gaussian(mean1, cov1, mean2, cov2):
    """
    计算两个多元高斯分布之间的 KL 散度
    mean1, mean2: 均值向量
    cov1, cov2: 协方差矩阵
    """
    # 计算协方差矩阵的行列式和逆
    cov2_inv = np.linalg.inv(cov2)
    log_det_cov1 = np.log(np.linalg.det(cov1))
    log_det_cov2 = np.log(np.linalg.det(cov2))

    # KL 散度公式的各项计算
    trace_term = np.trace(np.dot(cov2_inv, cov1))
    mean_diff = mean2 - mean1
    mean_term = np.dot(np.dot(mean_diff.T, cov2_inv), mean_diff)

    # KL 散度计算
    kl = 0.5 * (log_det_cov2 - log_det_cov1 - len(mean1) + trace_term + mean_term)

    return kl


def calculate_wd_and_kl(trajectory_list, label_acc):
    wd_scores = []
    kl_scores = []

    for idx, trajectory in enumerate(trajectory_list):
        try:
            trajectory_np = trajectory.cpu().numpy()
            # 提取 acc_x 和 acc_y
            acc_x = trajectory_np[:, 0]
            acc_y = trajectory_np[:, 1]
            valid_length = np.argmax(acc_x[::-1] != 0)  # 从后往前找到第一个非零项
            if valid_length == 24:
                continue
            acc_x = acc_x[:len(acc_x) - valid_length]
            acc_y = acc_y[:len(acc_y) - valid_length]
            label_acc_x = label_acc[idx, :24 - valid_length, 0]
            label_acc_y = label_acc[idx, :24 - valid_length, 1]
            # --- Wasserstein Distance (WD) ---
            # 分别计算 acc_x 和 acc_y 的 WD
            wd_x = wasserstein_distance(acc_x, label_acc_x)  # 示例中假设与自己比较
            wd_y = wasserstein_distance(acc_y, label_acc_y)  # 示例中假设与自己比较
            # 最终 WD 结果取平均
            wd_avg = (wd_x + wd_y) / 2
            wd_scores.append(wd_avg)

            # --- Kullback-Leibler Divergence (KL) ---
            # 假设 acc_x 和 acc_y 的分布是多元高斯分布
            # 计算均值和协方差
            mean_1 = np.mean(trajectory_np, axis=0)
            cov_1 = np.cov(trajectory_np.T)

            # 假设第二个轨迹分布与第一个分布对比 (这里你可以提供另一组数据)
            mean_2 = np.mean(label_acc[idx], axis=0)  # 假设同样的分布
            cov_2 = np.cov(label_acc[idx].T)

            # 计算两个多元高斯分布之间的KL散度
            # kl_div = multivariate_normal(mean_1, cov_1).kl_divergence(multivariate_normal(mean_2, cov_2))
            kl_div = kl_divergence_gaussian(mean_1, cov_1, mean_2, cov_2)
            kl_scores.append(kl_div)
        except:
            continue

    return wd_scores, kl_scores


def compute_incomplete_route(perd, labels, completion_threshold=1.0):
    incomplete_count = 0
    total_scenarios = len(perd)

    for pred_traj, true_traj in zip(perd, labels):
        # 筛选出有效的预测轨迹点（排除 (0,0) 的无效点）
        valid_mask = (pred_traj[:, 0] != 0) | (pred_traj[:, 1] != 0)
        pred_xy = pred_traj[valid_mask, :2]  # 只取有效轨迹 (x, y)
        true_xy = true_traj[:len(pred_xy)]  # 截取对应长度的真实轨迹

        # 计算有效轨迹的终点和真实轨迹终点之间的距离
        pred_end = pred_xy[-1]
        true_end = true_traj[-1]
        end_distance = torch.norm(pred_end - true_end)

        if end_distance > completion_threshold:
            incomplete_count += 1

    incomplete_route_rate = incomplete_count / total_scenarios * 100
    return incomplete_route_rate


def process_tensor_list(tensor_list):
    # 筛选每个 tensor 中包含 0 和大于 30 的元素
    filtered_tensors = [tensor[(tensor != 0) & (tensor <= 30)] for tensor in tensor_list]

    # 筛选后长度不为 24 的 tensor 删去
    filtered_tensors = [tensor for tensor in filtered_tensors if tensor.size(0) == 24]

    if not filtered_tensors:
        return torch.tensor([])  # 如果没有满足条件的 tensor，返回空 tensor

    # 按照每个步长取平均
    stacked_tensors = torch.stack(filtered_tensors)
    result = torch.mean(stacked_tensors, dim=0)

    return result, filtered_tensors


def compute_speed_satisfaction(perd, labels, speed_tolerance=2.0):
    satisfied_count = 0
    total_time_steps = 0

    for pred_traj, true_traj in zip(perd, labels):
        # 筛选出有效的预测轨迹点（排除 (0,0) 的无效点）
        valid_mask = (pred_traj[:, 0] != 0) | (pred_traj[:, 1] != 0)
        pred_traj_valid = pred_traj[valid_mask]  # 有效预测轨迹
        true_traj_valid = true_traj.cpu()[:len(pred_traj_valid)]  # 对应的有效真实轨迹

        # 计算预测和真实的速度
        pred_v_x, pred_v_y = pred_traj_valid[:, 0], pred_traj_valid[:, 1]
        true_v_x = torch.diff(true_traj_valid[:, 0], dim=0) / 0.2
        true_v_y = torch.diff(true_traj_valid[:, 1], dim=0) / 0.2

        pred_speeds = torch.sqrt(pred_v_x ** 2 + pred_v_y ** 2)  # 预测速度
        true_speeds = torch.sqrt(true_v_x ** 2 + true_v_y ** 2)  # 真实速度

        if len(pred_speeds) != len(true_speeds):
            pred_speeds = pred_speeds[:len(true_speeds)]
        # 判断速度偏差是否在容差范围内
        speed_diff = torch.abs(pred_speeds - true_speeds)
        satisfied = speed_diff <= speed_tolerance
        satisfied_count += torch.sum(satisfied).item()
        total_time_steps += len(satisfied)

    speed_satisfaction = satisfied_count / total_time_steps
    return speed_satisfaction


# 假设每个 filtered_list 的结构为 [torch.Tensor(24,), ...]，长度相同且已被处理
def plot_ttc_curves(algorithm_names, avg_ttc_list, filtered_ttc_list):
    plt.figure(figsize=(5, 3))
    sns.set_theme(style="darkgrid")
    # 设置颜色
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']

    for i, (name, avg_ttc, filtered_list) in enumerate(zip(algorithm_names, avg_ttc_list, filtered_ttc_list)):
        # 堆叠每个算法的筛选列表，计算每步的最大和最小值
        stacked_tensors = torch.stack(filtered_list)
        min_vals = torch.min(stacked_tensors, dim=0).values
        max_vals = torch.max(stacked_tensors, dim=0).values

        # 绘制平均值曲线
        plt.plot(avg_ttc, label=f"{name}", color=colors[i], linewidth=2)

        # 绘制波动范围
        # plt.fill_between(range(24), min_vals, max_vals, color=colors[i], alpha=0.3, label=f"{name} Range")

    plt.xlabel("Steps", fontsize=16, fontfamily='Times New Roman')
    plt.ylabel("TTC", fontsize=16, fontfamily='Times New Roman')
    plt.title("TTC Average", fontsize=16, fontfamily='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.grid(True)
    plt.savefig('./ttc.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_jerk_curves(algorithm_names, avg_jerk_list, filtered_jerk_list):
    plt.figure(figsize=(5, 3))
    sns.set_theme(style="darkgrid")

    # 设置颜色
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']

    for i, (name, avg_ttc, filtered_list) in enumerate(zip(algorithm_names, avg_jerk_list, filtered_jerk_list)):
        # 堆叠每个算法的筛选列表，计算每步的最大和最小值
        stacked_tensors = torch.stack(filtered_list)
        min_vals = torch.min(stacked_tensors, dim=0).values
        max_vals = torch.max(stacked_tensors, dim=0).values

        # 绘制平均值曲线
        plt.plot(avg_ttc, label=f"{name}", color=colors[i], linewidth=2)

        # 绘制波动范围
        # plt.fill_between(range(24), min_vals, max_vals, color=colors[i], alpha=0.3, label=f"{name} Range")

    plt.xlabel("Steps", fontsize=16, fontfamily='Times New Roman')
    plt.ylabel("Jerk", fontsize=16, fontfamily='Times New Roman')
    plt.title("Jerk Average", fontsize=16, fontfamily='Times New Roman')
    plt.ylim(0, 9)
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.grid(True)
    plt.savefig('./jerk.png', dpi=600, bbox_inches='tight')
    plt.show()


# DMGTS环境下ego车辆的轨迹（相对）
sac_dmgts_traj = torch.stack(torch.load('./data/SAC_traj_dmgts.pt'))
# 计算碰撞用的bool数组
sac_dmgts_collision = torch.load('./data/SAC_collision_dmgts.pt')
# 直接训练的结果，ego车辆的轨迹（相对）
sac_traj = torch.stack(torch.load('./data/SAC_traj.pt'))
# 计算碰撞使用的bool数组
sac_collision = torch.load('./data/SAC_collision.pt')
# 标签轨迹
label_traj = torch.load('/home/moresweet/gitCloneZone/DMGTS/visualization/new/future.pt')
# 提取的加速度数据
sac_traj_acc = torch.load('data/SAC_acc.pt')
sac_traj_acc_dmgts = torch.load('data/SAC_acc_dmgts.pt')

sac_traj_vel = torch.load('data/SAC_vel.pt')
sac_traj_vel_dmgts = torch.load('data/SAC_vel_dmgts.pt')

label_acceleration = calculate_acceleration(label_traj)

# 判断碰撞很简单，只要记录数组中，最后一个元素是True，那么就是撞了，当然，还有0的说明场景生成的不合适，这样的去除统计比较好
sac_valid_collision = [i for i in sac_collision if len(i) != 0]
sac_collision_rate = sum([1 for i in sac_valid_collision if i[-1] is True]) / len(sac_valid_collision)

sac_dmgts_collision = [i for i in sac_dmgts_collision if len(i) != 0]
sac_dmgts_collision_rate = sum([1 for i in sac_dmgts_collision if i[-1] is True]) / len(sac_dmgts_collision)

# 筛选出已经走完的路径
sac_complete_route = [i for i in sac_traj if not (i == 0).all() and not (i[-1] == 0).all()]
sac_dmgts_complete_route = [i for i in sac_dmgts_traj if not (i == 0).all() and not (i[-1] == 0).all()]
# 成功走完24步（仿真中是48步即算完成）  IR
sac_ir = len(sac_complete_route) / sac_traj.shape[0]
sac_dmgts_ir = len(sac_dmgts_complete_route) / sac_dmgts_traj.shape[0]

sac_ss = compute_speed_satisfaction(sac_traj_vel, label_traj)
sac_dmgts_ss = compute_speed_satisfaction(sac_traj_vel_dmgts, label_traj)

# wd和kl
sac_wd, sac_kl = calculate_wd_and_kl(sac_traj_acc, label_acceleration)
sac_dmgts_wd, sac_dmgts_kl = calculate_wd_and_kl(sac_traj_acc_dmgts, label_acceleration)

# DMGTS环境下ego车辆的轨迹（相对）
ddpg_dmgts_traj = torch.stack(torch.load('./data/DDPG_traj_dmgts.pt'))
# 计算碰撞用的bool数组
ddpg_dmgts_collision = torch.load('./data/DDPG_collision_dmgts.pt')
# 直接训练的结果，ego车辆的轨迹（相对）
ddpg_traj = torch.stack(torch.load('./data/DDPG_traj.pt'))
# 计算碰撞使用的bool数组
ddpg_collision = torch.load('./data/DDPG_collision.pt')
# 提取的加速度数据
ddpg_traj_acc = torch.load('data/DDPG_acc.pt')
ddpg_traj_acc_dmgts = torch.load('data/DDPG_acc_dmgts.pt')

# 判断碰撞很简单，只要记录数组中，最后一个元素是True，那么就是撞了，当然，还有0的说明场景生成的不合适，这样的去除统计比较好
ddpg_valid_collision = [i for i in ddpg_collision if len(i) != 0]
ddpg_collision_rate = sum([1 for i in ddpg_valid_collision if i[-1] is True]) / len(ddpg_valid_collision)

ddpg_dmgts_collision = [i for i in ddpg_dmgts_collision if len(i) != 0]
ddpg_dmgts_collision_rate = sum([1 for i in ddpg_dmgts_collision if i[-1] is True]) / len(ddpg_dmgts_collision)

# 筛选出已经走完的路径
ddpg_complete_route = [i for i in ddpg_traj if not (i == 0).all() and not (i[-1] == 0).all()]
ddpg_dmgts_complete_route = [i for i in ddpg_dmgts_traj if not (i == 0).all() and not (i[-1] == 0).all()]
# 成功走完24步（仿真中是48步即算完成）  IR
ddpg_ir = len(ddpg_complete_route) / ddpg_traj.shape[0]
ddpg_dmgts_ir = len(ddpg_dmgts_complete_route) / ddpg_dmgts_traj.shape[0]

# wd和kl
ddpg_wd, ddpg_kl = calculate_wd_and_kl(ddpg_traj_acc, label_acceleration)
ddpg_dmgts_wd, ddpg_dmgts_kl = calculate_wd_and_kl(ddpg_traj_acc_dmgts, label_acceleration)

ddpg_traj_vel = torch.load('data/DDPG_vel.pt')
ddpg_traj_vel_dmgts = torch.load('data/DDPG_vel_dmgts.pt')
ddpg_ss = compute_speed_satisfaction(ddpg_traj_vel, label_traj)
ddpg_dmgts_ss = compute_speed_satisfaction(ddpg_traj_vel_dmgts, label_traj)

# DMGTS环境下ego车辆的轨迹（相对）
td3_dmgts_traj = torch.stack(torch.load('./data/TD3_traj_dmgts.pt'))
# 计算碰撞用的bool数组
td3_dmgts_collision = torch.load('./data/TD3_collision_dmgts.pt')
# 直接训练的结果，ego车辆的轨迹（相对）
td3_traj = torch.stack(torch.load('./data/TD3_traj.pt'))
# 计算碰撞使用的bool数组
td3_collision = torch.load('./data/TD3_collision.pt')
# 提取的加速度数据
td3_traj_acc = torch.load('data/TD3_acc.pt')
td3_traj_acc_dmgts = torch.load('data/TD3_acc_dmgts.pt')

# 判断碰撞很简单，只要记录数组中，最后一个元素是True，那么就是撞了，当然，还有0的说明场景生成的不合适，这样的去除统计比较好
td3_valid_collision = [i for i in td3_collision if len(i) != 0]
td3_collision_rate = sum([1 for i in td3_valid_collision if i[-1] is True]) / len(td3_valid_collision)

td3_dmgts_collision = [i for i in td3_dmgts_collision if len(i) != 0]
td3_dmgts_collision_rate = sum([1 for i in td3_dmgts_collision if i[-1] is True]) / len(td3_dmgts_collision)

# 筛选出已经走完的路径
td3_complete_route = [i for i in td3_traj if not (i == 0).all() and not (i[-1] == 0).all()]
td3_dmgts_complete_route = [i for i in td3_dmgts_traj if not (i == 0).all() and not (i[-1] == 0).all()]
# 成功走完24步（仿真中是48步即算完成）  IR
td3_ir = len(td3_complete_route) / td3_traj.shape[0]
td3_dmgts_ir = len(td3_dmgts_complete_route) / td3_dmgts_traj.shape[0]

# wd和kl
td3_wd, td3_kl = calculate_wd_and_kl(td3_traj_acc, label_acceleration)
td3_dmgts_wd, td3_dmgts_kl = calculate_wd_and_kl(td3_traj_acc_dmgts, label_acceleration)

td3_traj_vel = torch.load('data/TD3_vel.pt')
td3_traj_vel_dmgts = torch.load('data/TD3_vel_dmgts.pt')
td3_ss = compute_speed_satisfaction(td3_traj_vel, label_traj)
td3_dmgts_ss = compute_speed_satisfaction(td3_traj_vel_dmgts, label_traj)

# DMGTS环境下ego车辆的轨迹（相对）
ppo_dmgts_traj = torch.stack(torch.load('./data/PPO_traj_dmgts.pt'))
# 计算碰撞用的bool数组
ppo_dmgts_collision = torch.load('./data/PPO_collision_dmgts.pt')
# 直接训练的结果，ego车辆的轨迹（相对）
ppo_traj = torch.stack(torch.load('./data/PPO_traj.pt'))
# 计算碰撞使用的bool数组
ppo_collision = torch.load('./data/PPO_collision.pt')
# 提取的加速度数据
ppo_traj_acc = torch.load('data/PPO_acc.pt')
ppo_traj_acc_dmgts = torch.load('data/PPO_acc_dmgts.pt')

# 判断碰撞很简单，只要记录数组中，最后一个元素是True，那么就是撞了，当然，还有0的说明场景生成的不合适，这样的去除统计比较好
ppo_valid_collision = [i for i in ppo_collision if len(i) != 0]
ppo_collision_rate = sum([1 for i in ppo_valid_collision if i[-1] is True]) / len(ppo_valid_collision)

ppo_dmgts_collision = [i for i in ppo_dmgts_collision if len(i) != 0]
ppo_dmgts_collision_rate = sum([1 for i in ppo_dmgts_collision if i[-1] is True]) / len(ppo_dmgts_collision)

# 筛选出已经走完的路径
ppo_complete_route = [i for i in ppo_traj if not (i == 0).all() and not (i[-1] == 0).all()]
ppo_dmgts_complete_route = [i for i in ppo_dmgts_traj if not (i == 0).all() and not (i[-1] == 0).all()]
# 成功走完24步（仿真中是48步即算完成）  IR
ppo_ir = len(ppo_complete_route) / ppo_traj.shape[0]
ppo_dmgts_ir = len(ppo_dmgts_complete_route) / ppo_dmgts_traj.shape[0]

# wd和kl
ppo_wd, ppo_kl = calculate_wd_and_kl(ppo_traj_acc, label_acceleration)
ppo_dmgts_wd, ppo_dmgts_kl = calculate_wd_and_kl(ppo_traj_acc_dmgts, label_acceleration)

ppo_traj_vel = torch.load('data/PPO_vel.pt')
ppo_traj_vel_dmgts = torch.load('data/PPO_vel_dmgts.pt')
ppo_ss = compute_speed_satisfaction(ppo_traj_vel, label_traj)
ppo_dmgts_ss = compute_speed_satisfaction(ppo_traj_vel_dmgts, label_traj)

print_data = {'Algorithm': ['DDPG', 'SAC', 'PPO', 'TD3', 'DDPG_DMGTS', 'SAC_DMGTS', 'PPO_DMGTS', 'TD3_DMGTS'],
              'WD': [np.max(ddpg_wd), np.max(sac_wd), np.max(ppo_wd), np.max(td3_wd), np.min(ddpg_dmgts_wd),
                     np.min(sac_dmgts_wd), np.min(ppo_dmgts_wd), np.min(td3_dmgts_wd)],
              'KL': [np.max(ddpg_kl), np.max(sac_kl), np.max(ppo_kl), np.max(td3_kl), np.min(ddpg_dmgts_kl),
                     np.min(sac_dmgts_kl), np.min(ppo_dmgts_kl), np.min(td3_dmgts_kl)],
              'CR': [ddpg_collision_rate, sac_collision_rate, ppo_collision_rate, td3_collision_rate,
                     ddpg_dmgts_collision_rate,
                     sac_dmgts_collision_rate, ppo_dmgts_collision_rate, td3_dmgts_collision_rate],
              'IR': [ddpg_ir, sac_ir, ppo_ir, td3_ir, ddpg_dmgts_ir, sac_dmgts_ir, ppo_dmgts_ir, td3_dmgts_ir],
              'SS': [ddpg_ss, sac_ss, ppo_ss, td3_ss, ddpg_dmgts_ss, sac_dmgts_ss, ppo_dmgts_ss, td3_dmgts_ss]}

# 创建 DataFrame
df = pd.DataFrame(print_data)

# 计算TTC和加速度
# TTC应该在环境中算
ddpg_ttc = torch.load('data/DDPG_ttc.pt')
sac_ttc = torch.load('data/SAC_ttc.pt')
ppo_ttc = torch.load('data/PPO_ttc.pt')
td3_ttc = torch.load('data/TD3_ttc.pt')

ddpg_ttc, ddpg_ttc_filted_list = process_tensor_list(ddpg_ttc)
sac_ttc, sac_ttc_filtered_list = process_tensor_list(sac_ttc)
ppo_ttc, ppo_ttc_filtered_list = process_tensor_list(ppo_ttc)
td3_ttc, td3_ttc_filtered_list = process_tensor_list(td3_ttc)

# 示例调用
# 假设你已有各算法的 avg_ttc 及对应的 filtered_ttc_list
algorithm_names = ["DDPG", "SAC", "PPO", "TD3"]
avg_ttc_list = [ddpg_ttc, sac_ttc, ppo_ttc, td3_ttc]  # 每个算法的24步平均TTC
filtered_ttc_list = [ddpg_ttc_filted_list, sac_ttc_filtered_list, ppo_ttc_filtered_list, td3_ttc_filtered_list]

plot_ttc_curves(algorithm_names, avg_ttc_list, filtered_ttc_list)

ddpg_jerk = [torch.norm(x, dim=1) for x in ddpg_traj_acc]
sac_jerk = [torch.norm(x, dim=1) for x in sac_traj_acc]
ppo_jerk = [torch.norm(x, dim=1) for x in ppo_traj_acc]
td3_jerk = [torch.norm(x, dim=1) for x in td3_traj_acc]

ddpg_jerk, ddpg_jerk_filted_list = process_tensor_list(ddpg_jerk)
sac_jerk, sac_jerk_filtered_list = process_tensor_list(sac_jerk)
ppo_jerk, ppo_jerk_filted_list = process_tensor_list(ppo_jerk)
td3_jerk, td3_jerk_filted_list = process_tensor_list(td3_jerk)
avg_jerk_list = [ddpg_jerk, sac_jerk, ppo_jerk, td3_jerk]
filtered_jerk_list = [ddpg_jerk_filted_list, sac_jerk_filtered_list, ppo_jerk_filted_list, td3_jerk_filted_list]
plot_jerk_curves(algorithm_names, avg_jerk_list, filtered_jerk_list)

# 提取车的数量
nbrs_count = torch.load('/home/moresweet/gitCloneZone/DMGTS/visualization/new/batch_nbrs_count.pt',
                        map_location=torch.device('cpu'))

# 绘制平均线速度图
plot_average_speeds(nbrs_count, sac_traj_vel_dmgts, ddpg_traj_vel_dmgts, ppo_traj_vel_dmgts, td3_traj_vel_dmgts,
                    'Speed')
# 绘制平均角速度图
ddpg_steer_dmgts = torch.load('data/DDPG_steer_dmgts.pt', map_location=torch.device('cpu'))
sac_steer_dmgts = torch.load('data/SAC_steer_dmgts.pt', map_location=torch.device('cpu'))
ppo_steer_dmgts = torch.load('data/PPO_steer_dmgts.pt', map_location=torch.device('cpu'))
td3_steer_dmgts = torch.load('data/TD3_steer_dmgts.pt', map_location=torch.device('cpu'))
plot_average_speeds(nbrs_count, sac_steer_dmgts, ddpg_steer_dmgts, ppo_steer_dmgts, td3_steer_dmgts, 'Steer')

print(df)
