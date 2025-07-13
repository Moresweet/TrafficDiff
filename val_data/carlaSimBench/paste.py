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

# 判断碰撞很简单，只要记录数组中，最后一个元素是True，那么就是撞了，当然，还有0的说明场景生成的不合适，这样的去除统计比较好
ppo_valid_collision = [i for i in ppo_collision if len(i) != 0]
ppo_collision_rate = sum([1 for i in ppo_valid_collision if i[-1] is True]) / len(ppo_valid_collision)

ppo_dmgts_collision = [i for i in ppo_dmgts_collision if len(i) != 0]
ppo_dmgts_collision_rate = sum([1 for i in ppo_dmgts_collision if i[-1] is True]) / len(ppo_dmgts_collision)

# 筛选出已经走完的路径
ppo_complete_route = [i for i in ppo_traj if not (i == 0).all() and not (i[-1] == 0).all()]
ppo_dmgts_complete_route = [i for i in ppo_dmgts_traj if not (i == 0).all() and not (i[-1] == 0).all()]
# 成功走完24步（仿真中是48步即算完成）  IR
ir = len(ppo_complete_route) / ppo_traj.shape[0]
dmgts_ir = len(ppo_dmgts_complete_route) / ppo_dmgts_traj.shape[0]

# wd和kl
ppo_wd, ppo_kl = calculate_wd_and_kl(ppo_traj_acc, label_acceleration)