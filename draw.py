import time

import torch
import matplotlib.pyplot as plt
import os

# 读取存储的真值轨迹点
true_trajectory = torch.load('draw_paradigm_fut.pt', map_location='cpu')[40:50,:,:]  # (B, 24, 2)

# 获取当前目录下的所有去噪结果文件
denoised_files = [f for f in os.listdir('visualization/step_matrix') if f.startswith('cur_y_') and f.endswith('.pt')]
denoised_files.sort()  # 按文件名排序

# 轨迹的批大小
batch_size = true_trajectory.shape[0]

# 颜色列表，用于绘制不同的去噪结果
colors = plt.cm.get_cmap('Spectral', len(denoised_files))

# 为每个文件生成图像
for file_index, denoised_file in enumerate(denoised_files):
    denoised_trajectory = torch.load('visualization/step_matrix/'+denoised_file, map_location='cpu')[:,:,:]  # (B, 24, 2)

    # 绘制每个批次中的轨迹
    for batch in range(batch_size):
        # 绘制真值轨迹
        plt.plot(true_trajectory[batch, :, 0], true_trajectory[batch, :, 1], 'o-',
                 color='#C82423', alpha=0.7)

        # 绘制去噪结果
        plt.plot(denoised_trajectory[0, :, 0], denoised_trajectory[0, :, 1], 'o-'
                 , color = colors(file_index), alpha=0.7)
    plt.plot([], [], 'o-', color='#C82423', label='Reference trajectory')
    plt.plot([], [], 'o-', color = colors(file_index), label='Denoised trajectory')
    #　color = colors(file_index)
    # 设置图例、标题和标签
    # plt.title(f'Trajectory Comparison for File {file_index + 1}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(f'visualization/comparsion_pics/trajectory_comparison_{file_index + 1}.png')
    plt.show()
    # time.sleep(0.1)

# 最后会生成多张图，每次去噪结果和真值的对比图
