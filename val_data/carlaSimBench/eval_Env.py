import gym
from gym import spaces
import carla
import random
import time
import numpy as np
import torch
import pygame
import math


class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # 初始化Carla客户端和世界
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()

        # 同步模式设置
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # 同步模式
        self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)
        self.spectator = self.world.get_spectator()

        # 蓝图
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.*')[1]  # ego车辆
        self.npc_bp = self.blueprint_library.filter('vehicle.*')[0]  # NPC车辆
        # 碰撞检测器蓝图
        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')  # 碰撞传感器

        # 动作空间和状态空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # 假设两个动作：加速度和转向角
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(10,), dtype=float)  # 假设状态为10个数值

        # 记录车辆
        self.vehicle = None
        self.npc_vehicles = []
        self.npc_waiting = []
        self.init_npc_location = []
        self.collision_detected = False

        # 保存的碰撞情况
        self.collision_record = []
        # 一个scene的碰撞情况
        self.scene_collision = []

        # 返回的tensor  （轨迹，变换后的轨迹，本身就是相对坐标，(0,0)开始）
        self.record_tensor = []
        self.temp_tensor = torch.zeros(24, 2)

        # 保存acc_x,acc_y，
        self.record_acc = []
        self.record_vel = []
        self.record_angle = []
        self.temp_acc = torch.zeros(24, 2)
        self.temp_vel = torch.zeros(24, 2)
        self.temp_angle = torch.zeros(24, 2)

        # 保存acc_x,acc_y，
        self.record_vel = []
        self.temp_vel = torch.zeros(24, 2)
        self.record_angle = []
        self.temp_angle = torch.zeros(24, 2)

        # 保存TTC
        self.record_ttc = []
        self.temp_ttc = torch.zeros(24)

        # 参数
        self.x_range = (-350, 350)
        self.y_range = (26, 37)
        # self.z = 11.19
        self.z = 0.2819
        # self.pitch = 0.3
        # self.yaw = 0.0
        # self.roll = 0.0
        self.pitch = 0.0
        self.yaw = -0.29
        self.roll = 0.0
        self.num_vehicles = 40
        self.max_nearby_vehicles = 11  # 限制返回的最大邻近车辆数
        self.use_relative_coordinates = True  # 是否使用相对坐标
        self.init_frame = 0

        self.use_proposal = False

        # 每个batch的场景索引
        self.npc_scene_id = -1
        self.data_dir = '/home/moresweet/gitCloneZone/DMGTS/visualization/new/'
        self.npc_start_points = torch.load(self.data_dir + 'batch_nbrs_start.pt', map_location=torch.device('cpu'))
        self.npc_future_points = torch.load(self.data_dir + 'nbrs_fut.pt', map_location=torch.device('cpu'))
        self.npc_dmgts_points = torch.load(self.data_dir + 'all_nbr_predictions.pt', map_location=torch.device('cpu'))
        self.npc_scene_vehicle_numbers = torch.load(self.data_dir + 'batch_nbrs_count.pt',
                                                    map_location=torch.device('cpu'))
        # 使用仿真的延迟，把车辆的初速度推起来，推起来之后的ego车辆作为原点
        self.ego_init_x = None
        self.ego_init_y = None
        # 计算的实际坐标
        self.npc_traj = None
        self.dmgts_traj = None
        self.npc_start_scaled = None
        # 坐标系对不上，进行缩放
        self.scale_factor = 0.4
        # step数，上限24,指示24步渲染
        self.step_index = 0
        self.npc_offset = -1
        self.npc_velocity = None

        # 加入摄像头
        self.camera_front = None
        self.camera_back = None
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')
        self.camera_bp.set_attribute('image_size_y', '500')
        self.camera_bp.set_attribute('fov', '110')
        self.front_image = np.zeros((500, 800, 3), dtype=np.uint8)
        self.back_image = np.zeros((500, 800, 3), dtype=np.uint8)
        self.screen = None

        # 初始化ego和NPC车辆
        self.init_vehicles()

    def emergency_stop(self):
        control = self.vehicle.get_control()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 1.0
        for _ in range(10):
            self.vehicle.apply_control(control)
            self.world.tick()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        # 松刹车
        self.vehicle.apply_control(control)
        self.world.tick()

    def init_convert_coord(self):
        # 转换npc的未来轨迹，倒转(x,y)，然后加上起始点（相对坐标->绝对坐标），乘以地图缩放系数（将坐标在carla的地图中合理），
        # 然后加上ego车辆的起始位置，加上偏置（数据集坐标系->carla坐标系）
        self.npc_traj = ((self.npc_future_points[..., [1, 0]] +
                          torch.stack(self.npc_start_points).unsqueeze(1)[..., [1, 0]]) * self.scale_factor +
                         torch.FloatTensor([self.vehicle.get_location().x, self.vehicle.get_location().y]))[:, :24,
                        :]  # (1480, 24, 2)
        # 转换算法生成轨迹
        self.dmgts_traj = ((torch.stack(self.npc_dmgts_points).squeeze(1)[..., [1, 0]] +
                            torch.stack(self.npc_start_points).unsqueeze(1).unsqueeze(1)[..., [1, 0]]) *
                           self.scale_factor) + torch.FloatTensor(
            [self.vehicle.get_location().x, self.vehicle.get_location().y])  # (1480,2,24,2)
        # 地图中的运动和数据集中的是垂直关系，因此需要(x,y)逆置
        self.npc_start_scaled = (torch.stack([i[[1, 0]] * self.scale_factor for i in self.npc_start_points]) +
                                 torch.FloatTensor([self.vehicle.get_location().x, self.vehicle.get_location().y]))
        self.ego_init_x = self.vehicle.get_location().x
        self.ego_init_y = self.vehicle.get_location().y

    def init_vehicles(self):
        """
        初始化ego和NPC车辆，只生成一次。
        """
        # 生成ego车辆
        spawn_point = carla.Transform(carla.Location(x=190.25, y=-370.10, z=self.z),
                                      carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)
        # 添加碰撞传感器
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(),
                                                       attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        # 添加摄像头
        camera_transform_front = carla.Transform(carla.Location(x=1.5, y=0, z=30), carla.Rotation(pitch=-90))
        self.camera_front = self.world.spawn_actor(self.camera_bp, camera_transform_front, attach_to=self.vehicle)
        camera_transform_back = carla.Transform(carla.Location(x=-4, y=-0.2, z=3), carla.Rotation(pitch=-30, yaw=0))
        self.camera_back = self.world.spawn_actor(self.camera_bp, camera_transform_back, attach_to=self.vehicle)
        self.camera_front.listen(lambda image: self.front_camera_callback(image))
        self.camera_back.listen(lambda image: self.back_camera_callback(image))
        vehicle_location = self.vehicle.get_location()
        spectator_transform = carla.Transform(
            carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 100),  # 上方100米
            carla.Rotation(pitch=-90)  # 视角向下看
        )
        self.spectator.set_transform(spectator_transform)
        pygame.init()
        # 创建pygame窗口
        self.screen = pygame.display.set_mode((800, 1000))
        pygame.display.set_caption("DMGTS valid window")

        # 生成NPC车辆
        self.npc_vehicles = []
        # for idx in range(self.npc_scene_vehicle_numbers[self.npc_scene_id]):
        #     # spawn_x = self.npc_start_points[self.npc_offset + idx][0]
        #     # spawn_y = self.npc_start_points[self.npc_offset + idx][1]
        #     spawn_x = self.npc_start_scaled[self.npc_offset+idx][0].item()
        #     spawn_y = self.npc_start_scaled[self.npc_offset+idx][1].item()
        #     spawn_point = carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=self.z),
        #                                   carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        #     vehicle_npc = self.world.try_spawn_actor(self.npc_bp, spawn_point)
        #     self.npc_vehicles.append(vehicle_npc)
        # self.npc_offset += self.npc_scene_vehicle_numbers[self.npc_scene_id]
        for idx in range(20):
            # spawn_x = self.npc_start_points[self.npc_offset + idx][0]
            # spawn_y = self.npc_start_points[self.npc_offset + idx][1]
            spawn_x = idx // 4 * 5 - 350
            spawn_y = idx % 5 * 2 + 26
            self.init_npc_location.append(np.array((spawn_x, spawn_y)))
            spawn_point = carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=self.z),
                                          carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
            vehicle_npc = self.world.try_spawn_actor(self.npc_bp, spawn_point)
            self.npc_waiting.append(vehicle_npc)

    def process_img(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def front_camera_callback(self, image):
        self.front_image = self.process_img(image)

    def back_camera_callback(self, image):
        self.back_image = self.process_img(image)

    def collision_data(self, event):
        """
        处理碰撞事件的回调函数。
        """
        self.collision_detected = True
        # print(f"Collision detected with {event.other_actor.type_id} at {event.timestamp}")

    def reset(self):
        self.step_index = 0
        self.npc_scene_id += 1
        # 一开始不能偏移，需要过后再偏移，offset应该滞后于
        if self.npc_offset == -1:
            self.npc_offset = 0
        else:
            self.npc_offset += self.npc_scene_vehicle_numbers[self.npc_scene_id - 1]
        # if self.vehicle:
        #     self.vehicle.destroy()
        # 清除当前的NPC车辆
        # for npc in self.npc_vehicles:
        #     time.sleep(0.5)
        #     npc.destroy()
        # self.npc_vehicles = []
        # spawn_point = carla.Transform(carla.Location(x=0, y=31.43, z=self.z),
        #                               carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        spawn_point = carla.Transform(carla.Location(x=190.25, y=-370.10, z=self.z),
                                      carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        self.vehicle.set_transform(spawn_point)
        # self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)
        self.vehicle.set_target_velocity(carla.Vector3D(x=random.uniform(8, 14), y=0, z=0))
        vehicle_location = self.vehicle.get_location()
        spectator_transform = carla.Transform(
            carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 100),  # 上方100米
            carla.Rotation(pitch=-90)  # 视角向下看
        )
        self.spectator.set_transform(spectator_transform)

        # 推进仿真几帧，让车辆运动起来
        for _ in range(10):  # 推进10帧
            self.world.tick()
        self.init_convert_coord()
        # 处理npc数组和waiting数组，多退少补
        if len(self.npc_vehicles) > self.npc_scene_vehicle_numbers[self.npc_scene_id]:
            # 多退
            excess_vehicles = len(self.npc_vehicles) - self.npc_scene_vehicle_numbers[self.npc_scene_id]
            # 移动多余的车辆到 npc_waiting
            for _ in range(excess_vehicles):
                vehicle = self.npc_vehicles.pop()  # 从 npc_vehicles 移除一个车辆
                vehicle.set_transform(
                    carla.Transform(carla.Location(x=self.init_npc_location[len(self.npc_vehicles)][0].astype(float),
                                                   y=self.init_npc_location[len(self.npc_vehicles)][1].astype(float),
                                                   z=self.z),
                                    carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)))
                self.npc_waiting.insert(0, vehicle)  # 插入到 npc_waiting 的开头
            # 多退的车辆需要手动归位

        elif len(self.npc_vehicles) < self.npc_scene_vehicle_numbers[self.npc_scene_id]:
            # 少补
            shortage_vehicles = self.npc_scene_vehicle_numbers[self.npc_scene_id] - len(self.npc_vehicles)
            # 从 npc_waiting 中补足缺少的车辆，补足的车辆位置后续由程序托管
            for _ in range(shortage_vehicles):
                if self.npc_waiting:
                    vehicle = self.npc_waiting.pop(0)  # 从 npc_waiting 中取出第一个车辆
                    self.npc_vehicles.append(vehicle)  # 添加到 npc_vehicles 中
        for idx in range(self.npc_scene_vehicle_numbers[self.npc_scene_id]):
            spawn_x = self.npc_start_scaled[self.npc_offset + idx][0].item()
            spawn_y = self.npc_start_scaled[self.npc_offset + idx][1].item()
            spawn_point = carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=self.z),
                                          carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
            # vehicle_npc = self.world.try_spawn_actor(self.npc_bp, spawn_point)
            self.npc_vehicles[idx].set_transform(spawn_point)
            # time.sleep(0.2)
            # self.npc_vehicles.append(vehicle_npc)
        # 重置碰撞检测
        self.collision_detected = False
        self.init_frame = self.world.get_snapshot().frame
        conj_start_traj = None
        if self.use_proposal is False:
            conj_start_traj = torch.cat((self.npc_traj, self.npc_start_scaled.unsqueeze(1)), dim=1)
        else:
            conj_start_traj = torch.cat((self.npc_traj, self.npc_start_scaled.unsqueeze(1)), dim=1)
        self.npc_velocity = (conj_start_traj[:, 1:, :] - conj_start_traj[:, :-1, :]) / 0.2
        # 返回初始状态和信息
        state = self.get_observation()
        info = {'action': [0, 0], 'crashed': False, 'rewards': {}, 'speed': 0.0}
        return state, info

    def is_terminate(self):
        """
        判断是否终止仿真：碰撞或者车辆离开当前路面。
        """
        # 检查是否发生碰撞
        crashed = self._check_collision()

        # 获取车辆位置
        location = self.vehicle.get_location()

        # 判断是否离开路面
        off_road = not (-350 < location.x < 350 and 26 < location.y < 37)
        if self.step_index >= 2:
            self.scene_collision.append(crashed)
        # 碰撞或离开路面则返回True
        return crashed or off_road

    def adjust_npc_y(self, cur_npc_y, error_y):
        """
        根据 error_y 对 cur_npc_y 进行累加偏移修正。

        参数:
            cur_npc_y (torch.Tensor): 当前 NPC 车辆的 y 方向坐标，形状为 (24,)。
            error_y (float): NPC 车辆和 ego 车辆在 y 方向的差值。

        返回:
            torch.Tensor: 调整后的 NPC 车辆 y 方向坐标。
        """
        # 初始化累加修正量
        adjusted_y = cur_npc_y.clone()  # 拷贝当前 y 值
        cumulative_offset = 0.0  # 累计偏移量

        for i in range(adjusted_y.shape[0]):
            # 生成 0 到 0.05 之间的随机值
            random_factor = torch.rand(1).item() * 0.005
            # 计算当前点的偏移量
            current_offset = error_y * random_factor
            # 累加偏移量
            cumulative_offset += current_offset
            # 更新 y 值
            adjusted_y[i] += cumulative_offset

        return adjusted_y

    def calculate_ttc(self, ego_vehicle, target_vehicle):
        # 获取位置
        ego_location = ego_vehicle.get_location()
        target_location = target_vehicle.get_location()

        # 获取速度
        ego_velocity = ego_vehicle.get_velocity()
        target_velocity = target_vehicle.get_velocity()

        # 计算距离
        distance = math.sqrt(
            (ego_location.x - target_location.x) ** 2 +
            (ego_location.y - target_location.y) ** 2
        )

        # 计算相对速度
        relative_velocity = math.sqrt(
            (ego_velocity.x - target_velocity.x) ** 2 +
            (ego_velocity.y - target_velocity.y) ** 2
        )

        # 计算TTC
        if relative_velocity > 0:
            ttc = distance / relative_velocity
        else:
            ttc = float('inf')  # 表示不会发生碰撞

        return ttc

    def step(self, action):
        """
        执行动作，控制ego车辆，并返回下一步的状态、奖励、是否结束等信息。
        """
        throttle_input = float(action[0])  # 加速/减速
        steer = float(action[1])  # 转向

        # 如果throttle_input > 0，表示加速，设置throttle；否则设置brake
        if throttle_input > 0:
            throttle = throttle_input  # 取值范围0到1
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_input  # 取值范围0到1

        # 创建车辆控制指令
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        vehicle_index = 0
        # 颜色和线宽度的设置
        line_color = carla.Color(1, 1, 0)  # 荧光蓝绿色
        line_thickness = 0.2  # 线的粗细
        # 0,1帧是初始点状态
        if self.step_index >= 2:
            for idx in range(self.npc_scene_vehicle_numbers[self.npc_scene_id]):
                spawn_x = self.npc_traj[self.npc_offset + idx][self.step_index // 2 - 1][0].item()
                spawn_y = self.npc_traj[self.npc_offset + idx][self.step_index // 2 - 1][1].item()
                spawn_point = carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=self.z),
                                              carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
                self.npc_vehicles[vehicle_index].set_transform(spawn_point)
                vehicle_index += 1
                trajectory = self.npc_traj[self.npc_offset + idx]
                # 超出一定距离的不动了，下面的15是生效范围
                if torch.norm(trajectory[0, :] - torch.FloatTensor((self.ego_init_x, self.ego_init_y))) < 15:
                    # 下面是修正逻辑
                    adjusted_y = trajectory[:, 1].clone()
                    new_y = self.adjust_npc_y(adjusted_y, self.vehicle.get_location().y - trajectory[:, 1][0])
                    trajectory[:, 1] = new_y
                # 遍历轨迹点，绘制轨迹线
                for i in range(self.step_index // 2 - 1, len(trajectory) - 1):
                    start_x, start_y = trajectory[i][0].item(), trajectory[i][1].item()
                    end_x, end_y = trajectory[i + 1][0].item(), trajectory[i + 1][1].item()

                    # 设置起点和终点的高度为固定值self.z
                    start_location = carla.Location(x=start_x, y=start_y, z=self.z)
                    end_location = carla.Location(x=end_x, y=end_y, z=self.z)

                    # 绘制轨迹线
                    self.world.debug.draw_line(
                        start_location, end_location,
                        thickness=line_thickness, color=line_color, life_time=0.2  # life_time设为短值，以实现动态刷新
                    )

                # 也可以选择在每个轨迹点绘制一个小点，以更清晰地展示轨迹路径
                for point in trajectory[self.step_index // 2 - 1:, :]:
                    point_location = carla.Location(x=point[0].item(), y=point[1].item(), z=self.z)
                    self.world.debug.draw_point(
                        point_location, size=0.1, color=line_color, life_time=0.2
                    )

        self.temp_tensor[self.step_index // 2, :] = torch.FloatTensor(
            (self.vehicle.get_location().x - self.ego_init_x, self.vehicle.get_location().y - self.ego_init_y))
        self.temp_acc[self.step_index // 2, :] = torch.FloatTensor(
            (self.vehicle.get_acceleration().x, self.vehicle.get_acceleration().y))
        self.temp_vel[self.step_index // 2, :] = torch.FloatTensor(
            (self.vehicle.get_velocity().x, self.vehicle.get_velocity().y))
        self.temp_angle[self.step_index // 2, :] = torch.FloatTensor(
            (self.vehicle.get_angular_velocity().x, self.vehicle.get_angular_velocity().y))
        # 每步计算ego车辆与所有NPC车辆的TTC
        min_ttc = float('inf')
        for target_vehicle in self.npc_vehicles:
            ttc = self.calculate_ttc(self.vehicle, target_vehicle)
            if ttc < min_ttc:
                min_ttc = ttc
        self.temp_ttc[self.step_index // 2] = torch.FloatTensor([min_ttc])
        self.step_index += 1

        # 推动仿真世界前进
        self.world.tick()
        # 渲染前后视角
        front_surface = pygame.surfarray.make_surface(self.front_image.swapaxes(0, 1))
        back_surface = pygame.surfarray.make_surface(self.back_image.swapaxes(0, 1))

        self.screen.blit(front_surface, (0, 0))
        self.screen.blit(back_surface, (0, 500))
        pygame.display.flip()
        # if self.step_index == 16:
        #     print("debug")

        # 检查是否发生碰撞
        crashed = self._check_collision()

        # 获取下一个状态
        next_state = self.get_observation()

        # 计算奖励
        rewards = self.reward(crashed)

        # 判断仿真是否结束
        done = self.is_terminate()  # 如果碰撞则结束
        truncated = self.world.get_snapshot().frame - self.init_frame >= 48
        if done or truncated:
            self.record_tensor.append(self.temp_tensor)
            self.record_acc.append(self.temp_acc)
            self.record_vel.append(self.temp_vel)
            self.record_angle.append(self.temp_angle)
            self.record_ttc.append(self.temp_ttc)
            self.collision_record.append(self.scene_collision)
            self.scene_collision = []
            self.temp_tensor = torch.zeros(24, 2)
            self.temp_acc = torch.zeros(24, 2)
            self.temp_vel = torch.zeros(24, 2)
            self.temp_angle = torch.zeros(24, 2)
            self.temp_ttc = torch.zeros(24)
            self.emergency_stop()

        # 更新info
        info = {
            'action': action,
            'crashed': crashed,
            'rewards': rewards,
            'speed': self.vehicle.get_velocity().length()
        }

        # 总奖励
        total_reward = sum(rewards.values())

        return next_state, total_reward, done, truncated, info

    def reward(self, crashed):
        """
        计算奖励，包括碰撞奖励、高速奖励和保持在车道内的奖励。
        """
        collision_reward = -100 if crashed else 0
        speed = self.vehicle.get_velocity().length()

        # 高速奖励，假设速度超过一定值则奖励
        high_speed_reward = 0
        if speed <= 0:
            high_speed_reward = -1
        else:
            high_speed_reward = speed * 2

        # 朝向奖励，鼓励车辆沿着x轴方向行驶
        forward_vector = self.vehicle.get_transform().get_forward_vector()
        heading_reward = forward_vector.x * 2  # 只考虑x方向前进的奖励
        # high_speed_reward = high_speed_reward * forward_vector.x
        if heading_reward < 0:
            heading_reward = heading_reward * 10
            # high_speed_reward = high_speed_reward * 0.5

        # 控制动作惩罚，减少过度的方向盘和油门/刹车动作
        control = self.vehicle.get_control()
        # action_penalty = -(control.steer ** 2 + control.throttle ** 2 + control.brake ** 2) * 0.1
        action_penalty = -(control.steer ** 2) * 3

        # 保持在道路上奖励（假设在地图坐标范围内）
        location = self.vehicle.get_location()
        on_road_reward = 0.1 if -350 < location.x < 350 and 26 < location.y < 37 else -100

        return {
            'collision_reward': collision_reward,
            'high_speed_reward': high_speed_reward,
            'on_road_reward': on_road_reward,
            'heading_reward': heading_reward,
            'action_penalty': action_penalty,
            'right_lane_reward': 0.0  # 暂时不考虑车道奖励
        }

    def _check_collision(self):
        """
        检查是否发生碰撞，返回碰撞标志。
        """
        # collision_sensor = self.world.get_actors().filter('sensor.other.collision')
        # for sensor in collision_sensor:
        #     if sensor.get_collision_history():  # 如果有碰撞记录
        #         return True
        # return False
        return self.collision_detected

    def get_observation(self):
        """
        获取ego车辆和附近车辆的状态信息，返回每辆车的(x, y, vx, vy)，可设置是否返回相对坐标。

        参数:
            max_nearby_vehicles (int): 附近车辆的最大数量。
            relative_coords (bool): 是否返回相对坐标，True 表示返回相对于ego车辆的坐标。
        返回:
            numpy array: 返回每辆车的状态 (x, y, vx, vy)，每行代表一辆车。
        """
        # 获取ego车辆的状态
        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        if self.use_relative_coordinates is False:
            ego_state = np.array([ego_transform.location.x, ego_transform.location.y,
                                  ego_velocity.x, ego_velocity.y])  # ego车辆的状态
        else:
            ego_state = np.array([0, 0, ego_velocity.x, ego_velocity.y])

        nearby_vehicles_state = []

        for idx, vehicle in enumerate(self.npc_vehicles):
            other_transform = vehicle.get_transform()
            other_velocity = None
            if (self.step_index // 2) == 24:
                # 最后一个点的速度没算，直接顺延吧
                other_velocity = self.npc_velocity[self.npc_offset + idx][self.step_index // 2 - 1]
            else:
                other_velocity = self.npc_velocity[self.npc_offset + idx][self.step_index // 2]
            if self.use_relative_coordinates:
                # 计算相对坐标
                other_x = other_transform.location.x - ego_transform.location.x
                other_y = other_transform.location.y - ego_transform.location.y
            else:
                # 使用全局坐标
                other_x = other_transform.location.x
                other_y = other_transform.location.y

            other_state = np.array([other_x, other_y,
                                    other_velocity[0].item(), other_velocity[1].item()])
            nearby_vehicles_state.append(other_state)

        # 按照 (x, y) 距离 (0, 0) 的远近排序
        nearby_vehicles_state = sorted(nearby_vehicles_state, key=lambda v: np.linalg.norm(v[:2]))
        # 将ego车辆的状态与附近车辆的状态合并
        if len(nearby_vehicles_state) < self.max_nearby_vehicles:
            # 如果附近车辆数不足最大数量，补零
            nearby_vehicles_state += [np.zeros(4)] * (self.max_nearby_vehicles - len(nearby_vehicles_state))
            # 如果附近车辆数不足最大数量，补-inf
            # nearby_vehicles_state += [np.full(4, -np.inf)] * (self.max_nearby_vehicles - len(nearby_vehicles_state))

        # 合并ego状态和他车状态
        observation = np.vstack([ego_state] + nearby_vehicles_state)

        return observation

    def generate_random_spawn_point(self):
        """
        根据给定的范围生成一个随机的车辆生成点。
        """
        x = random.uniform(self.x_range[0], self.x_range[1])
        y = random.uniform(self.y_range[0], self.y_range[1])
        return carla.Transform(carla.Location(x=x, y=y, z=self.z),
                               carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))

    def close(self, algorithm_name=''):
        """
        清理环境，销毁车辆和关闭同步模式。
        """
        if self.use_proposal is False:
            torch.save(self.record_angle, './data/' + algorithm_name + '_steer.pt')
            torch.save(self.record_ttc, './data/' + algorithm_name + '_ttc.pt')
            torch.save(self.record_tensor, './data/' + algorithm_name + '_traj.pt')
            torch.save(self.collision_record, './data/' + algorithm_name + '_collision.pt')
            torch.save(self.record_acc, './data/' + algorithm_name + '_acc.pt')
            torch.save(self.record_vel, './data/' + algorithm_name + '_vel.pt')
        else:
            torch.save(self.record_angle, './data/' + algorithm_name + '_steer_dmgts.pt')
            torch.save(self.record_ttc, './data/' + algorithm_name + '_ttc_dmgts.pt')
            torch.save(self.record_tensor, './data/' + algorithm_name + '_traj_dmgts.pt')
            torch.save(self.collision_record, './data/' + algorithm_name + '_collision_dmgts.pt')
            torch.save(self.record_acc, './data/' + algorithm_name + '_acc_dmgts.pt')
            torch.save(self.record_vel, './data/' + algorithm_name + '_vel_dmgts.pt')
        if self.vehicle:
            self.vehicle.destroy()
        for npc in self.npc_vehicles:
            npc.destroy()

        # 退出同步模式
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

# # 实例化环境
# env = CarlaEnv()
#
# # 重置环境，生成场景
# state, info = env.reset()
# print("Initial state:", state)
# print("Initial info:", info)
#
# # 模拟一个随机策略运行10步
# for step in range(1000):
#     # 生成随机动作
#     action = np.random.uniform(-1, 1, size=(2,))
#
#     # 执行动作并返回新的状态、奖励、是否结束的信息
#     next_state, reward, done, truncated, info = env.step(action)
#
#     # 打印当前步的信息
#     # print(f"Step {step + 1}:")
#     # print("Action:", action)
#     # print("Next state:", next_state)
#     # print("Reward:", reward)
#     # print("Done:", done)
#     # print("Info:", info)
#
#     if done:
#         print("Collision detected! Ending simulation.")
#         break
#
# # 关闭环境
# env.close()
