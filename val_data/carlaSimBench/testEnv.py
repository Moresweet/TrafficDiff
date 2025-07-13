import gym
from gym import spaces
import carla
import random
import time
import numpy as np


class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # 初始化Carla客户端和世界
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        # self.world = self.client.get_world()
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()

        # 同步模式设置
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # 同步模式
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)
        self.spectator = self.world.get_spectator()

        # 蓝图
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.*')[1]  # ego车辆
        self.npc_bp = self.blueprint_library.filter('vehicle.*')[0]  # NPC车辆

        # 动作空间和状态空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # 假设两个动作：加速度和转向角
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(10,), dtype=float)  # 假设状态为10个数值

        # 记录车辆
        self.vehicle = None
        self.npc_vehicles = []

        # 参数
        self.x_range = (-350, 350)
        # self.y_range = (6, 16)
        self.y_range = (26, 37)
        self.z = 11.19
        self.pitch = 0.2
        self.yaw = 0
        self.roll = 0.0
        self.num_vehicles = 40

    def reset(self):
        """
        重新生成场景，放置ego车辆和NPC车辆
        """
        """
            重新生成场景，放置ego车辆和NPC车辆，并为每辆车设定初速度。
            """
        # 销毁已有车辆
        if self.vehicle:
            self.vehicle.destroy()
        for npc in self.npc_vehicles:
            npc.destroy()

        # 生成ego车辆
        spawn_point = carla.Transform(carla.Location(x=0, y=31.43, z=self.z),
                                      carla.Rotation(pitch=self.pitch, yaw=0, roll=self.roll))
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)

        # 设置ego车辆的随机初速度
        ego_velocity = self.generate_random_velocity()
        self.vehicle.set_target_velocity(ego_velocity)

        # 设置摄像机视角
        vehicle_location = self.vehicle.get_location()
        spectator_transform = carla.Transform(
            carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 100),  # 上方100米
            carla.Rotation(pitch=-90)  # 视角向下看
        )
        self.spectator.set_transform(spectator_transform)

        # 生成NPC车辆
        self.npc_vehicles = []
        for _ in range(self.num_vehicles):
            spawn_point = self.generate_random_spawn_point()
            vehicle_npc = self.world.try_spawn_actor(self.npc_bp, spawn_point)
            if vehicle_npc:
                # 设置NPC车辆的随机初速度
                npc_velocity = self.generate_random_velocity()
                vehicle_npc.set_target_velocity(npc_velocity)

                vehicle_npc.set_autopilot(True)
                self.npc_vehicles.append(vehicle_npc)

        return self.get_observation()  # 返回初始观测状态

    def generate_random_velocity(self):
        """
        生成一个随机初速度，用于车辆的初始状态。
        """
        # 定义速度范围，比如 0 到 20 m/s（约 0 到 72 km/h）
        velocity_x = random.uniform(0, 20)  # x方向的初速度
        velocity_y = random.uniform(-2, 2)  # y方向的初速度，可以设置为0让车辆沿x轴前进
        velocity_z = 0  # 假设z轴上的速度为0
        return carla.Vector3D(x=velocity_x, y=velocity_y, z=velocity_z)

    def step(self, action):
        """
        执行动作，控制ego车辆，并返回下一步的状态、奖励、是否结束等信息。
        """
        throttle = float(action[0])  # 加速/减速
        steer = float(action[1])  # 转向

        control = carla.VehicleControl(throttle=throttle, steer=steer)
        self.vehicle.apply_control(control)

        # 推动仿真世界前进
        self.world.tick()

        # 检查是否发生碰撞
        reward, done = self._check_collision()

        # 获取下一个状态
        next_state = self.get_observation()

        return next_state, reward, done, {}

    def _check_collision(self):
        """
        检查是否发生碰撞，并根据碰撞情况返回奖励和结束标志。
        """
        collision_sensor = self.world.get_actors().filter('sensor.other.collision')
        reward = 0
        done = False
        for sensor in collision_sensor:
            if sensor.get_collision_history():  # 如果有碰撞记录
                reward = -100
                done = True
                break
        return reward, done

    def get_observation(self, max_nearby_vehicles=5, relative_coords=True):
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
        if relative_coords is False:
            ego_state = np.array([ego_transform.location.x, ego_transform.location.y,
                              ego_velocity.x, ego_velocity.y])  # ego车辆的状态
        else:
            ego_state = np.array([0, 0, ego_velocity.x, ego_velocity.y])
        # 定义感知范围
        perception_range = 50.0  # 假设ego车辆感知50米范围内的他车

        # 获取所有车辆
        vehicles = self.world.get_actors().filter('vehicle.*')

        nearby_vehicles_state = []

        for vehicle in vehicles:
            if vehicle.id != self.vehicle.id:  # 排除ego车辆
                other_transform = vehicle.get_transform()
                other_velocity = vehicle.get_velocity()

                # 计算他车与ego车辆的距离
                distance = ego_transform.location.distance(other_transform.location)

                # 如果在感知范围内，记录他车的状态
                if distance <= perception_range:
                    if relative_coords:
                        # 计算相对坐标
                        other_x = other_transform.location.x - ego_transform.location.x
                        other_y = other_transform.location.y - ego_transform.location.y
                    else:
                        # 使用全局坐标
                        other_x = other_transform.location.x
                        other_y = other_transform.location.y

                    other_state = np.array([other_x, other_y,
                                            other_velocity.x, other_velocity.y])
                    nearby_vehicles_state.append(other_state)

                # 如果已经收集到的车辆达到上限，则停止
                if len(nearby_vehicles_state) >= max_nearby_vehicles:
                    break

        # 将ego车辆的状态与附近车辆的状态合并
        if len(nearby_vehicles_state) < max_nearby_vehicles:
            # 如果附近车辆数不足最大数量，补零
            nearby_vehicles_state += [np.zeros(4)] * (max_nearby_vehicles - len(nearby_vehicles_state))

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

    def close(self):
        """
        清理环境，销毁车辆和关闭同步模式。
        """
        if self.vehicle:
            self.vehicle.destroy()
        for npc in self.npc_vehicles:
            npc.destroy()

        # 退出同步模式
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)


# 实例化环境
env = CarlaEnv()

# 重置环境，生成场景
state = env.reset()
print("Initial state:", state)

# 模拟一个随机策略运行10步
for step in range(100):
    # 生成随机动作，假设 throttle 和 steer 在 [-1, 1] 之间
    action = np.random.uniform(-1, 1, size=(2,))

    # 执行动作并返回新的状态、奖励、是否结束的信息
    next_state, reward, done, _ = env.step(action)

    # 打印当前步的信息
    # print(f"Step {step + 1}:")
    # print("Action:", action)
    # print("Next state:", next_state)
    # print("Reward:", reward)
    # print("Done:", done)

    # 如果发生碰撞，提前结束仿真
    if done:
        print("Collision detected! Ending simulation.")
        break

# 关闭环境，清理生成的车辆
env.close()