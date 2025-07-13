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
        self.collision_detected = False

        # 参数
        self.x_range = (-350, 350)
        self.y_range = (26, 37)
        self.z = 11.19
        self.pitch = 0.2
        self.yaw = 0.0
        self.roll = 0.0
        self.num_vehicles = 40
        self.max_nearby_vehicles = 11  # 限制返回的最大邻近车辆数
        self.use_relative_coordinates = True  # 是否使用相对坐标
        self.init_frame = 0
        # 初始化ego和NPC车辆
        self.init_vehicles()

    def init_vehicles(self):
        """
        初始化ego和NPC车辆，只生成一次。
        """
        # 生成ego车辆
        spawn_point = carla.Transform(carla.Location(x=0, y=31.43, z=self.z),
                                      carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
        self.vehicle.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))  # 重置速度
        self.vehicle.set_autopilot(False)
        # 添加碰撞传感器
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(),
                                                       attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

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
                # 赋予随机初速度
                random_velocity = carla.Vector3D(x=random.uniform(0, 5), y=random.uniform(0, 5), z=0)
                vehicle_npc.set_target_velocity(random_velocity)
                vehicle_npc.set_autopilot(True)
                self.npc_vehicles.append(vehicle_npc)

    def collision_data(self, event):
        """
        处理碰撞事件的回调函数。
        """
        self.collision_detected = True
        # print(f"Collision detected with {event.other_actor.type_id} at {event.timestamp}")

    def reset(self):
        # if self.vehicle:
        #     self.vehicle.destroy()
        #     time.sleep(1)
        # 重置ego车辆位置和速度
        spawn_point = carla.Transform(carla.Location(x=0, y=31.43, z=self.z),
                                      carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll))
        self.vehicle.set_transform(spawn_point)
        # self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
        # while self.vehicle is None:
        #     time.sleep(0.1)
        #     self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
        #     time.sleep(1)
        self.vehicle.set_target_velocity(carla.Vector3D(x=random.uniform(8, 14), y=0, z=0))  # 重置速度
        self.vehicle.set_autopilot(False)

        # 重置NPC车辆位置和速度
        for vehicle_npc in self.npc_vehicles:
            spawn_point = self.generate_random_spawn_point()
            vehicle_npc.set_transform(spawn_point)
            random_velocity = carla.Vector3D(x=random.uniform(0, 3), y=random.uniform(-0.1, 0.1), z=0)
            vehicle_npc.set_target_velocity(random_velocity)
            vehicle_npc.set_autopilot(True)

        # 重置碰撞检测
        self.collision_detected = False

        # 推进仿真几帧，让车辆运动起来
        for _ in range(20):  # 推进10帧
            self.world.tick()
        self.vehicle.set_autopilot(False)
        self.init_frame = self.world.get_snapshot().frame
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

        # 碰撞或离开路面则返回True
        return crashed or off_road

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

        # 推动仿真世界前进
        self.world.tick()

        # 检查是否发生碰撞
        crashed = self._check_collision()

        # 获取下一个状态
        next_state = self.get_observation()

        # 计算奖励
        rewards = self.reward(crashed)

        # 判断仿真是否结束
        done = self.is_terminate()  # 如果碰撞则结束
        truncated = self.world.get_snapshot().frame - self.init_frame >= 48

        # 更新info
        info = {
            'action': action,
            'crashed': crashed,
            'rewards': rewards,
            'speed': self.vehicle.get_velocity().length()
        }

        # 总奖励
        total_reward = sum(rewards.values())
        # 车辆停稳后，再重启，防止初速度带来的影响
        if truncated or done:
            self.emergency_stop()
            # print("stop!")
        return next_state, total_reward, done, truncated, info

    def reward(self, crashed):
        """
        计算奖励，包括碰撞奖励、高速奖励和保持在车道内的奖励。
        """
        collision_reward = -200 if crashed else 0
        # speed = self.vehicle.get_velocity().length()
        speed = self.vehicle.get_velocity().x
        # 高速奖励，假设速度超过一定值则奖励
        high_speed_reward = 0
        if speed <= 0:
            high_speed_reward = -1
        else:
            high_speed_reward = speed * 2

        # 朝向奖励，鼓励车辆沿着x轴方向行驶
        forward_vector = self.vehicle.get_transform().get_forward_vector()
        heading_reward = forward_vector.x * 4  # 只考虑x方向前进的奖励
        # high_speed_reward = high_speed_reward * forward_vector.x
        if heading_reward < 0:
            heading_reward = heading_reward * 10
            # high_speed_reward = high_speed_reward * 0.5

        # 控制动作惩罚，减少过度的方向盘和油门/刹车动作
        control = self.vehicle.get_control()
        # action_penalty = -(control.steer ** 2 + control.throttle ** 2 + control.brake ** 2) * 0.1
        action_penalty = -(control.steer ** 2) * 5

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
        # 定义感知范围
        perception_range = 100.0  # 假设ego车辆感知50米范围内的他车

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
                    if self.use_relative_coordinates:
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
                if len(nearby_vehicles_state) >= self.max_nearby_vehicles:
                    break

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
