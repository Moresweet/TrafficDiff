import carla
import json
import random
import numpy as np


class CarlaEnv:
    def __init__(self, config_file):
        # 连接到 Carla 仿真器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town03')  # 可以更改为你想要的地图
        self.synchronous_mode = True

        # 读取配置文件
        with open(config_file) as f:
            self.config = json.load(f)

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]  # 主角车辆
        self.npc_bps = self.blueprint_library.filter('vehicle.*')  # NPC车辆

        # 初始化控制变量
        self.vehicle = None
        self.npcs = []
        self.collision_hist = []
        self.actor_list = []
        self.setup_sensors()

        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def setup_sensors(self):
        # 传感器设置，比如碰撞检测
        collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.collision_sensor)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self):
        # 清除之前的车辆
        self.clear_vehicles()
        self.collision_hist = []

        # 重置主车辆
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # 重置NPC车辆
        for npc_config in self.config["npcs"]:
            npc_spawn_point = carla.Transform(
                carla.Location(x=npc_config["x"], y=npc_config["y"], z=npc_config["z"]),
                carla.Rotation(pitch=0, yaw=npc_config["yaw"], roll=0)
            )
            npc_bp = random.choice(self.npc_bps)
            npc_vehicle = self.world.spawn_actor(npc_bp, npc_spawn_point)
            self.actor_list.append(npc_vehicle)
            self.npcs.append(npc_vehicle)

        # 轨迹重置
        self.npc_trajectories = {}
        for npc_config in self.config["npcs"]:
            self.npc_trajectories[npc_config["id"]] = npc_config["trajectory"]

        self.world.tick()

        return self.get_state()

    def clear_vehicles(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

    def get_state(self):
        # 返回车辆状态，可以包括位置、速度等信息
        vehicle_transform = self.vehicle.get_transform()
        vehicle_velocity = self.vehicle.get_velocity()
        state = {
            "position": (vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z),
            "velocity": (vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z),
            "collision": len(self.collision_hist) > 0
        }
        return state

    def step(self, action):
        # action可以是加速度、转向等控制指令
        throttle, steer, brake = action
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        # 更新 NPC 的轨迹
        for npc_vehicle in self.npcs:
            npc_id = npc_vehicle.id
            if npc_id in self.npc_trajectories and len(self.npc_trajectories[npc_id]) > 0:
                next_waypoint = self.npc_trajectories[npc_id].pop(0)
                npc_vehicle.set_transform(carla.Transform(
                    carla.Location(x=next_waypoint[0], y=next_waypoint[1], z=next_waypoint[2]),
                    carla.Rotation(yaw=next_waypoint[3])
                ))

        # 让 Carla 仿真前进一步
        self.world.tick()

        # 获取下一步的状态
        next_state = self.get_state()

        # 计算奖励
        done = next_state["collision"]
        reward = -100 if done else 0

        return next_state, reward, done

# 使用方法
env = CarlaEnv('npc_config.json')
state = env.reset()
action = [0.5, 0.0, 0.0]  # 假设加速0.5，方向盘居中，不刹车
next_state, reward, done = env.step(action)
print("ds")
