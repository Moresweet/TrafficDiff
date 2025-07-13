import carla
import random
import time

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
# 加载Town04地图
world = client.load_world('Town04')

# 设置同步模式
settings = world.get_settings()
settings.synchronous_mode = True  # 同步模式
settings.fixed_delta_seconds = 0.05  # 时间步长
world.apply_settings(settings)
map = world.get_map()

# 获取车辆蓝图
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[1]  # 选择一辆车

# 获取观众（视角）对象
spectator = world.get_spectator()
# 生成主车位置
spawn_point = carla.Transform(carla.Location(x=0, y=13.43, z=11.19), carla.Rotation(pitch=0.2, yaw=-180, roll=0.0))
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

vehicle_location = vehicle.get_location()
spectator_transform = carla.Transform(
            carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 100),  # 上方100米
            carla.Rotation(pitch=-90)  # 视角向下看，pitch设为-90度
        )
# 设置观众的位置
spectator.set_transform(spectator_transform)
def generate_random_spawn_point(x_range, y_range, z, pitch, yaw, roll):
    """
    根据给定的范围生成一个随机的车辆生成点。
    """
    x = random.uniform(x_range[0], x_range[1])  # 在指定的x范围内随机生成
    y = random.uniform(y_range[0], y_range[1])  # 在指定的y范围内随机生成
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

# 参数定义
x_range = (-350, 350)
y_range = (6, 16)
z = 11.19
pitch = 0.2
yaw = -180
roll = 0.0
num_vehicles = 40  # 需要生成的车辆数量
npc_bp = blueprint_library.filter('vehicle.*')[0]
# 自动驾驶车辆生成
npc_vehicles = []
for _ in range(num_vehicles):
    spawn_point = generate_random_spawn_point(x_range, y_range, z, pitch, yaw, roll)  # 随机生成位置

    vehicle_npc = world.try_spawn_actor(npc_bp, spawn_point)
    if vehicle_npc is not None:
        vehicle_npc.set_autopilot(True)  # 开启自动驾驶
        npc_vehicles.append(vehicle_npc)

print(f"成功生成了{len(npc_vehicles)}辆自动驾驶车辆。")

# 运行模拟器
try:
    while True:
        world.tick()  # 推动仿真世界前进
        time.sleep(0.05)
finally:
    # 清除车辆和传感器
    vehicle.destroy()
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    # 退出同步模式
    settings.synchronous_mode = False
    world.apply_settings(settings)
