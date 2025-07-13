import carla
import random

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 加载Town04地图
world = client.load_world('Town04')

# 获取地图对象
map = world.get_map()

# 获取所有出生点
spawn_points = map.get_spawn_points()

# 随机选择一个出生点
# spawn_point = spawn_points[42]
for idx, spawn_point in enumerate(spawn_points):
    # 从蓝图库中选择车辆类型
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[1]  # 随机选择一个车辆

    # 生成车辆并将其放置在随机出生点
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle:
        print("Vehicle spawned successfully!")

        # 获取观众（视角）对象
        spectator = world.get_spectator()

        # 获取车辆的当前位置
        vehicle_location = vehicle.get_location()

        # 创建一个新的transform，将观众移动到车辆上方100米处
        spectator_transform = carla.Transform(
            carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 100),  # 上方100米
            carla.Rotation(pitch=-90)  # 视角向下看，pitch设为-90度
        )

        # 设置观众的位置
        spectator.set_transform(spectator_transform)

        print("Spectator view set to 100 meters above the vehicle.")
        vehicle.destroy()
    else:
        print("Failed to spawn vehicle.")
        vehicle.destroy()


