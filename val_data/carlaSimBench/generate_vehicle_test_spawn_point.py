import carla
import time

# 连接到CARLA模拟器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # 设置超时时间

# 获取模拟世界
world = client.get_world()

# 选择Town04地图
if world.get_map().name != "Town04":
    world = client.load_world("Town04")

# 获取蓝图库
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.*")[0]  # 选择第一个可用车辆

# 获取可用的车辆出生点
spawn_points = world.get_map().get_spawn_points()

if len(spawn_points) == 0:
    print("未找到出生点！")
else:
    # 选择第一个出生点
    spawn_point = spawn_points[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print(f"成功生成车辆：{vehicle.type_id} 在 {spawn_point.location}")

        # 获取观众（Spectator）对象
        spectator = world.get_spectator()

        # 持续更新观众位置，使其保持在车辆上方30米
        try:
            while True:
                transform = vehicle.get_transform()
                camera_location = transform.location + carla.Location(z=50)
                camera_rotation = carla.Rotation(pitch=-90)  # 俯视角度
                spectator.set_transform(carla.Transform(camera_location, camera_rotation))
                time.sleep(0.05)  # 控制更新频率，防止过载
        except KeyboardInterrupt:
            print("停止更新视角")