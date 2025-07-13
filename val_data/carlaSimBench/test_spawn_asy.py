import carla
import time

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 加载Town04地图
client.load_world('Town04')
world = client.get_world()

# 设置同步模式
settings = world.get_settings()
settings.synchronous_mode = True  # 启用同步模式
settings.fixed_delta_seconds = 0.1  # 设置每个步骤的固定时间步长
world.apply_settings(settings)

# 设置车辆生成的地点
spawn_point = carla.Transform(carla.Location(x=0, y=31.43, z=11.2))

# 获取蓝图库并选择车辆蓝图
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[1]  # 选择第一个车辆蓝图

# 获取spectator对象
spectator = world.get_spectator()

# 开始循环10次生成和销毁车辆
for i in range(10):
    # 在指定位置生成车辆
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print(f"第{i + 1}次生成车辆成功，id: {vehicle.id}")
    else:
        print(f"第{i + 1}次生成车辆失败")
        continue

    # 设置spectator视角在车辆上方俯视
    spectator_transform = carla.Transform(
        carla.Location(x=0, y=31.43, z=20),  # 设置spectator在车辆上方20米处
        carla.Rotation(pitch=-90, yaw=0, roll=0)  # 俯视角度
    )

    # 更新spectator的位置和方向
    spectator.set_transform(spectator_transform)

    # 在同步模式下推进一步
    world.tick()

    # 等待2秒，模拟车辆存在的时间
    time.sleep(2)

    # 销毁车辆
    if vehicle is not None:
        vehicle.destroy()
        print(f"第{i + 1}次销毁车辆成功")

    # 在同步模式下推进一步
    world.tick()

    # 等待1秒再进行下一次循环
    time.sleep(1)

# 恢复到异步模式
settings.synchronous_mode = False
world.apply_settings(settings)
