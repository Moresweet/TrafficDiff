import carla

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取当前世界和地图
world = client.get_world()
map = world.get_map()

# 获取所有waypoints，采样距离设置为1米（可以根据需要调整）
waypoints = map.generate_waypoints(1.0)

# 筛选出所有 road_id 为 1091 的 waypoints
target_road_id = 1091
road_waypoints = [wp for wp in waypoints if wp.road_id == target_road_id]

# 确保找到了目标road_id的waypoints
if not road_waypoints:
    print(f"No waypoints found for road_id {target_road_id}!")
else:
    print(f"Found {len(road_waypoints)} waypoints for road_id {target_road_id}.")

    # 遍历所有的waypoints，绘制红色线段
    for i in range(len(road_waypoints) - 1):
        start_waypoint = road_waypoints[i].transform.location
        end_waypoint = road_waypoints[i + 1].transform.location

        # 使用debug工具绘制线段，颜色为红色，持续时间为10秒
        world.debug.draw_line(
            start_waypoint,
            end_waypoint,
            thickness=0.1,  # 线的粗细
            color=carla.Color(255, 0, 0),  # 红色
            life_time=10.0  # 持续时间（秒）
        )
    print("Red lines drawn for all waypoints on road_id 1091.")
