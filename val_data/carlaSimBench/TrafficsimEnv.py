import carla
import random
import time


def is_spawn_point_valid(spawn_point, vehicles):
    for vehicle in vehicles:
        if vehicle.get_location().distance(spawn_point.location) < 2.0:  # 检查距离
            return False
    return True


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager()

    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settingdss(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicles_list = []
    num_vehicles = 30  # 车辆数量

    for _ in range(num_vehicles):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        valid_spawn = False

        while not valid_spawn:
            spawn_point = random.choice(spawn_points)
            if is_spawn_point_valid(spawn_point, vehicles_list):
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles_list.append(vehicle)
                valid_spawn = True

    try:
        for _ in range(200):  # 运行200帧
            world.tick()
            time.sleep(0.05)
    finally:
        for vehicle in vehicles_list:
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == '__main__':
    main()
