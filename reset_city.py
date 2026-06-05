import carla

HOST = "127.0.0.1"
PORT = 2000


def remove_all_vehicles(world):
    actors = world.get_actors()

    vehicles = actors.filter("vehicle.*")
    walkers = actors.filter("walker.*")
    controllers = actors.filter("controller.ai.walker")

    print(f"Found {len(vehicles)} vehicles")
    print(f"Found {len(walkers)} walkers")
    print(f"Found {len(controllers)} walker controllers")

    for actor in controllers:
        try:
            actor.stop()
        except:
            pass

    destroy_list = []

    destroy_list.extend(controllers)
    destroy_list.extend(walkers)
    destroy_list.extend(vehicles)

    for actor in destroy_list:
        try:
            actor.destroy()
        except Exception as e:
            print(e)

    print("City cleared")


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()

    remove_all_vehicles(world)


if __name__ == "__main__":
    main()