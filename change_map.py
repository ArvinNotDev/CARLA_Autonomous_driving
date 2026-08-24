import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# List available maps
print(client.get_available_maps())

# Load a new map
world = client.load_world('Town10HD_Opt')