#######
# Setup
#######

import sys
import glob
import os

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import numpy as np
import pandas as pd
import argparse

client = carla.Client('localhost', 2000)
world = client.get_world()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', "--label", help='label of this run of data output')
args = parser.parse_args()

###################
# Spawn ego vehicle
###################

ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name','ego')

ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color',ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

# Find spawn point and spawn
if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
    print('\nEgo is spawned')
else: 
    logging.warning('Could not find any spawn points')

############################
# Attatch sensors to vehicle
############################

# RGB front facing camera
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x',str(320))
cam_bp.set_attribute('image_size_y',str(240))
cam_bp.set_attribute('fov',str(105))
cam_location = carla.Location(2,0,3) # On the front hood
cam_rotation = carla.Rotation(0,0,0) # Facing forward
cam_transform = carla.Transform(cam_location, cam_rotation)
ego_cam = world.spawn_actor(cam_bp, 
        cam_transform, 
        attach_to=ego_vehicle, 
        attachment_type=carla.AttachmentType.Rigid)
ego_cam.listen(lambda image: image.save_to_disk(f'output/{args.label}/rgb_%.6d.jpg' % image.frame))

# Front facing depth camera
depth_cam = None
depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x',str(320))
depth_bp.set_attribute('image_size_y',str(240))
depth_location = carla.Location(2,0,3) # On the front hood
depth_rotation = carla.Rotation(0,0,0) # Front facing
depth_transform = carla.Transform(depth_location,depth_rotation)
depth_cam = world.spawn_actor(depth_bp,depth_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
depth_cam.listen(lambda image: image.save_to_disk(f'output/{args.label}/depth_%.6d.jpg' % image.frame,carla.ColorConverter.LogarithmicDepth))

################
# Start recoding
################

# client.start_recorder('./recording.log')

ego_vehicle.set_autopilot(True)
print('\nAutopilot recording...')

try:
    controls = np.empty(4)
    while True:
        world_snapshot = world.wait_for_tick()
        vehicle = ego_vehicle.get_control() # Get throttle, steer, brake etc
        controls = np.vstack([controls, 
            np.array([world_snapshot.frame,
                vehicle.steer,
                vehicle.throttle,
                vehicle.brake])])
except KeyboardInterrupt:
    pass
except RuntimeError:
    print('\nRuntimeError: time-out of 10000ms while waiting for the simulator')
    pass

controls = np.delete(controls, (0), axis=0)
np.save(f'output/{args.label}/controls', controls)
print(f'\nObservations collected: {controls.shape[0]}')
