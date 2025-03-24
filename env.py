import numpy as np
import gym
from gym import spaces
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging
from beamngpy.sensors import Camera, Lidar, Electrics, Damage, AdvancedIMU

'''
env needs to be able to

env.make()
env.reset()
env.step(): output should be next_state, reward, done, info

'''

class BeamNGEnv(gym.Env):

    def __init__(self, beamng_home, beamng_user, port):
        super().__init__()

        self.beamng = BeamNGpy(beamng_home, beamng_user, port=port)
        self.beamng.settings.set_deterministic(60)
        self.vehicle = None
        self.scenario = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), #steering, throttle, brake
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.image_width = 320
        self.image_height = 240
        self.lidar_points = 512

        self.spawn_point = (-717.121, 101, 118.675)
        # camera, lidar, vehicle state

        self.observation_space = spaces.Dict({
            'camera': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8
            ),
            'lidar': spaces.Box(
                low=0,
                high=100,
                shape=(self.lidar_points, 3),
                dtype=np.float32
            ),
            # Vehicle state: velocity (3), position (3), rotation (3), steering (1), 
            # throttle (1), brake (1), damage (1)
            'state': spaces.Box(
                low=np.array([-100, -100, -100, -10000, -10000, -10000, -180, -180, -180, -1, 0, 0, 0]),
                high=np.array([100, 100, 100, 10000, 10000, 10000, 180, 180, 180, 1, 1, 1, np.Inf]),
                dtype=np.float32
            )
        })

        self.camera_data = None
        self.lidar_data = None
        self.electrics_data = None
        self.damage_data = None
        self.imu_data = None
        
        # Episode settings
        self.max_episode_steps = 1000
        self.current_step = 0
        self.done = False

    def _setup_scenario(self):
        self.scenario = Scenario('west_coast_usa', 'rl_training')
        self.vehicle = Vehicle('ego', model='etk800', licence='ego', color='Green')
        self.vehicle.set_shift_mode('arcade')

        # Add vehicle to scenario at a random spawn point
        
        self.scenario.add_vehicle(self.vehicle, pos=self.spawn_point, rot=(0, 0, 0, 0))

        self.lidar = Lidar('lidar', vehicle=self.vehicle, bng=self.beamng,
                          pos=(0, 0, 1.7),
                          vertical_resolution=16,
                          max_distance=120,
                          is_360_mode=True)
        
        self.camera = Camera('camera', vehicle=self.vehicle, bng=self.beamng,
                            pos=(0, -1, 1.7), # Position relative to vehicle
                            dir=(0, -1, 0), # Forward-facing
                            fov=90,
                            resolution=(self.image_width, self.image_height),
                            near_far_planes=(0.1, 500), # closest and farthest objects to render
                            colour=True)
        
        self.electrics = Electrics()
        
        # Damage sensor
        self.damage = Damage()
        
        # IMU sensor for acceleration, gyroscope
        self.imu = AdvancedIMU('imu', bng=self.beamng, vehicle=self.vehicle, gfx_update_time=0.01, is_send_immediately=True)
        
        # TODO Add a road network for the AI to drive on
        # (You can define specific road networks or use built-in roads)
        
        # Compile the scenario and place it in BeamNG
        self.scenario.make(self.beamng)

    def reset(self):
        """Reset the environment to begin a new episode"""
        # Teleport to start location and reset
        self.vehicle.teleport(
            pos=self.spawn_point, 
            rot_quat=(0,0,0,0), 
            reset=True
        )
        
        # Poll sensors to get initial data
        self.vehicle.poll_sensors()
        
        # Get sensor data
        self.camera_data = self.camera.poll()
        self.lidar_data = self.lidar.poll()
        self.electrics_data = self.electrics.data
        self.damage_data = self.damage.data
        self.imu_data = self.imu.poll()
        
        # Reset episode variables
        self.current_step = 0
        self.done = False
        
        # Return the initial observation
        return self._get_observation()
    
    # TODO
    # step fuction
    def step(self, action):
        steering, throttle, brake = action
        self.vehicle.control(throttle=float(throttle), steering=float(steering), brake=float(brake))

        # Step the simulation
        self.beamng.step(1)
        

        # Poll sensors to get data
        self.vehicle.poll_sensors()
        
        # Get sensor data
        self.camera_data = self.camera.poll()
        self.lidar_data = self.lidar.poll()
        self.electrics_data = self.electrics.data
        self.damage_data = self.damage.data
        self.imu_data = self.imu.poll()

        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            self.done = True

        if self.damage_data['damage'] > 100:
            self.done = True

        obversation = self._get_observation()
        reward = self._get_reward()
        info = self._get_info()

        return obversation, reward, self.done, info




