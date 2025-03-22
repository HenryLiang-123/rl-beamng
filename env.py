import numpy as np
import gym
from gym import spaces
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging
from beamngpy.sensors import Camera, Lidar, Electrics, Damage, IMU

class BeamNGEnv(gym.Env):

    def __init__(self, beamng_home, beamng_user, port):
        super(BeamNGEnv, self).__init__()

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
                high=np.array([100, 100, 100, 10000, 10000, 10000, 180, 180, 180, 1, 1, 1, 1]),
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

    def setup_scenario(self):
        self.scenario = Scenario('west_coast_usa', 'rl_training')
        self.vehicle = Vehicle('ego', model='etk800', licence='ego', color='Green')
        self.vehicle.set_shift_mode('arcade')



        self.lidar = Lidar('lidar', self.vehicle,
                          pos=(0, 0, 1.7),
                          direction=(0, 0, 1),
                          vertical_resolution=16,
                          max_distance=100,
                          is_360_mode=True)
        
        self.camera = Camera('camera', self.vehicle, 
                            pos=(0, -1, 1.7), # Position relative to vehicle
                            direction=(0, 1, 0), # Forward-facing
                            fov=90,
                            resolution=(self.image_width, self.image_height),
                            near_far_planes=(0.1, 500), # closest and farthest objects to render
                            colour=True)


        self.electrics = Electrics('electrics')
        
        # Damage sensor
        self.damage = Damage('damage')
        
        # IMU sensor for acceleration, gyroscope
        self.imu = IMU('imu')
        
        # Add vehicle to scenario at a random spawn point
        spawn_point = np.random.choice([
            (-717.121, 101, 118.675),
            (-717.121, 101, 118.675),
            (-717.121, 101, 118.675)
        ])
        self.scenario.add_vehicle(self.vehicle, pos=spawn_point, rot=(0, 0, 0))
        
        # Add a road network for the AI to drive on
        # (You can define specific road networks or use built-in roads)
        
        # Compile the scenario and place it in BeamNG
        self.scenario.make(self.beamng)

    def reset(self):
        """Reset the environment to begin a new episode"""
        # Connect to BeamNG if not already connected
        self.beamng.open()
        
        # Set up a new scenario
        if self.scenario is not None:
            self.scenario.close()
        self._setup_scenario()
        
        # Load the scenario
        self.beamng.load_scenario(self.scenario)
        self.beamng.start_scenario()
        
        # Pause the simulation initially
        self.beamng.pause()
        
        # Reset sensors and data
        self.vehicle.attach_sensor('camera', self.camera)
        self.vehicle.attach_sensor('lidar', self.lidar)
        self.vehicle.attach_sensor('electrics', self.electrics)
        self.vehicle.attach_sensor('damage', self.damage)
        self.vehicle.attach_sensor('imu', self.imu)
        
        # Poll sensors to get initial data
        self.vehicle.poll_sensors()
        
        # Get sensor data
        self.camera_data = self.camera.data
        self.lidar_data = self.lidar.data
        self.electrics_data = self.electrics.data
        self.damage_data = self.damage.data
        self.imu_data = self.imu.data
        
        # Reset episode variables
        self.current_step = 0
        self.done = False
        
        # Return the initial observation
        return self._get_observation()