## Env

1. Define the beamng environment. It should follow the gym interface

Observation space:
- Lidar
  - Lidar is like vertical beams. The more channels we have, the more "together" the vertical cuts are so we get a higher res point cloud.
- Camera sensors
- Vehicle state
  - Velocity (x,y,z)
  - Position (x,y,z)
  - Brake (0-1)
  - Throttle (0-1)
  - Steering (-1, 1)
  - Damage (0-1)
   
steering: Rotation of the steering wheel, from -1.0 to 1.0.
throttle: Intensity of the throttle, from 0.0 to 1.0.
brake: Intensity of the brake, from 0.0 to 1.0.