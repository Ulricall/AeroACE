# ros_px4_aeroace

A `rospy + MAVROS` adapter package for running AeroACE on a PX4 vehicle. The node
reuses the outer-loop control logic of `controller.AeroACE` from this repository
and publishes `/mavros/setpoint_raw/attitude`; PX4 closes the inner attitude,
rate, and motor loops.

## Control interface

- State input: `/mavros/local_position/odom`, `/mavros/local_position/velocity_local`, `/mavros/imu/data`
- Control output: `mavros_msgs/AttitudeTarget`
- AeroACE output: desired total thrust `T` and attitude quaternion `q`
- PX4 execution: OFFBOARD attitude/thrust setpoint

The node does not switch to `OFFBOARD` or arm automatically by default. Verify
topics, frames, thrust calibration, and failsafe in SITL or with the propellers
removed before flying.

## Build

Place `code_sub/ros_px4_aeroace` under the `src` directory of a catkin workspace,
or symlink it there, then build:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Run

Start MAVROS, then launch the node:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch
```

If the package is not located under this repository's `code_sub/` directory,
specify the algorithm code and checkpoint paths explicitly:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  algorithm_code_dir:=/path/to/AeroACE-PAMI/code_sub \
  checkpoint:=/path/to/AeroACE-PAMI/code_sub/params/aeroace_trained.pt
```

Automatic mode switching and arming must be enabled explicitly:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  auto_offboard:=true auto_arm:=true
```

## Reference trajectory

With no external reference, the node uses a built-in takeoff/hover reference: it
holds the `x/y` position recorded at startup and climbs to `takeoff_altitude`
over `takeoff_ramp_s` seconds.

External references can be published on:

- `/aeroace/reference`: `trajectory_msgs/MultiDOFJointTrajectoryPoint`, supporting position, velocity, acceleration, and yaw
- `/aeroace/pose_reference`: `geometry_msgs/PoseStamped`, position and yaw only, with velocity and acceleration set to zero

The node falls back to the built-in reference after an external reference times
out, as controlled by `reference_timeout`.

## Calibration parameters

- `hover_thrust`: PX4 normalized hover throttle, normally matching `MPC_THR_HOVER`
- `thrust_scale`: if greater than 0, use `normalized_thrust = thrust_scale * thrust_newton` directly
- `min_thrust` / `max_thrust`: normalized thrust limits sent to PX4
- `accel_source`: `velocity_derivative` or `imu`
- `max_position_error` / `max_tilt_rad`: runtime safety thresholds

## Online Expert Dictionary update

`aero_online_update` defaults to `false`, so inference does not modify the
dictionary. To test online update on hardware, enable it explicitly and give each
flight a separate CSV path:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  online_update:=true \
  flight_log_path:=/path/to/logs/online_01.csv
```

The force signal used for onboard updates is
`m * a_world + m * g * e3 - T_cmd * R * e3`. By default `a_world` is derived from
IMU specific force, with body-to-world rotation, gravity removal, and a causal
LPF; `R` comes from the odometry attitude, and `T_cmd` is the total thrust
corresponding to the thrust command sent to PX4 in the previous control cycle.
This path does not read ground-truth wind force or an external force sensor.

A candidate update is allowed only when low dictionary similarity occurs on
consecutive steps. Non-finite residuals and residuals above a hard threshold are
rejected; remaining candidates are clipped by force norm, compared against the
per-axis median of a recent residual window, and subject to a minimum update
interval. Dictionary entries loaded before deployment remain read-only. Once a
candidate passes, near-duplicate online entries are merged with an EMA, otherwise
a new entry is appended. New entries are rejected once the dictionary is full; no
pruning, confidence weighting, or forgetting is performed.

`aeroace_online_update_experiment.launch` starts the controller together with a
repeatable reference publisher. The default trajectory is a smooth takeoff
followed by hover; small `circle` and `figure8` trajectories are also available.
Static and online flights should use identical parameters, changing only the
switch and the log path:

```bash
roslaunch ros_px4_aeroace aeroace_online_update_experiment.launch \
  online_update:=false trajectory:=hover \
  flight_log_path:=/path/to/logs/static_01.csv

roslaunch ros_px4_aeroace aeroace_online_update_experiment.launch \
  online_update:=true trajectory:=hover \
  flight_log_path:=/path/to/logs/online_01.csv
```

Automatic OFFBOARD switching and arming remain disabled by default. The reference
trajectory and the online-update counters start only after the UAV is armed and
has entered OFFBOARD; before the switch, the node publishes only the current
position to prime the setpoint stream.

Each CSV logs flight state, filtered world-frame acceleration, measured attitude,
the model thrust corresponding to the previous cycle's commanded thrust, the
reconstructed residual, update status, and cumulative rejection counts.

Debug topics:

- `~status`: state and failsafe text
- `~debug`: state, thrust, gate, and residual, followed by the online-update flag, dictionary size, similarity, residual deviation, and candidate/accepted counts
