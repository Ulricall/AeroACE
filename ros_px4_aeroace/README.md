# ros_px4_aeroace

AeroACE runs on an onboard companion computer through `rospy + MAVROS`, while
PX4 receives attitude and thrust setpoints in `OFFBOARD` mode.

## Control interface

- State input: `/mavros/local_position/odom`, `/mavros/local_position/velocity_local`, `/mavros/imu/data`
- Control output: `/mavros/setpoint_raw/attitude` (`mavros_msgs/AttitudeTarget`)

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

The default reference is takeoff followed by hover. Switching to `OFFBOARD` and
arming are manual by default. To request both automatically:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  auto_offboard:=true auto_arm:=true
```

To enable the online Expert Dictionary update and save a flight log:

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  online_update:=true \
  flight_log_path:=/path/to/logs/online_01.csv
```
