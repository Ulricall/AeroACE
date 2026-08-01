# ros_px4_aeroace

`ros_px4_aeroace` 是一个基于 `rospy + MAVROS` 的 PX4 实机运行适配包。节点复用本仓库 `controller.AeroACE` 的外环控制逻辑，发布 `/mavros/setpoint_raw/attitude`，由 PX4 执行底层姿态、角速度和电机闭环。

## 控制接口

- 输入状态：`/mavros/local_position/odom`、`/mavros/local_position/velocity_local`、`/mavros/imu/data`
- 输出控制：`mavros_msgs/AttitudeTarget`
- AeroACE 输出：期望总推力 `T` 和姿态四元数 `q`
- PX4 执行：OFFBOARD attitude/thrust setpoint

默认不会自动切 `OFFBOARD`，也不会自动解锁。先在 SITL 或拆桨状态下验证 topic、坐标系、推力标定和 failsafe。

## 使用

把 `code_sub/ros_px4_aeroace` 放在 catkin 工作空间的 `src` 下，或在 `src` 下建立符号链接，然后编译：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

启动 MAVROS 后运行：

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch
```

如果包不在本仓库 `code_sub/` 目录下，需要显式指定算法代码和 checkpoint：

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  algorithm_code_dir:=/path/to/AeroACE-PAMI/code_sub \
  checkpoint:=/path/to/AeroACE-PAMI/code_sub/params/aeroace_trained.pt
```

实机自动切模式/解锁必须显式开启：

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  auto_offboard:=true auto_arm:=true
```

## 参考轨迹

没有外部参考时，节点使用内置起飞/悬停参考：保持启动时的 `x/y`，在 `takeoff_ramp_s` 秒内上升到 `takeoff_altitude`。

也可以发布外部参考：

- `/aeroace/reference`：`trajectory_msgs/MultiDOFJointTrajectoryPoint`，支持位置、速度、加速度和 yaw
- `/aeroace/pose_reference`：`geometry_msgs/PoseStamped`，只给位置和 yaw，速度/加速度置零

外部参考超时后会回到内置参考，超时时间由 `reference_timeout` 控制。

## 关键标定参数

- `hover_thrust`：PX4 悬停归一化油门，通常对应 `MPC_THR_HOVER`
- `thrust_scale`：若大于 0，则直接用 `normalized_thrust = thrust_scale * thrust_newton`
- `min_thrust` / `max_thrust`：发送给 PX4 的归一化推力限幅
- `accel_source`：`velocity_derivative` 或 `imu`
- `max_position_error` / `max_tilt_rad`：运行时安全门限

## Online Expert Dictionary update

默认 `aero_online_update: false`，因此主体实验对应的推理过程不会修改 dictionary。实机测试 online update 时需要显式开启，并为每次飞行指定不同的 CSV 路径：

```bash
roslaunch ros_px4_aeroace aeroace_px4_offboard.launch \
  online_update:=true \
  flight_log_path:=/path/to/logs/online_01.csv
```

实机更新使用的力信号为
`m * a_world + m * g * e3 - T_cmd * R * e3`。默认从 IMU specific force 得到 `a_world`，完成机体系到世界系转换、重力去除和因果 LPF；`R` 来自里程计姿态，`T_cmd` 是上一控制周期发送给 PX4 的推力命令所对应的总推力。该路径不读取 ground-truth wind force 或外部力传感器。

只有低 dictionary similarity 连续出现时才允许候选更新。非有限值和超过硬阈值的残差会被拒绝；其余候选先做力范数裁剪，再与最近残差窗口的逐轴中位数比较，并受最小更新间隔限制。部署前加载的 dictionary entries 保持只读；通过检查后，近重复的在线 entries 之间使用 EMA 合并，否则追加新条目。dictionary 满后拒绝新条目，不执行 pruning、confidence weighting 或 forgetting。

`aeroace_online_update_experiment.launch` 同时启动控制器和可重复的 reference publisher。默认轨迹是平滑起飞后 hover；也可选择小幅 `circle` 或 `figure8`。static 和 online 飞行应使用完全相同的参数，只改变开关和日志路径：

```bash
roslaunch ros_px4_aeroace aeroace_online_update_experiment.launch \
  online_update:=false trajectory:=hover \
  flight_log_path:=/path/to/logs/static_01.csv

roslaunch ros_px4_aeroace aeroace_online_update_experiment.launch \
  online_update:=true trajectory:=hover \
  flight_log_path:=/path/to/logs/online_01.csv
```

建议按 static/online 交替顺序各运行五次，并保持 checkpoint、trajectory、reference 参数和测试区域一致。自动切换 OFFBOARD 和解锁仍然默认关闭。reference 轨迹和 online update 计数只在 UAV 已解锁且进入 OFFBOARD 后开始；切换前只发布当前位置用于 setpoint 预热。

每个 CSV 同时记录飞行状态、滤波后的世界系加速度、实测姿态、上一周期指令推力对应的模型推力、重构 residual、更新状态和累计拒绝计数。发布前应由操作者根据实机安全流程独立检查 static/online 开关、时间戳、累计计数和 dictionary 大小。

调试 topic：

- `~status`：状态和 failsafe 文本
- `~debug`：在原有状态、推力、gate 和 residual 后附加 online update 开关、dictionary 大小、similarity、residual deviation、候选数和接受数
