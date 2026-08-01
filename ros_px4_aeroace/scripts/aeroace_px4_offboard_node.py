#!/usr/bin/env python3
"""Run AeroACE as a MAVROS/PX4 offboard attitude-thrust controller.

The node reuses the repository's AeroACE implementation for the outer-loop
force/attitude command and lets PX4 keep its onboard attitude/rate/motor loops.
All local-frame quantities are expected in MAVROS ENU/FLU conventions.
"""

import importlib
import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_CODE_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, os.pardir))
DEFAULT_CKPT = os.path.join(DEFAULT_CODE_DIR, "params", "aeroace_trained.pt")


@dataclass
class ReferenceState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    stamp: rospy.Time
    external: bool = False


def _vec3_from_param(name, default):
    value = rospy.get_param(name, default)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return np.asarray(value, dtype=np.float64)


def _quat_msg_to_wxyz(q_msg):
    q = np.asarray([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_wxyz_to_msg(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / norm
    out = Quaternion()
    out.w = float(q[0])
    out.x = float(q[1])
    out.y = float(q[2])
    out.z = float(q[3])
    return out


def _yaw_from_quat_msg(q_msg):
    q = _quat_msg_to_wxyz(q_msg)
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _fresh(stamp, timeout_s):
    return stamp is not None and (rospy.Time.now() - stamp).to_sec() <= timeout_s


class AeroACEPX4OffboardNode:
    def __init__(self):
        rospy.init_node("aeroace_px4_offboard")

        self.algorithm_code_dir = rospy.get_param("~algorithm_code_dir", "") or DEFAULT_CODE_DIR
        self.checkpoint = rospy.get_param("~checkpoint", "") or DEFAULT_CKPT
        self.mavros_ns = rospy.get_param("~mavros_ns", "/mavros").rstrip("/")
        self.control_rate = float(rospy.get_param("~control_rate", 20.0))
        self.reference_timeout = float(rospy.get_param("~reference_timeout", 0.5))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.5))
        self.imu_timeout = float(rospy.get_param("~imu_timeout", 0.5))
        self.velocity_timeout = float(rospy.get_param("~velocity_timeout", 0.5))

        self.target_position = _vec3_from_param("~target_position", [0.0, 0.0, 1.5])
        self.use_current_xy = bool(rospy.get_param("~use_current_xy", True))
        self.takeoff_altitude = float(rospy.get_param("~takeoff_altitude", 1.5))
        self.takeoff_ramp_s = max(float(rospy.get_param("~takeoff_ramp_s", 8.0)), 1e-3)
        self.default_yaw = float(rospy.get_param("~yaw", 0.0))

        self.accel_source = rospy.get_param("~accel_source", "imu")
        self.accel_lpf_alpha = float(np.clip(rospy.get_param("~accel_lpf_alpha", 0.35), 0.0, 1.0))
        self.imu_linear_accel_is_specific_force = bool(
            rospy.get_param("~imu_linear_accel_is_specific_force", True)
        )

        self.hover_thrust = float(rospy.get_param("~hover_thrust", 0.5))
        self.thrust_scale = float(rospy.get_param("~thrust_scale", 0.0))
        self.min_thrust = float(rospy.get_param("~min_thrust", 0.05))
        self.max_thrust = float(rospy.get_param("~max_thrust", 0.85))
        self.max_position_error = float(rospy.get_param("~max_position_error", 4.0))
        self.max_tilt_rad = float(rospy.get_param("~max_tilt_rad", 0.9))
        self.stop_setpoints_on_failsafe = bool(rospy.get_param("~stop_setpoints_on_failsafe", True))
        self.latch_failsafe = bool(rospy.get_param("~latch_failsafe", True))
        self.flight_log_path = os.path.expanduser(rospy.get_param("~flight_log_path", ""))

        self.auto_offboard = bool(rospy.get_param("~auto_offboard", False))
        self.auto_arm = bool(rospy.get_param("~auto_arm", False))
        self.arm_only_in_offboard = bool(rospy.get_param("~arm_only_in_offboard", True))
        self.offboard_mode = rospy.get_param("~offboard_mode", "OFFBOARD")
        self.setpoint_warmup_s = float(rospy.get_param("~setpoint_warmup_s", 3.0))
        self.mode_request_period_s = float(rospy.get_param("~mode_request_period_s", 5.0))
        self.online_update_configured = bool(
            rospy.get_param("~aero_online_update", False)
        )

        self.controller = self._load_controller()
        self.mass = float(self.controller.params["m"])
        self.gravity = float(self.controller.params["g"])
        self.ct = float(self.controller.params["C_T"])
        self.motor_min_speed = float(self.controller.params["motor_min_speed"])
        self.motor_max_speed = float(self.controller.params["motor_max_speed"])
        self.hover_motor_speed = math.sqrt(max(self.mass * self.gravity / (4.0 * self.ct), 0.0))

        self.state = State()
        self.last_odom = None
        self.last_odom_stamp = None
        self.last_imu = None
        self.last_imu_stamp = None
        self.last_velocity = None
        self.last_velocity_stamp = None
        self.prev_velocity = None
        self.prev_velocity_time = None
        self.accel_world_lpf = np.zeros(3, dtype=np.float64)
        self.home_position = None
        self.start_time = None
        self.last_thrust_n = self.mass * self.gravity
        self.external_reference = None
        self.failsafe = False
        self.flight_session_started = False
        self.current_flight_active = False
        self.setpoints_sent = 0
        self.last_mode_request = rospy.Time(0)
        self.flight_log_file = None
        self.flight_log_writer = None
        self._open_flight_log()
        rospy.on_shutdown(self._close_flight_log)

        self.att_pub = rospy.Publisher(
            f"{self.mavros_ns}/setpoint_raw/attitude",
            AttitudeTarget,
            queue_size=10,
        )
        self.debug_pub = rospy.Publisher("~debug", Float64MultiArray, queue_size=10)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

        rospy.Subscriber(f"{self.mavros_ns}/state", State, self._state_cb, queue_size=10)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", f"{self.mavros_ns}/local_position/odom"),
            Odometry,
            self._odom_cb,
            queue_size=20,
        )
        rospy.Subscriber(
            rospy.get_param("~velocity_topic", f"{self.mavros_ns}/local_position/velocity_local"),
            TwistStamped,
            self._velocity_cb,
            queue_size=20,
        )
        rospy.Subscriber(
            rospy.get_param("~imu_topic", f"{self.mavros_ns}/imu/data"),
            Imu,
            self._imu_cb,
            queue_size=50,
        )
        rospy.Subscriber(
            rospy.get_param("~reference_topic", "/aeroace/reference"),
            MultiDOFJointTrajectoryPoint,
            self._reference_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            rospy.get_param("~pose_reference_topic", "/aeroace/pose_reference"),
            PoseStamped,
            self._pose_reference_cb,
            queue_size=10,
        )

        self.set_mode_srv = rospy.ServiceProxy(f"{self.mavros_ns}/set_mode", SetMode)
        self.arm_srv = rospy.ServiceProxy(f"{self.mavros_ns}/cmd/arming", CommandBool)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self._control_timer)
        self._publish_status(
            f"AeroACE PX4 node ready; checkpoint={self.checkpoint}; code_dir={self.algorithm_code_dir}"
        )

    def _load_controller(self):
        if not os.path.isdir(self.algorithm_code_dir):
            raise RuntimeError(f"AeroACE code directory does not exist: {self.algorithm_code_dir}")
        if self.algorithm_code_dir not in sys.path:
            sys.path.insert(0, self.algorithm_code_dir)

        ctrl_module = importlib.import_module("controller")
        self.deployment_signal = importlib.import_module("aeroace_deployment_signal")
        seq_len = int(rospy.get_param("~aero_seq_len", 10))
        dict_kwargs = dict(
            dict_max_entries=int(rospy.get_param("~aero_dict_max_entries", 10000)),
            dict_normalize_keys=bool(rospy.get_param("~aero_dict_normalize_keys", False)),
            dict_temperature=float(rospy.get_param("~aero_dict_temperature", 1.0)),
            dict_min_cosine_distance=float(rospy.get_param("~aero_dict_min_cosine_distance", 0.01)),
            dict_max_entries_per_bucket=int(rospy.get_param("~aero_dict_max_entries_per_bucket", 1024)),
            dict_ema_alpha=float(rospy.get_param("~aero_dict_ema_alpha", 0.9)),
            online_update=self.online_update_configured,
            online_anomaly_similarity_threshold=float(
                rospy.get_param("~aero_online_anomaly_similarity_threshold", 0.9)
            ),
            online_min_anomaly_steps=int(rospy.get_param("~aero_online_min_anomaly_steps", 3)),
            online_warmup_steps=int(rospy.get_param("~aero_online_warmup_steps", 20)),
            online_update_interval_steps=int(
                rospy.get_param("~aero_online_update_interval_steps", 10)
            ),
            online_residual_window=int(rospy.get_param("~aero_online_residual_window", 5)),
            online_residual_consistency_threshold=float(
                rospy.get_param("~aero_online_residual_consistency_threshold", 6.0)
            ),
            online_force_clip_norm=float(
                rospy.get_param("~aero_online_force_clip_norm", 30.0)
            ),
            online_force_reject_norm=float(
                rospy.get_param("~aero_online_force_reject_norm", 60.0)
            ),
            online_require_anomaly=bool(
                rospy.get_param("~aero_online_require_anomaly", True)
            ),
        )
        ctrl = ctrl_module.AeroACE(given_pid=True, p=0.0, i=0.0, d=0.0, seq_len=seq_len, **dict_kwargs)
        if not os.path.isfile(self.checkpoint):
            raise RuntimeError(f"AeroACE checkpoint does not exist: {self.checkpoint}")
        ctrl.load(self.checkpoint, map_location="cpu")
        ctrl.state = "test"
        ctrl.reset_controller()
        if hasattr(ctrl, "reset_analysis_histories"):
            ctrl.reset_analysis_histories()
        if hasattr(ctrl, "set_dictionary_context"):
            ctrl.set_dictionary_context({"platform": "px4", "deployment": "real_world"})
        if hasattr(ctrl, "fgru"):
            ctrl.fgru.eval()
        return ctrl

    def _state_cb(self, msg):
        self.state = msg

    def _odom_cb(self, msg):
        self.last_odom = msg
        self.last_odom_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

    def _imu_cb(self, msg):
        self.last_imu = msg
        self.last_imu_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

    def _velocity_cb(self, msg):
        self.last_velocity = msg
        self.last_velocity_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

    def _reference_cb(self, msg):
        if not msg.transforms:
            return
        tr = msg.transforms[0].translation
        pd = np.asarray([tr.x, tr.y, tr.z], dtype=np.float64)
        vd = np.zeros(3, dtype=np.float64)
        ad = np.zeros(3, dtype=np.float64)
        if msg.velocities:
            lin = msg.velocities[0].linear
            vd = np.asarray([lin.x, lin.y, lin.z], dtype=np.float64)
        if msg.accelerations:
            lin = msg.accelerations[0].linear
            ad = np.asarray([lin.x, lin.y, lin.z], dtype=np.float64)
        yaw = self.default_yaw
        q = msg.transforms[0].rotation
        if abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) > 1e-9:
            yaw = _yaw_from_quat_msg(q)
        self.external_reference = ReferenceState(pd, vd, ad, yaw, rospy.Time.now(), external=True)

    def _pose_reference_cb(self, msg):
        p = msg.pose.position
        pd = np.asarray([p.x, p.y, p.z], dtype=np.float64)
        self.external_reference = ReferenceState(
            pd,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            _yaw_from_quat_msg(msg.pose.orientation),
            msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now(),
            external=True,
        )

    def _control_timer(self, _event):
        now = rospy.Time.now()
        if self.last_odom is None or not _fresh(self.last_odom_stamp, self.odom_timeout):
            rospy.logwarn_throttle(2.0, "Waiting for fresh MAVROS local odometry.")
            return

        if self.start_time is None:
            self.start_time = now
            p0 = self._position_from_odom(self.last_odom)
            self.home_position = p0.copy()
            if self.use_current_xy:
                self.target_position[0:2] = p0[0:2]

        flight_active = bool(
            self.state.armed
            and self.state.mode == self.offboard_mode
            and not self.failsafe
        )
        if flight_active and not self.flight_session_started:
            self.controller.reset_controller()
            if hasattr(self.controller, "reset_analysis_histories"):
                self.controller.reset_analysis_histories()
            if hasattr(self.controller, "set_dictionary_context"):
                self.controller.set_dictionary_context(
                    {"platform": "px4", "deployment": "real_world"}
                )
            self.flight_session_started = True
            rospy.loginfo("AeroACE flight session started; online update counters reset.")
        self.current_flight_active = flight_active
        self.controller.online_update_enabled = bool(
            self.online_update_configured and flight_active
        )

        X = self._build_aeroace_state(now)
        imu_acc_world = self._acceleration_world(now, X)
        reference = self._current_reference(now)
        previous_commanded_thrust_n = float(self.last_thrust_n)
        z_est = self._motor_speed_from_thrust(previous_commanded_thrust_n)
        residual_model_thrust_n = float(self.ct * np.sum(z_est ** 2))
        self.controller.motor_speed = z_est.copy()

        t = max((now - self.start_time).to_sec(), 0.0)
        try:
            thrust_n, q_cmd = self.controller.position(
                X=X,
                Z=z_est,
                imu=imu_acc_world,
                pd=reference.position,
                vd=reference.velocity,
                ad=reference.acceleration,
                last_wind_update=0.0,
                t=t,
                wind_gt=None,
            )
        except Exception as exc:
            rospy.logerr_throttle(1.0, f"AeroACE control step failed: {exc}")
            self._enter_failsafe("controller_exception")
            return

        thrust_n = float(thrust_n)
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(4)
        q_cmd = self._apply_reference_yaw(q_cmd, thrust_n, reference.yaw)
        thrust_norm = self._normalize_thrust(thrust_n)
        if not self._command_is_safe(X, reference, q_cmd, thrust_norm):
            if self.stop_setpoints_on_failsafe:
                return

        self.last_thrust_n = thrust_n
        self._publish_attitude_target(q_cmd, thrust_norm, now)
        self._publish_debug(t, X, reference, thrust_n, thrust_norm)
        self._write_flight_log(
            t,
            X,
            reference,
            imu_acc_world,
            previous_commanded_thrust_n,
            residual_model_thrust_n,
            thrust_n,
            thrust_norm,
        )
        self._maybe_request_offboard_and_arm(now)

    def _position_from_odom(self, odom):
        p = odom.pose.pose.position
        return np.asarray([p.x, p.y, p.z], dtype=np.float64)

    def _build_aeroace_state(self, now):
        odom = self.last_odom
        p = self._position_from_odom(odom)
        q = _quat_msg_to_wxyz(odom.pose.pose.orientation)
        if self.last_velocity is not None and _fresh(self.last_velocity_stamp, self.velocity_timeout):
            v_msg = self.last_velocity.twist.linear
        else:
            v_msg = odom.twist.twist.linear
        v = np.asarray([v_msg.x, v_msg.y, v_msg.z], dtype=np.float64)
        if self.last_imu is not None and _fresh(self.last_imu_stamp, self.imu_timeout):
            w_body = np.asarray(
                [
                    self.last_imu.angular_velocity.x,
                    self.last_imu.angular_velocity.y,
                    self.last_imu.angular_velocity.z,
                ],
                dtype=np.float64,
            )
        else:
            w_body = np.asarray(
                [
                    odom.twist.twist.angular.x,
                    odom.twist.twist.angular.y,
                    odom.twist.twist.angular.z,
                ],
                dtype=np.float64,
            )
        X = np.concatenate((p, q, v, w_body))
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _acceleration_world(self, now, X):
        if (
            self.accel_source == "imu"
            and self.last_imu is not None
            and _fresh(self.last_imu_stamp, self.imu_timeout)
        ):
            a_body = np.asarray(
                [
                    self.last_imu.linear_acceleration.x,
                    self.last_imu.linear_acceleration.y,
                    self.last_imu.linear_acceleration.z,
                ],
                dtype=np.float64,
            )
            a_world = self.deployment_signal.imu_acceleration_world(
                a_body,
                X[3:7],
                self.gravity,
                is_specific_force=self.imu_linear_accel_is_specific_force,
            )
        else:
            v = X[7:10]
            if self.prev_velocity is None or self.prev_velocity_time is None:
                raw = np.zeros(3, dtype=np.float64)
            else:
                dt = max((now - self.prev_velocity_time).to_sec(), 1e-3)
                raw = (v - self.prev_velocity) / dt
            self.prev_velocity = v.copy()
            self.prev_velocity_time = now
            a_world = raw

        if not np.all(np.isfinite(a_world)):
            a_world = np.zeros(3, dtype=np.float64)
        self.accel_world_lpf = self.deployment_signal.causal_low_pass(
            a_world,
            self.accel_world_lpf,
            self.accel_lpf_alpha,
        )
        return self.accel_world_lpf.copy()

    def _current_reference(self, now):
        if self.external_reference is not None and _fresh(self.external_reference.stamp, self.reference_timeout):
            return self.external_reference

        elapsed = max((now - self.start_time).to_sec(), 0.0)
        ramp = float(np.clip(elapsed / self.takeoff_ramp_s, 0.0, 1.0))
        pd = self.target_position.copy()
        vd = np.zeros(3, dtype=np.float64)
        ad = np.zeros(3, dtype=np.float64)
        if self.home_position is not None:
            pd[2] = self.home_position[2] + ramp * self.takeoff_altitude
            if ramp < 1.0:
                vd[2] = self.takeoff_altitude / self.takeoff_ramp_s
        return ReferenceState(pd, vd, ad, self.default_yaw, now, external=False)

    def _motor_speed_from_thrust(self, thrust_n):
        thrust_n = float(np.clip(thrust_n, 0.0, 4.0 * self.ct * self.motor_max_speed ** 2))
        omega = math.sqrt(max(thrust_n / max(4.0 * self.ct, 1e-12), 0.0))
        omega = float(np.clip(omega, self.motor_min_speed, self.motor_max_speed))
        return np.full(4, omega, dtype=np.float64)

    def _apply_reference_yaw(self, q_cmd, thrust_n, yaw):
        force_world = self.deployment_signal.rotation_matrix_from_wxyz(q_cmd) @ np.asarray(
            [0.0, 0.0, thrust_n], dtype=np.float64
        )
        if not np.all(np.isfinite(force_world)) or np.linalg.norm(force_world) < 1e-9:
            return q_cmd
        max_angle = float(self.controller.params.get("max_zenith_angle", math.pi / 4.0))
        return np.asarray(self.controller.get_q(force_world, yaw=float(yaw), max_angle=max_angle), dtype=np.float64)

    def _normalize_thrust(self, thrust_n):
        if self.thrust_scale > 0.0:
            thrust = self.thrust_scale * thrust_n
        else:
            thrust = self.hover_thrust * thrust_n / max(self.mass * self.gravity, 1e-6)
        return float(np.clip(thrust, self.min_thrust, self.max_thrust))

    def _command_is_safe(self, X, reference, q_cmd, thrust_norm):
        if self.latch_failsafe and self.failsafe:
            return False
        if not np.all(np.isfinite(q_cmd)) or not np.isfinite(thrust_norm):
            self._enter_failsafe("nonfinite_command")
            return False
        pos_error = float(np.linalg.norm(X[0:3] - reference.position))
        if pos_error > self.max_position_error:
            self._enter_failsafe(f"position_error_{pos_error:.2f}m")
            return False
        q = q_cmd / max(np.linalg.norm(q_cmd), 1e-12)
        tilt = math.acos(float(np.clip(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), -1.0, 1.0)))
        if tilt > self.max_tilt_rad:
            self._enter_failsafe(f"tilt_{tilt:.2f}rad")
            return False
        return True

    def _publish_attitude_target(self, q_cmd, thrust_norm, now):
        msg = AttitudeTarget()
        msg.header.stamp = now
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )
        msg.orientation = _quat_wxyz_to_msg(q_cmd)
        msg.thrust = float(thrust_norm)
        self.att_pub.publish(msg)
        self.setpoints_sent += 1

    def _publish_debug(self, t, X, reference, thrust_n, thrust_norm):
        msg = Float64MultiArray()
        gate = float(getattr(getattr(self.controller, "fgru", None), "last_g_t", float("nan")))
        residual = np.asarray(getattr(self.controller, "residual", np.zeros(3)), dtype=np.float64)
        online_report = (
            self.controller.get_online_update_report()
            if hasattr(self.controller, "get_online_update_report")
            else {}
        )
        msg.data = [
            float(t),
            *reference.position.astype(float).tolist(),
            *X[0:3].astype(float).tolist(),
            *reference.velocity.astype(float).tolist(),
            *X[7:10].astype(float).tolist(),
            float(thrust_n),
            float(thrust_norm),
            gate,
            *residual.astype(float).tolist(),
            float(bool(online_report.get("enabled", False))),
            float(online_report.get("dictionary_size", 0)),
            float(online_report.get("last_max_similarity", float("nan"))),
            float(online_report.get("last_residual_deviation", float("nan"))),
            float(online_report.get("candidates", 0)),
            float(online_report.get("accepted", 0)),
        ]
        self.debug_pub.publish(msg)

    def _open_flight_log(self):
        if not self.flight_log_path:
            return
        parent = os.path.dirname(os.path.abspath(self.flight_log_path))
        os.makedirs(parent, exist_ok=True)
        self.flight_log_file = open(self.flight_log_path, "w", newline="")
        fieldnames = [
            "t",
            "flight_active",
            "online_update_configured",
            "online_update_enabled",
            "mass_kg", "gravity_m_s2",
            "anomaly_similarity_threshold", "force_clip_norm_n",
            "force_reject_norm_n", "residual_consistency_threshold_n",
            "pd_x", "pd_y", "pd_z",
            "p_x", "p_y", "p_z",
            "vd_x", "vd_y", "vd_z",
            "v_x", "v_y", "v_z",
            "q_w", "q_x", "q_y", "q_z",
            "omega_x", "omega_y", "omega_z",
            "accel_world_x", "accel_world_y", "accel_world_z",
            "previous_commanded_thrust_n", "residual_model_thrust_n",
            "thrust_n", "thrust_norm",
            "residual_x", "residual_y", "residual_z",
            "update_force_x", "update_force_y", "update_force_z",
            "max_similarity", "residual_deviation", "update_status",
            "dictionary_size", "protected_dictionary_size",
            "candidates", "accepted", "inserted", "merged",
            "rejected_nonfinite", "rejected_force_norm", "rejected_inconsistent",
            "rejected_warmup", "rejected_anomaly_persistence",
            "rejected_rate_limit", "rejected_capacity",
        ]
        self.flight_log_writer = csv.DictWriter(self.flight_log_file, fieldnames=fieldnames)
        self.flight_log_writer.writeheader()
        self.flight_log_file.flush()

    def _write_flight_log(
        self,
        t,
        X,
        reference,
        accel_world,
        previous_commanded_thrust_n,
        residual_model_thrust_n,
        thrust_n,
        thrust_norm,
    ):
        if self.flight_log_writer is None:
            return
        report = (
            self.controller.get_online_update_report()
            if hasattr(self.controller, "get_online_update_report")
            else {}
        )
        residual = np.asarray(getattr(self.controller, "residual", np.zeros(3)), dtype=float)
        update_force = np.asarray(
            getattr(self.controller, "online_last_force_used", np.zeros(3)), dtype=float
        )
        accel_world = np.asarray(accel_world, dtype=float).reshape(3)
        row = {
            "t": float(t),
            "flight_active": int(self.current_flight_active),
            "online_update_configured": int(self.online_update_configured),
            "online_update_enabled": int(bool(report.get("enabled", False))),
            "mass_kg": float(self.mass),
            "gravity_m_s2": float(self.gravity),
            "anomaly_similarity_threshold": float(
                self.controller.online_anomaly_similarity_threshold
            ),
            "force_clip_norm_n": float(self.controller.online_force_clip_norm),
            "force_reject_norm_n": float(self.controller.online_force_reject_norm),
            "residual_consistency_threshold_n": float(
                self.controller.online_residual_consistency_threshold
            ),
            "pd_x": float(reference.position[0]),
            "pd_y": float(reference.position[1]),
            "pd_z": float(reference.position[2]),
            "p_x": float(X[0]),
            "p_y": float(X[1]),
            "p_z": float(X[2]),
            "vd_x": float(reference.velocity[0]),
            "vd_y": float(reference.velocity[1]),
            "vd_z": float(reference.velocity[2]),
            "v_x": float(X[7]),
            "v_y": float(X[8]),
            "v_z": float(X[9]),
            "q_w": float(X[3]),
            "q_x": float(X[4]),
            "q_y": float(X[5]),
            "q_z": float(X[6]),
            "omega_x": float(X[10]),
            "omega_y": float(X[11]),
            "omega_z": float(X[12]),
            "accel_world_x": float(accel_world[0]),
            "accel_world_y": float(accel_world[1]),
            "accel_world_z": float(accel_world[2]),
            "previous_commanded_thrust_n": float(previous_commanded_thrust_n),
            "residual_model_thrust_n": float(residual_model_thrust_n),
            "thrust_n": float(thrust_n),
            "thrust_norm": float(thrust_norm),
            "residual_x": float(residual[0]),
            "residual_y": float(residual[1]),
            "residual_z": float(residual[2]),
            "update_force_x": float(update_force[0]),
            "update_force_y": float(update_force[1]),
            "update_force_z": float(update_force[2]),
            "max_similarity": float(report.get("last_max_similarity", float("nan"))),
            "residual_deviation": float(
                report.get("last_residual_deviation", float("nan"))
            ),
            "update_status": str(report.get("last_status", "")),
            "dictionary_size": int(report.get("dictionary_size", 0)),
            "protected_dictionary_size": int(
                report.get("protected_dictionary_size", 0)
            ),
            "candidates": int(report.get("candidates", 0)),
            "accepted": int(report.get("accepted", 0)),
            "inserted": int(report.get("inserted", 0)),
            "merged": int(report.get("merged", 0)),
            "rejected_nonfinite": int(report.get("rejected_nonfinite", 0)),
            "rejected_force_norm": int(report.get("rejected_force_norm", 0)),
            "rejected_inconsistent": int(report.get("rejected_inconsistent", 0)),
            "rejected_warmup": int(report.get("rejected_warmup", 0)),
            "rejected_anomaly_persistence": int(
                report.get("rejected_anomaly_persistence", 0)
            ),
            "rejected_rate_limit": int(report.get("rejected_rate_limit", 0)),
            "rejected_capacity": int(report.get("rejected_capacity", 0)),
        }
        self.flight_log_writer.writerow(row)
        self.flight_log_file.flush()

    def _close_flight_log(self):
        if self.flight_log_file is not None:
            self.flight_log_file.flush()
            self.flight_log_file.close()
            self.flight_log_file = None
            self.flight_log_writer = None

    def _maybe_request_offboard_and_arm(self, now):
        if not self.state.connected:
            return
        if (now - self.start_time).to_sec() < self.setpoint_warmup_s:
            return
        if (now - self.last_mode_request).to_sec() < self.mode_request_period_s:
            return

        if self.auto_offboard and self.state.mode != self.offboard_mode:
            req = SetModeRequest()
            req.custom_mode = self.offboard_mode
            try:
                resp = self.set_mode_srv(req)
                if resp.mode_sent:
                    self._publish_status(f"Requested PX4 mode {self.offboard_mode}")
                self.last_mode_request = now
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, f"SetMode service call failed: {exc}")
            return

        can_arm = (not self.arm_only_in_offboard) or self.state.mode == self.offboard_mode
        if self.auto_arm and can_arm and not self.state.armed:
            req = CommandBoolRequest()
            req.value = True
            try:
                resp = self.arm_srv(req)
                if resp.success:
                    self._publish_status("Requested vehicle arm")
                self.last_mode_request = now
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, f"Arming service call failed: {exc}")

    def _enter_failsafe(self, reason):
        if not self.failsafe:
            self.failsafe = True
            rospy.logerr(f"AeroACE failsafe: {reason}")
            self._publish_status(f"failsafe: {reason}")

    def _publish_status(self, text):
        self.status_pub.publish(String(data=text))
        rospy.loginfo(text)

    def spin(self):
        rospy.spin()


def main():
    try:
        node = AeroACEPX4OffboardNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
