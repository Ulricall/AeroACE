#!/usr/bin/env python3
"""Publish reproducible position, velocity, and acceleration references for PX4 tests."""

import math

import numpy as np
import rospy
from geometry_msgs.msg import Transform, Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint


def _smoothstep5(t, duration):
    if duration <= 0.0 or t >= duration:
        return 1.0, 0.0, 0.0
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    u = t / duration
    scale = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
    scale_dot = (30.0 * u ** 2 - 60.0 * u ** 3 + 30.0 * u ** 4) / duration
    scale_ddot = (60.0 * u - 180.0 * u ** 2 + 120.0 * u ** 3) / duration ** 2
    return scale, scale_dot, scale_ddot


class ReferenceTrajectoryPublisher:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        self.reference_topic = rospy.get_param("~reference_topic", "/aeroace/reference")
        self.state_topic = rospy.get_param("~state_topic", "/mavros/state")
        self.offboard_mode = str(rospy.get_param("~offboard_mode", "OFFBOARD"))
        self.start_on_offboard = bool(rospy.get_param("~start_on_offboard", True))
        self.rate_hz = max(float(rospy.get_param("~rate", 20.0)), 1.0)
        self.trajectory = str(rospy.get_param("~trajectory", "hover")).lower()
        self.altitude = float(rospy.get_param("~altitude", 1.5))
        self.takeoff_s = max(float(rospy.get_param("~takeoff_s", 8.0)), 0.0)
        self.hold_s = max(float(rospy.get_param("~hold_s", 5.0)), 0.0)
        self.motion_ramp_s = max(float(rospy.get_param("~motion_ramp_s", 4.0)), 0.0)
        self.motion_duration_s = float(rospy.get_param("~motion_duration_s", 60.0))
        self.amplitude_x = float(rospy.get_param("~amplitude_x", 0.5))
        self.amplitude_y = float(rospy.get_param("~amplitude_y", 0.5))
        self.period_s = max(float(rospy.get_param("~period_s", 12.0)), 1e-3)
        self.yaw = float(rospy.get_param("~yaw", 0.0))
        if self.trajectory not in ("hover", "circle", "figure8"):
            raise ValueError(
                f"Unsupported trajectory '{self.trajectory}'; use hover, circle, or figure8"
            )

        self.origin = None
        self.start_time = None
        self.vehicle_state = State()
        self.publisher = rospy.Publisher(
            self.reference_topic,
            MultiDOFJointTrajectoryPoint,
            queue_size=10,
        )
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.state_topic, State, self._state_cb, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._timer_cb)

    def _odom_cb(self, message):
        if self.origin is not None:
            return
        p = message.pose.pose.position
        self.origin = np.asarray([p.x, p.y, p.z], dtype=float)
        rospy.loginfo(
            "AeroACE reference origin set to [%.3f, %.3f, %.3f]",
            self.origin[0],
            self.origin[1],
            self.origin[2],
        )

    def _state_cb(self, message):
        self.vehicle_state = message

    def _horizontal_reference(self, t):
        if self.trajectory == "hover":
            return np.zeros(3), np.zeros(3), np.zeros(3)

        omega = 2.0 * math.pi / self.period_s
        if self.trajectory == "circle":
            position = np.asarray([
                self.amplitude_x * (math.cos(omega * t) - 1.0),
                self.amplitude_y * math.sin(omega * t),
                0.0,
            ])
            velocity = np.asarray([
                -self.amplitude_x * omega * math.sin(omega * t),
                self.amplitude_y * omega * math.cos(omega * t),
                0.0,
            ])
            acceleration = np.asarray([
                -self.amplitude_x * omega ** 2 * math.cos(omega * t),
                -self.amplitude_y * omega ** 2 * math.sin(omega * t),
                0.0,
            ])
        else:
            position = np.asarray([
                self.amplitude_x * math.sin(omega * t),
                self.amplitude_y * math.sin(2.0 * omega * t),
                0.0,
            ])
            velocity = np.asarray([
                self.amplitude_x * omega * math.cos(omega * t),
                2.0 * self.amplitude_y * omega * math.cos(2.0 * omega * t),
                0.0,
            ])
            acceleration = np.asarray([
                -self.amplitude_x * omega ** 2 * math.sin(omega * t),
                -4.0 * self.amplitude_y * omega ** 2 * math.sin(2.0 * omega * t),
                0.0,
            ])

        scale, scale_dot, scale_ddot = _smoothstep5(t, self.motion_ramp_s)
        scaled_position = scale * position
        scaled_velocity = scale_dot * position + scale * velocity
        scaled_acceleration = (
            scale_ddot * position + 2.0 * scale_dot * velocity + scale * acceleration
        )
        return scaled_position, scaled_velocity, scaled_acceleration

    def _reference(self, elapsed):
        takeoff_scale, takeoff_velocity_scale, takeoff_acceleration_scale = _smoothstep5(
            elapsed, self.takeoff_s
        )
        position = self.origin + np.asarray([0.0, 0.0, self.altitude * takeoff_scale])
        velocity = np.asarray([0.0, 0.0, self.altitude * takeoff_velocity_scale])
        acceleration = np.asarray([0.0, 0.0, self.altitude * takeoff_acceleration_scale])
        if elapsed < self.takeoff_s + self.hold_s:
            return position, velocity, acceleration

        motion_t = elapsed - self.takeoff_s - self.hold_s
        if self.motion_duration_s > 0.0:
            motion_t = min(motion_t, self.motion_duration_s)
        horizontal_position, horizontal_velocity, horizontal_acceleration = (
            self._horizontal_reference(motion_t)
        )
        position = self.origin + np.asarray([0.0, 0.0, self.altitude])
        position += horizontal_position
        if self.motion_duration_s > 0.0 and motion_t >= self.motion_duration_s:
            return position, np.zeros(3), np.zeros(3)
        return position, horizontal_velocity, horizontal_acceleration

    def _timer_cb(self, _event):
        if self.origin is None:
            rospy.logwarn_throttle(2.0, "AeroACE reference publisher is waiting for odometry.")
            return
        now = rospy.Time.now()
        ready = bool(
            not self.start_on_offboard
            or (
                self.vehicle_state.armed
                and self.vehicle_state.mode == self.offboard_mode
            )
        )
        if self.start_time is None and ready:
            self.start_time = now
            rospy.loginfo("AeroACE reference trajectory started.")
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = max((now - self.start_time).to_sec(), 0.0)
        position, velocity, acceleration = self._reference(elapsed)

        transform = Transform()
        transform.translation.x = float(position[0])
        transform.translation.y = float(position[1])
        transform.translation.z = float(position[2])
        transform.rotation.z = math.sin(0.5 * self.yaw)
        transform.rotation.w = math.cos(0.5 * self.yaw)

        velocity_message = Twist()
        velocity_message.linear.x = float(velocity[0])
        velocity_message.linear.y = float(velocity[1])
        velocity_message.linear.z = float(velocity[2])

        acceleration_message = Twist()
        acceleration_message.linear.x = float(acceleration[0])
        acceleration_message.linear.y = float(acceleration[1])
        acceleration_message.linear.z = float(acceleration[2])

        message = MultiDOFJointTrajectoryPoint()
        message.transforms = [transform]
        message.velocities = [velocity_message]
        message.accelerations = [acceleration_message]
        message.time_from_start = rospy.Duration(elapsed)
        self.publisher.publish(message)


def main():
    rospy.init_node("aeroace_reference_trajectory")
    ReferenceTrajectoryPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
