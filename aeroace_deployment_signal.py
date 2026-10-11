"""Pure numerical helpers for the AeroACE onboard residual-force signal."""

import numpy as np


def normalize_quaternion_wxyz(quaternion):
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(quaternion)) or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion must be finite and have nonzero norm")
    return quaternion / norm


def rotation_matrix_from_wxyz(quaternion):
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def imu_acceleration_world(
    linear_acceleration_body,
    attitude_wxyz,
    gravity_m_s2,
    is_specific_force=True,
):
    acceleration_body = np.asarray(linear_acceleration_body, dtype=float).reshape(3)
    if not np.all(np.isfinite(acceleration_body)):
        raise ValueError("IMU acceleration must be finite")
    acceleration_world = rotation_matrix_from_wxyz(attitude_wxyz) @ acceleration_body
    if is_specific_force:
        acceleration_world -= np.asarray([0.0, 0.0, float(gravity_m_s2)])
    return acceleration_world


def causal_low_pass(current, previous, alpha):
    current = np.asarray(current, dtype=float).reshape(3)
    previous = np.asarray(previous, dtype=float).reshape(3)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
        raise ValueError("Low-pass filter inputs must be finite")
    return alpha * current + (1.0 - alpha) * previous


def residual_force_from_onboard(
    acceleration_world,
    attitude_wxyz,
    thrust_n,
    mass_kg,
    gravity_m_s2,
):
    acceleration_world = np.asarray(acceleration_world, dtype=float).reshape(3)
    if not np.all(np.isfinite(acceleration_world)):
        raise ValueError("World-frame acceleration must be finite")
    body_z_world = rotation_matrix_from_wxyz(attitude_wxyz)[:, 2]
    gravity_force = np.asarray([0.0, 0.0, float(mass_kg) * float(gravity_m_s2)])
    return (
        float(mass_kg) * acceleration_world
        + gravity_force
        - float(thrust_n) * body_z_world
    )
