import math
import unittest

import numpy as np

import aeroace_deployment_signal as signal


class AeroACEDeploymentSignalTest(unittest.TestCase):
    def test_hover_specific_force_becomes_zero_world_acceleration(self):
        gravity = 9.81
        acceleration = signal.imu_acceleration_world(
            [0.0, 0.0, gravity],
            [1.0, 0.0, 0.0, 0.0],
            gravity,
            is_specific_force=True,
        )
        np.testing.assert_allclose(acceleration, np.zeros(3), atol=1e-12)

    def test_specific_force_rotation_uses_measured_attitude(self):
        gravity = 9.81
        half_angle = 0.25 * math.pi
        attitude = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]
        acceleration = signal.imu_acceleration_world(
            [-gravity, 0.0, 0.0],
            attitude,
            gravity,
            is_specific_force=True,
        )
        np.testing.assert_allclose(acceleration, np.zeros(3), atol=1e-12)

    def test_causal_low_pass_uses_configured_current_sample_weight(self):
        filtered = signal.causal_low_pass(
            current=[2.0, -1.0, 4.0],
            previous=[0.0, 1.0, 2.0],
            alpha=0.35,
        )
        np.testing.assert_allclose(filtered, [0.7, 0.3, 2.7], atol=1e-12)

    def test_residual_force_uses_world_acceleration_gravity_and_thrust(self):
        residual = signal.residual_force_from_onboard(
            acceleration_world=[1.0, -2.0, 0.5],
            attitude_wxyz=[1.0, 0.0, 0.0, 0.0],
            thrust_n=9.81,
            mass_kg=1.0,
            gravity_m_s2=9.81,
        )
        np.testing.assert_allclose(residual, [1.0, -2.0, 0.5], atol=1e-12)

    def test_invalid_quaternion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Quaternion"):
            signal.rotation_matrix_from_wxyz([0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
