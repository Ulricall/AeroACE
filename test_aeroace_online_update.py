import tempfile
import unittest

import numpy as np
import torch

import controller


class AeroACEOnlineUpdateTest(unittest.TestCase):
    def make_controller(self, **overrides):
        kwargs = {
            "input_dim": 2,
            "hidden_dim": 2,
            "expert_dim": 6,
            "given_pid": True,
            "p": 1.0,
            "i": 0.0,
            "d": 1.0,
            "dict_max_entries": 8,
            "dict_min_cosine_distance": 0.01,
            "dict_max_entries_per_bucket": 8,
            "online_update": True,
            "online_anomaly_similarity_threshold": 0.5,
            "online_min_anomaly_steps": 1,
            "online_warmup_steps": 0,
            "online_update_interval_steps": 1,
            "online_residual_window": 3,
            "online_residual_consistency_threshold": 0.0,
            "online_force_clip_norm": 5.0,
            "online_force_reject_norm": 20.0,
        }
        kwargs.update(overrides)
        ctrl = controller.AeroACE(**kwargs)
        ctrl.reset_controller()
        ctrl.expert_dict.store(
            np.array([1.0, 0.0]),
            np.zeros(6),
            metadata={"bucket": "offline"},
        )
        return ctrl

    def test_static_mode_never_writes(self):
        ctrl = self.make_controller(online_update=False)
        size_before = ctrl.expert_dict.current_size

        updated = ctrl.maybe_online_update(
            np.array([0.0, 1.0]),
            np.zeros(6),
            np.array([1.0, 2.0, 3.0]),
        )

        self.assertFalse(updated)
        self.assertEqual(ctrl.expert_dict.current_size, size_before)
        self.assertEqual(ctrl.get_online_update_report()["last_status"], "disabled")

    def test_force_is_clipped_before_projection_and_insert(self):
        ctrl = self.make_controller()
        h_t = np.array([0.0, 1.0])

        updated = ctrl.maybe_online_update(
            h_t,
            np.zeros(6),
            np.array([6.0, 8.0, 0.0]),
        )

        self.assertTrue(updated)
        self.assertEqual(ctrl.expert_dict.last_store_status, "inserted")
        self.assertAlmostEqual(np.linalg.norm(ctrl.online_last_force_used), 5.0)
        c_new = ctrl.expert_dict.values[ctrl.expert_dict.current_size - 1].numpy()
        predicted_force = np.kron(np.eye(3), h_t) @ c_new
        np.testing.assert_allclose(predicted_force, np.array([3.0, 4.0, 0.0]), atol=1e-10)

    def test_hard_force_outlier_is_rejected(self):
        ctrl = self.make_controller()
        size_before = ctrl.expert_dict.current_size

        updated = ctrl.maybe_online_update(
            np.array([0.0, 1.0]),
            np.zeros(6),
            np.array([100.0, 0.0, 0.0]),
        )

        self.assertFalse(updated)
        self.assertEqual(ctrl.expert_dict.current_size, size_before)
        report = ctrl.get_online_update_report()
        self.assertEqual(report["last_status"], "rejected_force_norm")
        self.assertEqual(report["rejected_force_norm"], 1)

    def test_temporally_inconsistent_outlier_is_rejected(self):
        ctrl = self.make_controller(
            online_require_anomaly=False,
            online_force_clip_norm=0.0,
            online_force_reject_norm=100.0,
            online_residual_consistency_threshold=2.0,
            dict_min_cosine_distance=0.2,
        )
        h_t = np.array([0.0, 1.0])
        for _ in range(3):
            ctrl.maybe_online_update(h_t, np.zeros(6), np.array([1.0, 0.0, 0.0]))
        size_before = ctrl.expert_dict.current_size

        updated = ctrl.maybe_online_update(
            h_t,
            np.zeros(6),
            np.array([10.0, 0.0, 0.0]),
        )

        self.assertFalse(updated)
        self.assertEqual(ctrl.expert_dict.current_size, size_before)
        report = ctrl.get_online_update_report()
        self.assertEqual(report["last_status"], "rejected_inconsistent")
        self.assertEqual(report["rejected_inconsistent"], 1)

    def test_full_dictionary_rejects_new_entry_without_pruning(self):
        ctrl = self.make_controller(
            dict_max_entries=1,
            dict_max_entries_per_bucket=0,
            online_require_anomaly=False,
        )
        key_before = ctrl.expert_dict.keys[0].clone()
        value_before = ctrl.expert_dict.values[0].clone()

        updated = ctrl.maybe_online_update(
            np.array([0.0, 1.0]),
            np.zeros(6),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertFalse(updated)
        self.assertEqual(ctrl.expert_dict.current_size, 1)
        self.assertTrue(torch.equal(ctrl.expert_dict.keys[0], key_before))
        self.assertTrue(torch.equal(ctrl.expert_dict.values[0], value_before))
        report = ctrl.get_online_update_report()
        self.assertEqual(report["last_status"], "skipped_full")
        self.assertEqual(report["rejected_capacity"], 1)
        self.assertFalse(report["pruning"])
        self.assertFalse(report["forgetting"])
        self.assertFalse(report["confidence_weighting"])

    def test_online_merge_cannot_modify_predeployment_entry(self):
        ctrl = self.make_controller(
            online_require_anomaly=False,
            dict_min_cosine_distance=0.2,
        )
        key_before = ctrl.expert_dict.keys[0].clone()
        value_before = ctrl.expert_dict.values[0].clone()

        updated = ctrl.maybe_online_update(
            np.array([1.0, 0.0]),
            np.zeros(6),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(updated)
        self.assertEqual(ctrl.expert_dict.current_size, 2)
        self.assertEqual(ctrl.expert_dict.last_store_status, "inserted")
        self.assertTrue(torch.equal(ctrl.expert_dict.keys[0], key_before))
        self.assertTrue(torch.equal(ctrl.expert_dict.values[0], value_before))
        self.assertEqual(ctrl.get_online_update_report()["protected_dictionary_size"], 1)

        merged = ctrl.maybe_online_update(
            np.array([1.0, 0.0]),
            np.zeros(6),
            np.array([2.0, 0.0, 0.0]),
        )

        self.assertTrue(merged)
        self.assertEqual(ctrl.expert_dict.current_size, 2)
        self.assertEqual(ctrl.expert_dict.last_store_status, "merged")
        self.assertTrue(torch.equal(ctrl.expert_dict.keys[0], key_before))
        self.assertTrue(torch.equal(ctrl.expert_dict.values[0], value_before))

    def test_residual_force_uses_acceleration_attitude_and_commanded_thrust(self):
        ctrl = self.make_controller()
        state = np.zeros(13)
        state[3] = 1.0
        hover_speed = np.sqrt(
            ctrl.params["m"] * ctrl.params["g"] / (4.0 * ctrl.params["C_T"])
        )
        ctrl.motor_speed = np.full(4, hover_speed)
        acceleration = np.array([1.0, -2.0, 0.5])

        residual = ctrl.get_residual(state, acceleration)

        np.testing.assert_allclose(
            residual,
            ctrl.params["m"] * acceleration,
            atol=1e-10,
        )

    def test_old_checkpoint_keeps_runtime_dictionary_safeguards(self):
        ctrl = self.make_controller(
            dict_min_cosine_distance=0.123,
            dict_max_entries_per_bucket=7,
            dict_ema_alpha=0.8,
        )
        checkpoint = {
            "fgru": ctrl.fgru.state_dict(),
            "hidden_dim": 2,
            "expert_dim": 6,
            "expert_max_entries": 8,
            "expert_size": 1,
            "expert_keys": torch.tensor([[1.0, 0.0]]),
            "expert_values": torch.zeros((1, 6)),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(checkpoint, tmp.name)
            ctrl.load(tmp.name, map_location="cpu")

        self.assertAlmostEqual(ctrl.expert_dict.min_cosine_distance, 0.123)
        self.assertEqual(ctrl.expert_dict.max_entries_per_bucket, 7)
        self.assertAlmostEqual(ctrl.expert_dict.ema_alpha, 0.8)

    def test_dictionary_rejects_nonfinite_entry(self):
        expert_dict = controller.ExpertDictionary(2, 6, max_entries=2)

        stored = expert_dict.store(
            np.array([np.nan, 0.0]),
            np.zeros(6),
        )

        self.assertFalse(stored)
        self.assertEqual(expert_dict.current_size, 0)
        self.assertEqual(expert_dict.last_store_status, "skipped_nonfinite")
        self.assertEqual(expert_dict.describe()["num_skipped_nonfinite"], 1)


if __name__ == "__main__":
    unittest.main()
