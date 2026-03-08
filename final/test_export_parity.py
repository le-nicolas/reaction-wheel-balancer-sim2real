import argparse
import unittest

import mujoco
import numpy as np

import export_firmware_params as export
import final as runtime


def reference_bundle(args: argparse.Namespace):
    cfg = runtime.build_config(args)
    xml_path = __import__("pathlib").Path(__file__).with_name("final.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    a, b = export.linearize_model(model, data)

    nx = a.shape[0]
    nu = b.shape[1]
    a_aug = np.block([[a, b], [np.zeros((nu, nx)), np.eye(nu)]])
    b_aug = np.vstack([b, np.eye(nu)])
    q_aug = np.block([[cfg.qx, np.zeros((nx, nu))], [np.zeros((nu, nx)), cfg.qu]])
    p_aug = export._solve_discrete_are_robust(a_aug, b_aug, q_aug, cfg.r_du, label="Reference controller")
    k_du = export._solve_linear_robust(
        b_aug.T @ p_aug @ b_aug + cfg.r_du,
        b_aug.T @ p_aug @ a_aug,
        label="Reference controller gain",
    )

    c = runtime.build_partial_measurement_matrix(cfg)
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep
    wheel_lsb = (2.0 * np.pi) / (cfg.wheel_encoder_ticks_per_rev * control_dt)
    qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    rn = runtime.build_measurement_noise_cov(cfg, wheel_lsb)
    l = runtime.build_kalman_gain(a, qn, c, rn)

    return {
        "cfg": cfg,
        "a": a,
        "b": b,
        "c": c,
        "k_du": k_du,
        "l": l,
        "control_steps": control_steps,
        "control_dt": control_dt,
        "wheel_lsb": wheel_lsb,
    }


class ExportParityTests(unittest.TestCase):
    def _check_mode(self, mode: str):
        parser = export.build_arg_parser()
        args = parser.parse_args(["--mode", mode])

        got = export.compute_export_bundle(args)
        ref = reference_bundle(args)

        np.testing.assert_allclose(got["a"], ref["a"], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(got["b"], ref["b"], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(got["c"], ref["c"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(got["k_du"], ref["k_du"], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(got["l"], ref["l"], atol=1e-8, rtol=1e-8)

        self.assertTrue(np.all(np.isfinite(got["a"])))
        self.assertTrue(np.all(np.isfinite(got["b"])))
        self.assertTrue(np.all(np.isfinite(got["k_du"])))
        self.assertTrue(np.all(np.isfinite(got["l"])))

        np.testing.assert_allclose(got["cfg"].max_u, ref["cfg"].max_u, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(got["cfg"].max_du, ref["cfg"].max_du, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(got["cfg"].ki_base, ref["cfg"].ki_base, places=12)
        self.assertAlmostEqual(got["cfg"].u_bleed, ref["cfg"].u_bleed, places=12)
        self.assertEqual(got["control_steps"], ref["control_steps"])
        self.assertAlmostEqual(got["control_dt"], ref["control_dt"], places=12)
        self.assertAlmostEqual(got["wheel_lsb"], ref["wheel_lsb"], places=12)

        header = export.render_header(got, args)
        self.assertIn("CTRL_XML_SHA256", header)
        self.assertIn("CTRL_GENERATION_SIGNATURE_SHA256", header)

    def test_smooth_parity(self):
        self._check_mode("smooth")

    def test_robust_parity(self):
        self._check_mode("robust")

    def test_hardware_safe_limits(self):
        parser = export.build_arg_parser()
        args = parser.parse_args(["--mode", "smooth", "--hardware-safe"])
        got = export.compute_export_bundle(args)
        cfg = got["cfg"]

        self.assertTrue(cfg.hardware_safe)
        self.assertLessEqual(cfg.max_u[0], 0.05 + 1e-12)
        self.assertLessEqual(cfg.max_u[1], 4.0 + 1e-12)
        self.assertLessEqual(cfg.max_u[2], 4.0 + 1e-12)
        self.assertLessEqual(cfg.max_du[0], 0.6 + 1e-12)
        self.assertLessEqual(cfg.max_du[1], 0.18 + 1e-12)
        self.assertLessEqual(cfg.max_du[2], 0.18 + 1e-12)
        self.assertLessEqual(cfg.max_base_speed_m_s, 0.12 + 1e-12)
        self.assertLessEqual(cfg.max_pitch_roll_rate_rad_s, 4.0 + 1e-12)
        self.assertLessEqual(np.degrees(cfg.crash_angle_rad), 10.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
