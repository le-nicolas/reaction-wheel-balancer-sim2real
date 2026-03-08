import unittest

import mujoco
import numpy as np

import export_firmware_params as export
import final as runtime
from adaptive_id import AdaptiveGainScheduler


def _build_nominal_bundle():
    args = runtime.parse_args(
        [
            "--mode",
            "robust",
            "--enable-online-id",
            "--online-id-recompute-every",
            "8",
            "--online-id-min-updates",
            "20",
        ]
    )
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
    p_aug = export._solve_discrete_are_robust(a_aug, b_aug, q_aug, cfg.r_du, label="Adaptive test")
    k_du = export._solve_linear_robust(
        b_aug.T @ p_aug @ b_aug + cfg.r_du,
        b_aug.T @ p_aug @ a_aug,
        label="Adaptive test gain",
    )
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = float(control_steps * model.opt.timestep)
    return cfg, a, b, k_du, control_dt


class OnlineIdAdaptiveTests(unittest.TestCase):
    def test_adaptive_scheduler_tracks_scales_and_recomputes_gain(self):
        cfg, a_nom, b_nom, k_du, control_dt = _build_nominal_bundle()
        scheduler = AdaptiveGainScheduler(cfg=cfg, a_nominal=a_nom, b_nominal=b_nom, control_dt=control_dt)

        g_true = 1.35
        i_true = 0.72
        a_true = a_nom.copy()
        b_true = b_nom.copy()
        a_true[2, 0] *= g_true
        a_true[3, 1] *= g_true
        b_true[2, :] *= i_true
        b_true[3, :] *= i_true

        rng = np.random.default_rng(1234)
        x_prev = np.zeros(9, dtype=float)
        x_prev[0] = np.radians(2.0)
        x_prev[1] = -np.radians(1.5)
        k_curr = k_du.copy()

        prev_state = None
        for step in range(450):
            u_prev = np.array(
                [
                    0.28 * np.sin(0.03 * step) + 0.05 * rng.standard_normal(),
                    0.06 * np.sin(0.02 * step + 0.4),
                    0.06 * np.cos(0.017 * step - 0.2),
                ],
                dtype=float,
            )
            x_curr = a_true @ x_prev + b_true @ u_prev
            x_curr += 0.0005 * rng.standard_normal(9)
            x_curr = np.clip(
                x_curr,
                np.array([-0.55, -0.55, -10.0, -10.0, -220.0, -0.9, -0.9, -5.0, -5.0], dtype=float),
                np.array([0.55, 0.55, 10.0, 10.0, 220.0, 0.9, 0.9, 5.0, 5.0], dtype=float),
            )

            k_curr, _ = scheduler.maybe_update_gain(
                x_prev=prev_state,
                u_prev=u_prev,
                x_curr=x_curr,
                k_current=k_curr,
            )
            prev_state = x_curr.copy()
            x_prev = x_curr

        stats = scheduler.stats
        self.assertGreater(stats.rls_updates, 100)
        self.assertGreater(stats.gain_recomputes, 0)
        self.assertTrue(np.all(np.isfinite(k_curr)))
        self.assertLess(abs(stats.gravity_scale - g_true), 0.36)
        self.assertLess(abs(stats.inertia_inv_scale - i_true), 0.35)


if __name__ == "__main__":
    unittest.main()
