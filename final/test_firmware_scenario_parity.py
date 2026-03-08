import argparse
import unittest
from dataclasses import dataclass

import mujoco
import numpy as np

import export_firmware_params as export
import final as runtime
import runtime_config as runtime_cfg


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

    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep
    return {
        "cfg": cfg,
        "a": a,
        "b": b,
        "k_du": k_du,
        "control_dt": float(control_dt),
    }


@dataclass
class FirmwareControllerState:
    u_prev: np.ndarray
    base_int: np.ndarray


def emulated_firmware_controller_step(
    *,
    state: FirmwareControllerState,
    x_est: np.ndarray,
    cfg: runtime_cfg.RuntimeConfig,
    k_du: np.ndarray,
    control_dt: float,
) -> np.ndarray:
    x_ctrl = x_est.astype(float, copy=True)
    x_ctrl[5] -= cfg.x_ref
    x_ctrl[6] -= cfg.y_ref

    state.base_int[0] = float(np.clip(state.base_int[0] + x_ctrl[5] * control_dt, -cfg.int_clamp, cfg.int_clamp))
    state.base_int[1] = float(np.clip(state.base_int[1] + x_ctrl[6] * control_dt, -cfg.int_clamp, cfg.int_clamp))

    z = np.concatenate([x_ctrl, state.u_prev])
    du_cmd = -(k_du @ z)
    du_cmd[1] += -cfg.ki_base * state.base_int[0]
    du_cmd[2] += -cfg.ki_base * state.base_int[1]

    du = np.clip(du_cmd, -cfg.max_du, cfg.max_du)
    u_unc = state.u_prev + du
    u_cmd = np.clip(u_unc, -cfg.max_u, cfg.max_u)

    near_upright = (
        abs(float(x_ctrl[0])) < cfg.upright_angle_thresh
        and abs(float(x_ctrl[1])) < cfg.upright_angle_thresh
        and abs(float(x_ctrl[2])) < cfg.upright_vel_thresh
        and abs(float(x_ctrl[3])) < cfg.upright_vel_thresh
        and abs(float(x_ctrl[5])) < cfg.upright_pos_thresh
        and abs(float(x_ctrl[6])) < cfg.upright_pos_thresh
    )
    if near_upright:
        u_cmd = u_cmd * cfg.u_bleed
        u_cmd[np.abs(u_cmd) < 1e-3] = 0.0

    state.u_prev = u_cmd.astype(float, copy=True)
    return state.u_prev


class FirmwareScenarioParityTests(unittest.TestCase):
    def _disturbance_profile(self, step: int) -> np.ndarray:
        pulse = 0.0
        if 80 <= step < 140:
            pulse = 0.55
        elif 260 <= step < 320:
            pulse = -0.65
        elif 480 <= step < 540:
            pulse = 0.40
        return np.array(
            [
                pulse + 0.14 * np.sin(0.031 * step),
                0.09 * np.sin(0.019 * step + 0.2),
                -0.08 * np.cos(0.023 * step - 0.3),
            ],
            dtype=float,
        )

    def _run_scenario(self, mode: str) -> None:
        parser = export.build_arg_parser()
        args = parser.parse_args(["--mode", mode])
        got = export.compute_export_bundle(args)
        ref = reference_bundle(args)

        state_ref = FirmwareControllerState(u_prev=np.zeros(3, dtype=float), base_int=np.zeros(2, dtype=float))
        state_got = FirmwareControllerState(u_prev=np.zeros(3, dtype=float), base_int=np.zeros(2, dtype=float))
        x = np.zeros(9, dtype=float)
        x[0] = np.radians(2.5)
        x[1] = -np.radians(1.8)

        for step in range(700):
            u_ref = emulated_firmware_controller_step(
                state=state_ref,
                x_est=x,
                cfg=ref["cfg"],
                k_du=ref["k_du"],
                control_dt=ref["control_dt"],
            )
            u_got = emulated_firmware_controller_step(
                state=state_got,
                x_est=x,
                cfg=got["cfg"],
                k_du=got["k_du"],
                control_dt=got["control_dt"],
            )

            np.testing.assert_allclose(u_got, u_ref, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(state_got.base_int, state_ref.base_int, atol=1e-10, rtol=1e-10)

            disturbance_u = self._disturbance_profile(step)
            x = ref["a"] @ x + ref["b"] @ (u_ref + disturbance_u)
            x[0] += 0.0008 * np.sin(0.01 * step)
            x[1] += 0.0006 * np.cos(0.012 * step)
            x = np.clip(
                x,
                np.array(
                    [
                        -0.70,
                        -0.70,
                        -18.0,
                        -18.0,
                        -320.0,
                        -1.20,
                        -1.20,
                        -8.0,
                        -8.0,
                    ],
                    dtype=float,
                ),
                np.array(
                    [
                        0.70,
                        0.70,
                        18.0,
                        18.0,
                        320.0,
                        1.20,
                        1.20,
                        8.0,
                        8.0,
                    ],
                    dtype=float,
                ),
            )

        np.testing.assert_allclose(state_got.u_prev, state_ref.u_prev, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.all(np.isfinite(state_got.u_prev)))
        self.assertTrue(np.all(np.isfinite(state_got.base_int)))

    def test_smooth_firmware_scenario_parity(self):
        self._run_scenario("smooth")

    def test_robust_firmware_scenario_parity(self):
        self._run_scenario("robust")


if __name__ == "__main__":
    unittest.main()
