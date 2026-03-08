from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from runtime_config import RuntimeConfig


def _solve_discrete_are_robust(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, label: str) -> np.ndarray:
    reg_steps = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    eye = np.eye(r.shape[0], dtype=float)
    last_exc: Exception | None = None
    for eps in reg_steps:
        try:
            p = solve_discrete_are(a, b, q, r + eps * eye)
            if np.all(np.isfinite(p)):
                return 0.5 * (p + p.T)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"{label} DARE failed after regularization fallback: {last_exc}") from last_exc


def _solve_linear_robust(gram: np.ndarray, rhs: np.ndarray, label: str) -> np.ndarray:
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(f"{label} linear solve returned non-finite result.")
        return sol


@dataclass
class AdaptiveGainStats:
    rls_updates: int = 0
    rls_skipped: int = 0
    gain_recomputes: int = 0
    gain_recompute_failures: int = 0
    innovation_rms: float = 0.0
    innovation_abs_mean: float = 0.0
    gravity_scale: float = 1.0
    inertia_inv_scale: float = 1.0


class AdaptiveGainScheduler:
    """
    Online system-ID + gain-scheduled LQR:
    - RLS estimates pitch dynamics coefficients for [pitch -> pitch_rate_next, u_rw -> pitch_rate_next]
    - Estimated coefficients are mapped to gravity and inverse-inertia scales.
    - A/B are updated with those scales and K_du is recomputed periodically with smoothing.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        a_nominal: np.ndarray,
        b_nominal: np.ndarray,
        control_dt: float,
    ):
        self.cfg = cfg
        self.a_nominal = np.asarray(a_nominal, dtype=float)
        self.b_nominal = np.asarray(b_nominal, dtype=float)
        self.control_dt = float(max(control_dt, 1e-6))
        self.nx = int(self.a_nominal.shape[0])
        self.nu = int(self.b_nominal.shape[1])

        self.a20_nom = float(self.a_nominal[2, 0])
        self.a31_nom = float(self.a_nominal[3, 1])
        self.b2_nom = self.b_nominal[2, :].astype(float, copy=True)
        self.b3_nom = self.b_nominal[3, :].astype(float, copy=True)
        self.nom_coeff = np.array([self.a20_nom, float(self.b2_nom[0])], dtype=float)
        self.theta = self.nom_coeff.copy()
        self.p = np.eye(2, dtype=float) * float(max(self.cfg.online_id_init_cov, 1.0))
        self.gravity_scale = 1.0
        self.inertia_inv_scale = 1.0
        self.update_index = 0
        self._innovation_sq_sum = 0.0
        self._innovation_abs_sum = 0.0
        self.stats = AdaptiveGainStats()

    def _build_adapted_model(self) -> tuple[np.ndarray, np.ndarray]:
        a_adapt = self.a_nominal.copy()
        b_adapt = self.b_nominal.copy()
        a_adapt[2, 0] = self.a20_nom * self.gravity_scale
        a_adapt[3, 1] = self.a31_nom * self.gravity_scale
        b_adapt[2, :] = self.b2_nom * self.inertia_inv_scale
        b_adapt[3, :] = self.b3_nom * self.inertia_inv_scale
        return a_adapt, b_adapt

    def _update_rls(self, x_prev: np.ndarray, u_prev: np.ndarray, x_curr: np.ndarray) -> float | None:
        phi = np.array([float(x_prev[0]), float(u_prev[0])], dtype=float)
        excitation = float(np.linalg.norm(phi))
        if excitation < float(max(self.cfg.online_id_min_excitation, 1e-6)):
            self.stats.rls_skipped += 1
            return None
        if abs(float(x_prev[0])) > 0.90 * float(self.cfg.crash_angle_rad):
            # Skip adaptation near crash where dynamics are strongly nonlinear.
            self.stats.rls_skipped += 1
            return None

        rest_nom = (
            float(self.a_nominal[2, :] @ x_prev)
            + float(self.b_nominal[2, :] @ u_prev)
            - self.a20_nom * float(x_prev[0])
            - float(self.b2_nom[0]) * float(u_prev[0])
        )
        y = float(x_curr[2]) - rest_nom
        y_hat = float(phi @ self.theta)
        innovation = float(y - y_hat)
        clip_lim = float(max(self.cfg.online_id_innovation_clip, 1e-6))
        innovation = float(np.clip(innovation, -clip_lim, clip_lim))

        lam = float(np.clip(self.cfg.online_id_forgetting, 0.90, 0.999999))
        p_phi = self.p @ phi
        denom = float(lam + phi.T @ p_phi)
        if (not np.isfinite(denom)) or denom <= 1e-9:
            self.stats.rls_skipped += 1
            return None
        k = p_phi / denom
        self.theta = self.theta + k * innovation
        self.p = (self.p - np.outer(k, phi) @ self.p) / lam
        self.p = 0.5 * (self.p + self.p.T)
        self.stats.rls_updates += 1
        self._innovation_sq_sum += innovation * innovation
        self._innovation_abs_sum += abs(innovation)

        eps = 1e-9
        g_raw = self.gravity_scale
        i_raw = self.inertia_inv_scale
        if abs(self.a20_nom) > eps:
            g_raw = float(self.theta[0] / self.a20_nom)
        if abs(self.b2_nom[0]) > eps:
            i_raw = float(self.theta[1] / self.b2_nom[0])

        g_target = float(
            np.clip(
                g_raw,
                self.cfg.online_id_gravity_scale_min,
                self.cfg.online_id_gravity_scale_max,
            )
        )
        i_target = float(
            np.clip(
                i_raw,
                self.cfg.online_id_inertia_inv_scale_min,
                self.cfg.online_id_inertia_inv_scale_max,
            )
        )
        max_step = float(max(self.cfg.online_id_scale_rate_per_s, 0.0) * self.control_dt)
        if max_step > 0.0:
            self.gravity_scale = float(self.gravity_scale + np.clip(g_target - self.gravity_scale, -max_step, max_step))
            self.inertia_inv_scale = float(
                self.inertia_inv_scale + np.clip(i_target - self.inertia_inv_scale, -max_step, max_step)
            )
        else:
            self.gravity_scale = g_target
            self.inertia_inv_scale = i_target

        self.gravity_scale = float(
            np.clip(
                self.gravity_scale,
                self.cfg.online_id_gravity_scale_min,
                self.cfg.online_id_gravity_scale_max,
            )
        )
        self.inertia_inv_scale = float(
            np.clip(
                self.inertia_inv_scale,
                self.cfg.online_id_inertia_inv_scale_min,
                self.cfg.online_id_inertia_inv_scale_max,
            )
        )
        return innovation

    def maybe_update_gain(
        self,
        *,
        x_prev: np.ndarray | None,
        u_prev: np.ndarray,
        x_curr: np.ndarray,
        k_current: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        self.update_index += 1
        innovation = None
        if x_prev is not None:
            innovation = self._update_rls(
                x_prev=np.asarray(x_prev, dtype=float),
                u_prev=np.asarray(u_prev, dtype=float),
                x_curr=np.asarray(x_curr, dtype=float),
            )
            if self.stats.rls_updates > 0:
                self.stats.innovation_rms = float(np.sqrt(self._innovation_sq_sum / self.stats.rls_updates))
                self.stats.innovation_abs_mean = float(self._innovation_abs_sum / self.stats.rls_updates)
        recompute = (
            self.stats.rls_updates >= int(max(self.cfg.online_id_min_updates, 1))
            and (self.update_index % int(max(self.cfg.online_id_recompute_every, 1)) == 0)
        )
        if (not recompute) or (innovation is None):
            self.stats.gravity_scale = self.gravity_scale
            self.stats.inertia_inv_scale = self.inertia_inv_scale
            return k_current, False

        a_adapt, b_adapt = self._build_adapted_model()
        a_aug = np.block([[a_adapt, b_adapt], [np.zeros((self.nu, self.nx)), np.eye(self.nu)]])
        b_aug = np.vstack([b_adapt, np.eye(self.nu)])
        q_aug = np.block(
            [
                [self.cfg.qx, np.zeros((self.nx, self.nu))],
                [np.zeros((self.nu, self.nx)), self.cfg.qu],
            ]
        )
        try:
            p_aug = _solve_discrete_are_robust(a_aug, b_aug, q_aug, self.cfg.r_du, label="Adaptive scheduler")
            k_target = _solve_linear_robust(
                b_aug.T @ p_aug @ b_aug + self.cfg.r_du,
                b_aug.T @ p_aug @ a_aug,
                label="Adaptive scheduler gain",
            )
            blend = float(np.clip(self.cfg.online_id_gain_blend_alpha, 0.0, 1.0))
            k_next = (1.0 - blend) * np.asarray(k_current, dtype=float) + blend * k_target
            delta_lim = float(max(self.cfg.online_id_gain_max_delta, 0.0))
            if delta_lim > 0.0:
                delta = np.clip(k_next - k_current, -delta_lim, delta_lim)
                k_next = np.asarray(k_current, dtype=float) + delta
            if not np.all(np.isfinite(k_next)):
                raise RuntimeError("Adaptive scheduler produced non-finite K.")
            self.stats.gain_recomputes += 1
            self.stats.gravity_scale = self.gravity_scale
            self.stats.inertia_inv_scale = self.inertia_inv_scale
            return k_next.astype(float, copy=False), True
        except Exception:  # noqa: BLE001
            self.stats.gain_recompute_failures += 1
            self.stats.gravity_scale = self.gravity_scale
            self.stats.inertia_inv_scale = self.inertia_inv_scale
            return k_current, False
