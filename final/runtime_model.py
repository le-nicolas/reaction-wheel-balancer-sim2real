from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from runtime_config import RuntimeConfig


@dataclass(frozen=True)
class ModelIds:
    q_pitch: int
    q_roll: int
    q_base_x: int
    q_base_y: int
    q_base_quat_w: int
    q_base_quat_x: int
    q_base_quat_y: int
    q_base_quat_z: int
    v_pitch: int
    v_roll: int
    v_rw: int
    v_base_x: int
    v_base_y: int
    v_base_ang_x: int
    v_base_ang_y: int
    v_base_ang_z: int
    aid_rw: int
    aid_base_x: int
    aid_base_y: int
    base_x_body_id: int
    base_y_body_id: int
    stick_body_id: int
    payload_body_id: int
    payload_geom_id: int


@dataclass(frozen=True)
class SensorIds:
    imu_pitch_angle: int
    imu_roll_angle: int
    imu_pitch_rate: int
    imu_roll_rate: int
    wheel_rate: int
    base_x_pos: int
    base_y_pos: int
    base_x_vel: int
    base_y_vel: int

    def required(self, include_base: bool) -> tuple[int, ...]:
        core = (
            self.imu_pitch_angle,
            self.imu_roll_angle,
            self.imu_pitch_rate,
            self.imu_roll_rate,
            self.wheel_rate,
        )
        if include_base:
            return core + (self.base_x_pos, self.base_y_pos, self.base_x_vel, self.base_y_vel)
        return core


@dataclass
class SensorFrontendState:
    bias: np.ndarray
    y_filtered: np.ndarray
    y_delay_out: np.ndarray
    delay_queue: deque
    next_sample_time_s: float = 0.0
    last_sample_time_s: float = 0.0
    initialized: bool = False


def lookup_model_ids(model: mujoco.MjModel) -> ModelIds:
    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def aid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    jid_pitch = jid("stick_pitch")
    jid_roll = jid("stick_roll")
    jid_rw = jid("wheel_spin")
    jid_base_x = jid("base_x_slide")
    jid_base_y = jid("base_y_slide")
    jid_base_free = jid("base_free")
    qadr_base_free = int(model.jnt_qposadr[jid_base_free])
    dadr_base_free = int(model.jnt_dofadr[jid_base_free])

    return ModelIds(
        q_pitch=model.jnt_qposadr[jid_pitch],
        q_roll=model.jnt_qposadr[jid_roll],
        q_base_x=model.jnt_qposadr[jid_base_x],
        q_base_y=model.jnt_qposadr[jid_base_y],
        q_base_quat_w=qadr_base_free + 3,
        q_base_quat_x=qadr_base_free + 4,
        q_base_quat_y=qadr_base_free + 5,
        q_base_quat_z=qadr_base_free + 6,
        v_pitch=model.jnt_dofadr[jid_pitch],
        v_roll=model.jnt_dofadr[jid_roll],
        v_rw=model.jnt_dofadr[jid_rw],
        v_base_x=model.jnt_dofadr[jid_base_x],
        v_base_y=model.jnt_dofadr[jid_base_y],
        v_base_ang_x=dadr_base_free + 3,
        v_base_ang_y=dadr_base_free + 4,
        v_base_ang_z=dadr_base_free + 5,
        aid_rw=aid("wheel_spin"),
        aid_base_x=aid("base_x_force"),
        aid_base_y=aid("base_y_force"),
        base_x_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_x"),
        base_y_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_y"),
        stick_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick"),
        payload_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload"),
        payload_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom"),
    )


def lookup_sensor_ids(model: mujoco.MjModel) -> SensorIds:
    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    return SensorIds(
        imu_pitch_angle=sid("imu_pitch_angle"),
        imu_roll_angle=sid("imu_roll_angle"),
        imu_pitch_rate=sid("imu_pitch_rate"),
        imu_roll_rate=sid("imu_roll_rate"),
        wheel_rate=sid("wheel_rate_sensor"),
        base_x_pos=sid("base_x_pos_sensor"),
        base_y_pos=sid("base_y_pos_sensor"),
        base_x_vel=sid("base_x_vel_sensor"),
        base_y_vel=sid("base_y_vel_sensor"),
    )


def has_required_mujoco_sensors(cfg: RuntimeConfig, sensor_ids: SensorIds) -> bool:
    return all(idx >= 0 for idx in sensor_ids.required(include_base=cfg.base_state_from_sensors))


def resolve_sensor_source(cfg: RuntimeConfig, sensor_ids: SensorIds) -> str:
    if cfg.sensor_source == "direct":
        return "direct"
    if cfg.sensor_source == "mujoco":
        return "mujoco" if has_required_mujoco_sensors(cfg, sensor_ids) else "direct"
    return "mujoco" if has_required_mujoco_sensors(cfg, sensor_ids) else "direct"


def measurement_dim(cfg: RuntimeConfig) -> int:
    return 9 if cfg.base_state_from_sensors else 5


def create_sensor_frontend_state(cfg: RuntimeConfig) -> SensorFrontendState:
    dim = measurement_dim(cfg)
    delay_len = max(int(cfg.sensor_delay_steps), 0) + 1
    return SensorFrontendState(
        bias=np.zeros(dim, dtype=float),
        y_filtered=np.zeros(dim, dtype=float),
        y_delay_out=np.zeros(dim, dtype=float),
        delay_queue=deque(maxlen=delay_len),
        next_sample_time_s=0.0,
        last_sample_time_s=0.0,
        initialized=False,
    )


def reset_sensor_frontend_state(sensor_state: SensorFrontendState):
    sensor_state.bias[:] = 0.0
    sensor_state.y_filtered[:] = 0.0
    sensor_state.y_delay_out[:] = 0.0
    sensor_state.delay_queue.clear()
    sensor_state.next_sample_time_s = 0.0
    sensor_state.last_sample_time_s = 0.0
    sensor_state.initialized = False


def enforce_planar_root_attitude(model, data, ids: ModelIds, forward: bool = True):
    """
    Clamp free-joint attitude to upright.

    This removes an unobservable/uncontrolled free-body tumble mode that can
    dominate COM drift while stick-joint tilt remains small.
    """
    data.qpos[ids.q_base_quat_w] = 1.0
    data.qpos[ids.q_base_quat_x] = 0.0
    data.qpos[ids.q_base_quat_y] = 0.0
    data.qpos[ids.q_base_quat_z] = 0.0
    data.qvel[ids.v_base_ang_x] = 0.0
    data.qvel[ids.v_base_ang_y] = 0.0
    data.qvel[ids.v_base_ang_z] = 0.0
    if forward:
        mujoco.mj_forward(model, data)


def enforce_wheel_only_constraints(model, data, ids: ModelIds, lock_root_attitude: bool = True):
    """Pin base translation + roll so wheel-only mode stays single-axis."""
    data.qpos[ids.q_base_x] = 0.0
    data.qpos[ids.q_base_y] = 0.0
    data.qpos[ids.q_roll] = 0.0
    data.qvel[ids.v_base_x] = 0.0
    data.qvel[ids.v_base_y] = 0.0
    data.qvel[ids.v_roll] = 0.0
    if lock_root_attitude:
        enforce_planar_root_attitude(model, data, ids, forward=False)
    mujoco.mj_forward(model, data)


def get_true_state(data, ids: ModelIds) -> np.ndarray:
    """State vector used by estimator/controller."""
    return np.array(
        [
            data.qpos[ids.q_pitch],
            data.qpos[ids.q_roll],
            data.qvel[ids.v_pitch],
            data.qvel[ids.v_roll],
            data.qvel[ids.v_rw],
            data.qpos[ids.q_base_x],
            data.qpos[ids.q_base_y],
            data.qvel[ids.v_base_x],
            data.qvel[ids.v_base_y],
        ],
        dtype=float,
    )




def reset_state(model, data, q_pitch, q_roll, pitch_eq=0.0, roll_eq=0.0):
    # Start from XML-defined default pose (preserves freejoint height/orientation).
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[q_pitch] = pitch_eq
    data.qpos[q_roll] = roll_eq
    mujoco.mj_forward(model, data)


def set_payload_mass(model: mujoco.MjModel, data: mujoco.MjData, ids: ModelIds, payload_mass_kg: float) -> float:
    """Set runtime payload mass while keeping a valid positive-definite inertia tensor."""
    if ids.payload_body_id < 0 or ids.payload_geom_id < 0:
        return 0.0
    mass_target = float(max(payload_mass_kg, 0.0))
    mass_runtime = max(mass_target, 1e-6)
    sx, sy, sz = model.geom_size[ids.payload_geom_id, :3]
    ixx = (mass_runtime / 3.0) * (sy * sy + sz * sz)
    iyy = (mass_runtime / 3.0) * (sx * sx + sz * sz)
    izz = (mass_runtime / 3.0) * (sx * sx + sy * sy)
    model.body_mass[ids.payload_body_id] = mass_runtime
    model.body_inertia[ids.payload_body_id, :] = np.array([ixx, iyy, izz], dtype=float)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return mass_target


def compute_robot_com_distance_xy(model: mujoco.MjModel, data: mujoco.MjData, support_body_id: int) -> float:
    """Planar COM distance from support-body origin."""
    masses = model.body_mass[1:]
    total_mass = float(np.sum(masses))
    if total_mass <= 1e-12:
        return 0.0
    com_xy = (masses[:, None] * data.xipos[1:, :2]).sum(axis=0) / total_mass
    support_xy = data.xpos[support_body_id, :2]
    return float(np.linalg.norm(com_xy - support_xy))


def build_partial_measurement_matrix(cfg: RuntimeConfig):
    rows = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    if cfg.base_state_from_sensors:
        rows.extend(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    return np.array(rows, dtype=float)


def _measurement_noise_std_vector(cfg: RuntimeConfig) -> np.ndarray:
    return np.array(
        [
            cfg.imu_angle_noise_std_rad,
            cfg.imu_angle_noise_std_rad,
            cfg.imu_rate_noise_std_rad_s,
            cfg.imu_rate_noise_std_rad_s,
            cfg.wheel_encoder_rate_noise_std_rad_s,
            *(
                [
                    cfg.base_encoder_pos_noise_std_m,
                    cfg.base_encoder_pos_noise_std_m,
                    cfg.base_encoder_vel_noise_std_m_s,
                    cfg.base_encoder_vel_noise_std_m_s,
                ]
                if cfg.base_state_from_sensors
                else []
            ),
        ],
        dtype=float,
    )


def _measurement_bias_rw_std_vector(cfg: RuntimeConfig) -> np.ndarray:
    return np.array(
        [
            cfg.imu_angle_bias_rw_std_rad_sqrt_s,
            cfg.imu_angle_bias_rw_std_rad_sqrt_s,
            cfg.imu_rate_bias_rw_std_rad_s_sqrt_s,
            cfg.imu_rate_bias_rw_std_rad_s_sqrt_s,
            cfg.wheel_encoder_bias_rw_std_rad_s_sqrt_s,
            *(
                [
                    cfg.base_encoder_pos_bias_rw_std_m_sqrt_s,
                    cfg.base_encoder_pos_bias_rw_std_m_sqrt_s,
                    cfg.base_encoder_vel_bias_rw_std_m_s_sqrt_s,
                    cfg.base_encoder_vel_bias_rw_std_m_s_sqrt_s,
                ]
                if cfg.base_state_from_sensors
                else []
            ),
        ],
        dtype=float,
    )


def _measurement_clip_vector(cfg: RuntimeConfig) -> np.ndarray:
    return np.array(
        [
            cfg.imu_angle_clip_rad,
            cfg.imu_angle_clip_rad,
            cfg.imu_rate_clip_rad_s,
            cfg.imu_rate_clip_rad_s,
            cfg.wheel_rate_clip_rad_s,
            *(
                [
                    cfg.base_pos_clip_m,
                    cfg.base_pos_clip_m,
                    cfg.base_vel_clip_m_s,
                    cfg.base_vel_clip_m_s,
                ]
                if cfg.base_state_from_sensors
                else []
            ),
        ],
        dtype=float,
    )


def _measurement_lpf_cutoff_vector(cfg: RuntimeConfig) -> np.ndarray:
    return np.array(
        [
            cfg.imu_angle_lpf_hz,
            cfg.imu_angle_lpf_hz,
            cfg.imu_rate_lpf_hz,
            cfg.imu_rate_lpf_hz,
            cfg.wheel_rate_lpf_hz,
            *(
                [
                    cfg.base_pos_lpf_hz,
                    cfg.base_pos_lpf_hz,
                    cfg.base_vel_lpf_hz,
                    cfg.base_vel_lpf_hz,
                ]
                if cfg.base_state_from_sensors
                else []
            ),
        ],
        dtype=float,
    )


def _lpf_alpha(cutoff_hz: float, dt: float) -> float:
    if cutoff_hz <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(np.clip(dt / (tau + dt), 0.0, 1.0))


def _build_direct_measurement(cfg: RuntimeConfig, x_true: np.ndarray) -> np.ndarray:
    y = np.array([x_true[0], x_true[1], x_true[2], x_true[3], x_true[4]], dtype=float)
    if cfg.base_state_from_sensors:
        y = np.concatenate([y, np.array([x_true[5], x_true[6], x_true[7], x_true[8]], dtype=float)])
    return y


def _sensor_value_or_fallback(data: mujoco.MjData, sensor_idx: int, fallback: float) -> float:
    if sensor_idx < 0:
        return float(fallback)
    return float(data.sensordata[sensor_idx])


def _build_mujoco_measurement(
    cfg: RuntimeConfig,
    data: mujoco.MjData,
    sensor_ids: SensorIds,
    fallback: np.ndarray,
) -> np.ndarray:
    y = np.array(
        [
            _sensor_value_or_fallback(data, sensor_ids.imu_pitch_angle, fallback[0]),
            _sensor_value_or_fallback(data, sensor_ids.imu_roll_angle, fallback[1]),
            _sensor_value_or_fallback(data, sensor_ids.imu_pitch_rate, fallback[2]),
            _sensor_value_or_fallback(data, sensor_ids.imu_roll_rate, fallback[3]),
            _sensor_value_or_fallback(data, sensor_ids.wheel_rate, fallback[4]),
        ],
        dtype=float,
    )
    if cfg.base_state_from_sensors:
        y = np.concatenate(
            [
                y,
                np.array(
                    [
                        _sensor_value_or_fallback(data, sensor_ids.base_x_pos, fallback[5]),
                        _sensor_value_or_fallback(data, sensor_ids.base_y_pos, fallback[6]),
                        _sensor_value_or_fallback(data, sensor_ids.base_x_vel, fallback[7]),
                        _sensor_value_or_fallback(data, sensor_ids.base_y_vel, fallback[8]),
                    ],
                    dtype=float,
                ),
            ]
        )
    return y


def build_measurement_noise_cov(cfg: RuntimeConfig, wheel_lsb: float) -> np.ndarray:
    noise_std = _measurement_noise_std_vector(cfg)
    rw_std = _measurement_bias_rw_std_vector(cfg)
    sample_period = 1.0 / max(cfg.sensor_hz, 1e-6)
    variances = noise_std**2 + (rw_std**2) * sample_period
    variances[4] += (wheel_lsb**2) / 12.0
    return np.diag(variances)


def build_kalman_gain(A: np.ndarray, Qn: np.ndarray, C: np.ndarray, R: np.ndarray):
    Pk = solve_discrete_are(A.T, C.T, Qn, R)
    return Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + R)


def estimator_measurement_update(
    cfg: RuntimeConfig,
    x_true: np.ndarray,
    x_pred: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    rng: np.random.Generator,
    wheel_lsb: float,
    *,
    data: mujoco.MjData | None = None,
    sensor_ids: SensorIds | None = None,
    sensor_source: str = "direct",
    sensor_state: SensorFrontendState | None = None,
    sim_time_s: float = 0.0,
    control_dt: float = 0.0,
) -> np.ndarray:
    """One Kalman correction step from sampled, delayed, noisy, clipped sensors."""
    wheel_lsb = max(float(wheel_lsb), 1e-12)
    y_truth = _build_direct_measurement(cfg, x_true)
    noise_std = _measurement_noise_std_vector(cfg)
    clip_vec = _measurement_clip_vector(cfg)

    def _sample_measurement(bias: np.ndarray | None = None) -> np.ndarray:
        if sensor_source == "mujoco" and data is not None and sensor_ids is not None:
            y_src = _build_mujoco_measurement(cfg, data, sensor_ids, fallback=y_truth)
        else:
            y_src = y_truth.copy()
        y_src[4] = np.round(y_src[4] / wheel_lsb) * wheel_lsb
        if bias is not None:
            y_src = y_src + bias
        y_src = y_src + rng.normal(0.0, noise_std)
        return np.clip(y_src, -clip_vec, clip_vec)

    if sensor_state is None:
        y = _sample_measurement(bias=None)
    else:
        sample_period = 1.0 / max(cfg.sensor_hz, 1e-6)
        if not sensor_state.initialized:
            sensor_state.next_sample_time_s = float(sim_time_s)
            sensor_state.last_sample_time_s = float(sim_time_s - sample_period)

        if sim_time_s + 1e-12 >= sensor_state.next_sample_time_s:
            dt_sample = max(float(sim_time_s - sensor_state.last_sample_time_s), min(sample_period, max(control_dt, 1e-6)))
            sensor_state.last_sample_time_s = float(sim_time_s)
            while sensor_state.next_sample_time_s <= sim_time_s + 1e-12:
                sensor_state.next_sample_time_s += sample_period

            bias_rw_std = _measurement_bias_rw_std_vector(cfg)
            sensor_state.bias += rng.normal(0.0, bias_rw_std * np.sqrt(max(dt_sample, 1e-9)))

            y_noisy = _sample_measurement(bias=sensor_state.bias)
            lpf_cutoffs = _measurement_lpf_cutoff_vector(cfg)
            alphas = np.array([_lpf_alpha(cutoff, dt_sample) for cutoff in lpf_cutoffs], dtype=float)
            if not sensor_state.initialized:
                sensor_state.y_filtered[:] = y_noisy
            else:
                sensor_state.y_filtered += alphas * (y_noisy - sensor_state.y_filtered)
            y_filt = sensor_state.y_filtered.copy()

            if not sensor_state.initialized:
                sensor_state.delay_queue.clear()
                delay_len = sensor_state.delay_queue.maxlen if sensor_state.delay_queue.maxlen is not None else 1
                for _ in range(max(delay_len, 1)):
                    sensor_state.delay_queue.append(y_filt.copy())
            else:
                sensor_state.delay_queue.append(y_filt.copy())
            sensor_state.y_delay_out[:] = sensor_state.delay_queue[0]
            sensor_state.initialized = True

        y = sensor_state.y_delay_out.copy()

    x_est = x_pred + L @ (y - C @ x_pred)

    if not cfg.base_state_from_sensors:
        # Legacy teaching model: keep unobserved base states tied to truth.
        x_est[5] = x_true[5]
        x_est[6] = x_true[6]
        x_est[7] = x_true[7]
        x_est[8] = x_true[8]
    return x_est
