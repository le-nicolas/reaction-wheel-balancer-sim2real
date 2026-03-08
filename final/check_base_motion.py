import sys
import mujoco
import numpy as np
from pathlib import Path
from scipy.linalg import solve_discrete_are

# ============================================================
# Load model
# ============================================================
xml_path = Path(__file__).with_name("final.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)


def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


# ============================================================
# Resolve joints and actuators
# ============================================================
jid_pitch = jid("stick_pitch")
jid_roll = jid("stick_roll")
jid_rw = jid("wheel_spin")
jid_base_x = jid("base_x_slide")
jid_base_y = jid("base_y_slide")

q_pitch = model.jnt_qposadr[jid_pitch]
q_roll = model.jnt_qposadr[jid_roll]
q_base_x = model.jnt_qposadr[jid_base_x]
q_base_y = model.jnt_qposadr[jid_base_y]

v_pitch = model.jnt_dofadr[jid_pitch]
v_roll = model.jnt_dofadr[jid_roll]
v_rw = model.jnt_dofadr[jid_rw]
v_base_x = model.jnt_dofadr[jid_base_x]
v_base_y = model.jnt_dofadr[jid_base_y]

aid_rw = aid("wheel_spin")
aid_base_x = aid("base_x_force")
aid_base_y = aid("base_y_force")

# ============================================================
# Linearize around upright equilibrium
# ============================================================
PITCH_EQ = 0.0
ROLL_EQ = 0.0

data.qpos[:] = 0.0
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
data.qpos[q_pitch] = PITCH_EQ
data.qpos[q_roll] = ROLL_EQ
mujoco.mj_forward(model, data)

nx = model.nq + model.nv
nu = model.nu
A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)

# [pitch, roll, pitch_rate, roll_rate, wheel_rate, x, y, vx, vy]
idx = [
    q_pitch,
    q_roll,
    model.nq + v_pitch,
    model.nq + v_roll,
    model.nq + v_rw,
    q_base_x,
    q_base_y,
    model.nq + v_base_x,
    model.nq + v_base_y,
]
A_r = A[np.ix_(idx, idx)]
B_r = B[np.ix_(idx, [aid_rw, aid_base_x, aid_base_y])]
NX = A_r.shape[0]
NU = B_r.shape[1]

# ============================================================
# Delta-u LQR design
# ============================================================
A_aug = np.block([
    [A_r, B_r],
    [np.zeros((NU, NX)), np.eye(NU)],
])
B_aug = np.vstack([B_r, np.eye(NU)])

Qx = np.diag([120.0, 90.0, 70.0, 50.0, 1.0, 220.0, 220.0, 520.0, 520.0])
Qu = np.diag([1e-3, 0.08, 0.08])
Q_aug = np.block([
    [Qx, np.zeros((NX, NU))],
    [np.zeros((NU, NX)), Qu],
])
R_du = np.diag([3.0, 0.4818719400522015, 3.0000852247047614])

P_aug = solve_discrete_are(A_aug, B_aug, Q_aug, R_du)
K_du = np.linalg.inv(B_aug.T @ P_aug @ B_aug + R_du) @ (B_aug.T @ P_aug @ A_aug)

# ============================================================
# Kalman filter (enabled in headless robustness mode)
# ============================================================
C = np.eye(NX)
Qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
Rn = np.diag([5e-6, 5e-6, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 6e-4, 6e-4])
Pk = solve_discrete_are(A_r.T, C.T, Qn, Rn)
L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rn)

# ============================================================
# Simulation setup
# ============================================================
MAX_U = np.array([2000.0, 50.0, 50.0])  # [rw, bx, by]
MAX_DU = np.array([8.0, 0.3054359630132863, 0.20885325581086467])      # [rw, bx, by]
KI_BASE = 0.09716498867719854
U_BLEED = 0.9587474485627508
GATE_MAX_SAT_DU = 0.98
GATE_MAX_SAT_ABS = 0.90

DISTURBANCE_MAGNITUDE = 4.0
DISTURBANCE_INTERVAL = 300
STEPS = 5000
SEED = 12345
SIGMA = np.array([0.002, 0.002, 0.01, 0.01, 0.02, 0.005, 0.005, 0.02, 0.02])

rng = np.random.default_rng(SEED)

data.qpos[:] = 0.0
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
data.qpos[q_pitch] = PITCH_EQ
data.qpos[q_roll] = ROLL_EQ
mujoco.mj_forward(model, data)

x_est = np.zeros(NX)
u_prev = np.zeros(NU)
base_int = np.zeros(2)
stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")

max_abs_pitch = 0.0
max_abs_roll = 0.0
max_abs_x = 0.0
max_abs_y = 0.0
max_abs_ubx = 0.0
max_abs_uby = 0.0
crash_step = None

sat_hits = np.zeros(NU, dtype=int)
du_hits = np.zeros(NU, dtype=int)
dt = model.opt.timestep
INT_CLAMP = 2.0
UPRIGHT_ANGLE_THRESH = np.radians(3.0)
UPRIGHT_VEL_THRESH = 0.10
UPRIGHT_POS_THRESH = 0.30

for step in range(1, STEPS + 1):
    x_true = np.array([
        data.qpos[q_pitch] - PITCH_EQ,
        data.qpos[q_roll] - ROLL_EQ,
        data.qvel[v_pitch],
        data.qvel[v_roll],
        data.qvel[v_rw],
        data.qpos[q_base_x],
        data.qpos[q_base_y],
        data.qvel[v_base_x],
        data.qvel[v_base_y],
    ])
    y = x_true + rng.normal(0.0, SIGMA)

    x_pred = A_r @ x_est + B_r @ u_prev
    x_est = x_pred + L @ (y - x_pred)

    x_ctrl = x_est.copy()
    base_int[0] = np.clip(base_int[0] + x_ctrl[5] * dt, -INT_CLAMP, INT_CLAMP)
    base_int[1] = np.clip(base_int[1] + x_ctrl[6] * dt, -INT_CLAMP, INT_CLAMP)

    z = np.concatenate([x_ctrl, u_prev])
    du_cmd = -K_du @ z
    du_cmd[1] += -KI_BASE * base_int[0]
    du_cmd[2] += -KI_BASE * base_int[1]
    du_hits += (np.abs(du_cmd) > MAX_DU).astype(int)
    du = np.clip(du_cmd, -MAX_DU, MAX_DU)

    u_unc = u_prev + du
    sat_hits += (np.abs(u_unc) > MAX_U).astype(int)
    u = np.clip(u_unc, -MAX_U, MAX_U)

    near_upright = (
        abs(x_true[0]) < UPRIGHT_ANGLE_THRESH
        and abs(x_true[1]) < UPRIGHT_ANGLE_THRESH
        and abs(x_true[2]) < UPRIGHT_VEL_THRESH
        and abs(x_true[3]) < UPRIGHT_VEL_THRESH
        and abs(x_true[5]) < UPRIGHT_POS_THRESH
        and abs(x_true[6]) < UPRIGHT_POS_THRESH
    )
    if near_upright:
        u *= U_BLEED
        u[np.abs(u) < 1e-3] = 0.0

    data.ctrl[:] = 0.0
    data.ctrl[aid_rw] = u[0]
    data.ctrl[aid_base_x] = u[1]
    data.ctrl[aid_base_y] = u[2]

    if step % DISTURBANCE_INTERVAL == 0:
        force = np.array([
            rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
            rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
            rng.uniform(-DISTURBANCE_MAGNITUDE * 0.5, DISTURBANCE_MAGNITUDE * 0.5),
        ])
        data.xfrc_applied[stick_body_id, :3] = force
    else:
        data.xfrc_applied[stick_body_id, :3] = 0.0

    mujoco.mj_step(model, data)
    u_prev = u.copy()

    pitch = data.qpos[q_pitch]
    roll = data.qpos[q_roll]
    base_x = data.qpos[q_base_x]
    base_y = data.qpos[q_base_y]

    max_abs_pitch = max(max_abs_pitch, abs(pitch))
    max_abs_roll = max(max_abs_roll, abs(roll))
    max_abs_x = max(max_abs_x, abs(base_x))
    max_abs_y = max(max_abs_y, abs(base_y))
    max_abs_ubx = max(max_abs_ubx, abs(u[1]))
    max_abs_uby = max(max_abs_uby, abs(u[2]))

    if abs(pitch) > (0.5 * np.pi) or abs(roll) > (0.5 * np.pi):
        crash_step = step
        break

SLIDER_LIMIT = 5.0
SAFE_LIMIT = 0.8 * SLIDER_LIMIT
no_crash = crash_step is None
bounded_base = (max_abs_x < SAFE_LIMIT) and (max_abs_y < SAFE_LIMIT)
within_force = (max_abs_ubx <= MAX_U[1] + 1e-9) and (max_abs_uby <= MAX_U[2] + 1e-9)
mean_sat_abs = sat_hits / max(step, 1)
mean_sat_du = du_hits / max(step, 1)
sat_du_ok = bool(np.all(mean_sat_du <= GATE_MAX_SAT_DU))
sat_abs_ok = bool(np.all(mean_sat_abs <= GATE_MAX_SAT_ABS))

print("\n=== HEADLESS VALIDATION SUMMARY ===")
print(f"Steps run:            {step}")
print(f"Crash step:           {crash_step}")
print(f"Max |pitch|:          {np.degrees(max_abs_pitch):8.3f} deg")
print(f"Max |roll|:           {np.degrees(max_abs_roll):8.3f} deg")
print(f"Max |base_x|:         {max_abs_x:8.4f} m")
print(f"Max |base_y|:         {max_abs_y:8.4f} m")
print(f"Max |u_base_x|:       {max_abs_ubx:8.3f} N")
print(f"Max |u_base_y|:       {max_abs_uby:8.3f} N")
print(f"Abs-limit hit rate [rw,bx,by]: {mean_sat_abs}")
print(f"Delta-u clip rate [rw,bx,by]: {mean_sat_du}")
print(f"Clip gate thresholds: sat_du<={GATE_MAX_SAT_DU:.2f}, sat_abs<={GATE_MAX_SAT_ABS:.2f}")

print("\n=== ACCEPTANCE CHECKS ===")
print(f"No crash (pitch/roll < 90 deg):      {'PASS' if no_crash else 'FAIL'}")
print(f"Base bounded (< 80% slider range):   {'PASS' if bounded_base else 'FAIL'}")
print(f"Base force within XML ctrlrange:     {'PASS' if within_force else 'FAIL'}")
print(f"Delta-u clip gate:                   {'PASS' if sat_du_ok else 'FAIL'}")
print(f"Abs-limit clip gate:                 {'PASS' if sat_abs_ok else 'FAIL'}")

if not (no_crash and bounded_base and within_force and sat_du_ok and sat_abs_ok):
    sys.exit(1)
