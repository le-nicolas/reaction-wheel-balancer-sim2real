import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

# ============================================================
# Load model
# ============================================================
model = mujoco.MjModel.from_xml_path("final/final.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

# ============================================================
# Resolve joints
# ============================================================
def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

jid_stick = jid("stick_hinge")
jid_rw    = jid("wheel_spin")
jid_base  = jid("base_wheel")

q_stick = model.jnt_qposadr[jid_stick]
q_base  = model.jnt_qposadr[jid_base]
v_stick = model.jnt_dofadr[jid_stick]
v_rw    = model.jnt_dofadr[jid_rw]
v_base  = model.jnt_dofadr[jid_base]

# ============================================================
# Resolve actuators
# ============================================================
def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

aid_rw    = aid("wheel_spin")
aid_base  = aid("base_wheel")

# ============================================================
# Linearization
# ============================================================
def linearize():
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[q_stick] = 0.0
    mujoco.mj_forward(model, data)

    nx = model.nq + model.nv
    nu = model.nu

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)

    idx = [q_stick, model.nq + v_stick, model.nq + v_rw, model.nq + v_base]
    A_r = A[np.ix_(idx, idx)]
    B_r = B[np.ix_(idx, [aid_rw, aid_base])]

    return A_r, B_r

A, B = linearize()

# LQR
Q = np.diag([80.0, 30.0, 1.0, 800.0])
R = np.diag([1.0, 0.001])
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

# ============================================================
# Headless simulation - LONG TERM STABILITY TEST
# ============================================================
data.qpos[q_stick] = 0.0
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

x_est = np.array([0.0, 0.0, 0.0, 0.0])
u_prev = np.zeros(2)
MAX_TAU_RW = 500.0
MAX_F_BASE = 100.0

rng = np.random.default_rng()
SIGMA = np.array([0.002, 0.01, 0.02, 0.02])

C = np.diag([1.0, 0.5, 0.3, 0.3])
Qn = np.diag([1e-4, 1e-4, 1e-5, 1e-5])
Rn = np.diag([5e-6, 1e-4, 3e-4, 3e-4])
Pk = solve_discrete_are(A.T, C.T, Qn, Rn)
L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rn)

# ============================================================
# Disturbance parameters
# ============================================================
DISTURBANCE_MAGNITUDE = 20.0
DISTURBANCE_INTERVAL  = 100

print("\n=== LONG-TERM STABILITY TEST (30 seconds) ===\n")
angles = []
positions = []
crashed = False

for step in range(30000):  # 30 seconds
    stick_angle = data.qpos[q_stick]
    base_pos = data.qpos[q_base]
    
    angles.append(np.degrees(stick_angle))
    positions.append(base_pos)
    
    x_true = np.array([
        stick_angle,
        data.qvel[v_stick],
        data.qvel[v_rw],
        data.qvel[v_base]
    ])
    
    y = x_true + rng.normal(0.0, SIGMA)
    
    x_pred = A @ x_est + B @ u_prev
    y_pred = C @ x_pred
    x_est = x_pred + L @ (y - y_pred)
    
    u_lqr = -K @ x_est
    u = np.array([u_lqr[0], u_lqr[1]])
    u[0] = np.clip(u[0], -MAX_TAU_RW, MAX_TAU_RW)
    u[1] = np.clip(u[1], -MAX_F_BASE, MAX_F_BASE)
    
    data.ctrl[:] = 0.0
    data.ctrl[aid_rw] = u[0]
    data.ctrl[aid_base] = u[1]
    
    # Apply disturbance
    if step % DISTURBANCE_INTERVAL == 0:
        stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        force = np.array([
            rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
            0.0,
            rng.uniform(-DISTURBANCE_MAGNITUDE * 0.5, DISTURBANCE_MAGNITUDE * 0.5)
        ])
        data.xfrc_applied[stick_body_id, :3] = force
    else:
        stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        data.xfrc_applied[stick_body_id, :3] = 0.0
    
    mujoco.mj_step(model, data)
    u_prev = u.copy()
    
    # Check crash
    if abs(stick_angle) > np.pi * 0.5:
        print(f"CRASH at step {step} ({step*0.001:.1f}s): stick fell (angle={angles[-1]:.1f}deg)")
        crashed = True
        break
    
    if step % 5000 == 0 and step > 0:
        print(f"  Step {step} ({step*0.001:.1f}s): angle_max={max(abs(a) for a in angles[-500:]):.2f}deg  "
              f"pos_range=[{min(positions[-500:]):.4f}, {max(positions[-500:]):.4f}]m")

if not crashed:
    print(f"\nSUCCESS! System ran stable for 30 seconds")
    print(f"Stick angle: {np.degrees(angles[-1]):.2f}deg")
    print(f"Max angle reached: {max(abs(a) for a in angles):.2f}deg")
    print(f"RMS angle: {np.std(angles):.4f}deg")
    print(f"Base position range: [{min(positions):.4f}, {max(positions):.4f}] m")
    print(f"Total base displacement: {positions[-1] - positions[0]:.4f} m")
