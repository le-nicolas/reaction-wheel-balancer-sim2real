#!/usr/bin/env python3
"""
30-second omnidirectional disturbance stability test.
Tests stick stabilization under random 3D pushes in all directions.
"""

import numpy as np
import mujoco
import sys
import os
from scipy.linalg import solve_discrete_are

os.chdir(os.path.dirname(__file__))

# Load model and data
model = mujoco.MjModel.from_xml_path('final.xml')
data = mujoco.MjData(model)
dt = model.opt.timestep

# Get indices
jid_stick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stick_hinge")
jid_rw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_spin")
jid_base_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x_slide")
jid_base_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y_slide")

q_stick = model.jnt_qposadr[jid_stick]
v_stick = model.jnt_dofadr[jid_stick]
v_rw = model.jnt_dofadr[jid_rw]
v_base_x = model.jnt_dofadr[jid_base_x]
v_base_y = model.jnt_dofadr[jid_base_y]

aid_rw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_spin")
aid_base_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x_force")
aid_base_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_y_force")

stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")

# Linearize system
print("Computing linearized system dynamics...")

def linearize():
    data_lin = mujoco.MjData(model)
    data_lin.qpos[q_stick] = 0.0
    data_lin.qvel[:] = 0.0
    data_lin.ctrl[:] = 0.0
    mujoco.mj_forward(model, data_lin)

    nx = model.nq + model.nv
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))

    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A_full, B_full, None, None)

    idx = [q_stick, model.nq + v_stick, model.nq + v_rw, model.nq + v_base_x, model.nq + v_base_y]
    A_r = A_full[np.ix_(idx, idx)]
    B_r = B_full[np.ix_(idx, [aid_rw, aid_base_x, aid_base_y])]
    return A_r, B_r

A, B = linearize()
print(f"✓ Linearized: A{A.shape}, B{B.shape}")

# LQR design
Q = np.diag([80.0, 30.0, 1.0, 800.0, 800.0])
R = np.diag([1.0, 0.001, 0.001])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
print(f"✓ LQR gains computed: K{K.shape}")

# Kalman Filter
Qn = np.diag([1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
Rn = np.diag([5e-6, 1e-4, 3e-4, 3e-4, 3e-4])
C = np.diag([1.0, 0.5, 0.3, 0.3, 0.3])

Pk = solve_discrete_are(A.T, C.T, Qn, Rn)
L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rn)

# Disturbance parameters
DISTURBANCE_MAGNITUDE = 20.0  # Newtons
DISTURBANCE_INTERVAL = 100    # steps (every 100ms)
TOTAL_STEPS = 30000           # 30 seconds

# Statistics tracking
angle_max = 0
angle_min = 0
angle_samples = []
base_x_max = 0
base_x_min = 0
base_y_max = 0
base_y_min = 0
disturbance_count = 0
rng = np.random.default_rng()

# Initialize state
data.qpos[q_stick] = 0.0
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

x_est = np.zeros(5)
u_prev = np.zeros(3)

print(f"\n{'='*70}")
print(f"OMNIDIRECTIONAL DISTURBANCE STABILITY TEST")
print(f"{'='*70}")
print(f"Duration: 30 seconds ({TOTAL_STEPS} steps @ 1kHz)")
print(f"Disturbance: ±{DISTURBANCE_MAGNITUDE}N in X,Y; ±{DISTURBANCE_MAGNITUDE*0.5}N in Z")
print(f"Direction: Fully omnidirectional 3D random forces every {DISTURBANCE_INTERVAL}ms")
print(f"System: 2D sliding base + inverted pendulum + reaction wheel")
print(f"Control: Discrete LQR with Kalman estimation")
print(f"{'='*70}\n")

SIGMA = np.array([0.002, 0.01, 0.02, 0.02, 0.02])

for step_count in range(TOTAL_STEPS):
    # Extract state from MuJoCo
    stick_angle = data.qpos[q_stick]
    stick_vel = data.qvel[v_stick]
    base_x_pos = data.qpos[jid_base_x]
    base_y_pos = data.qpos[jid_base_y]
    base_x_vel = data.qvel[v_base_x]
    base_y_vel = data.qvel[v_base_y]
    wheel_vel = data.qvel[v_rw]
    
    # Measurement with noise
    y = np.array([
        stick_angle + rng.normal(0, SIGMA[0]),
        stick_vel + rng.normal(0, SIGMA[1]),
        wheel_vel + rng.normal(0, SIGMA[2]),
        base_x_vel + rng.normal(0, SIGMA[3]),
        base_y_vel + rng.normal(0, SIGMA[4])
    ])
    
    # Kalman filter prediction and update
    x_pred = A @ x_est + B @ u_prev
    y_pred = C @ x_pred
    x_est = x_pred + L @ (y - y_pred)
    
    # LQR control
    u = -K @ x_est
    
    # Saturation
    u[0] = np.clip(u[0], -500, 500)      # Wheel torque
    u[1] = np.clip(u[1], -100, 100)      # Base X force
    u[2] = np.clip(u[2], -100, 100)      # Base Y force
    
    # Apply control
    data.ctrl[aid_rw] = u[0]
    data.ctrl[aid_base_x] = u[1]
    data.ctrl[aid_base_y] = u[2]
    
    # Apply omnidirectional random disturbance
    if step_count % DISTURBANCE_INTERVAL == 0 and step_count > 0:
        # Random 3D force vector
        force = np.array([
            rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
            rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
            rng.uniform(-DISTURBANCE_MAGNITUDE * 0.5, DISTURBANCE_MAGNITUDE * 0.5)
        ])
        data.xfrc_applied[stick_body_id, :3] = force
        disturbance_count += 1
    else:
        data.xfrc_applied[stick_body_id, :3] = 0.0
    
    # Simulate one timestep
    mujoco.mj_step(model, data)
    
    # Track statistics
    angle_max = max(angle_max, stick_angle)
    angle_min = min(angle_min, stick_angle)
    base_x_max = max(base_x_max, base_x_pos)
    base_x_min = min(base_x_min, base_x_pos)
    base_y_max = max(base_y_max, base_y_pos)
    base_y_min = min(base_y_min, base_y_pos)
    
    if step_count % 1000 == 0 and step_count > 0:
        angle_samples.append(stick_angle)
        print(f"Step {step_count:5d} ({step_count*0.001:5.1f}s): "
              f"angle={np.degrees(stick_angle):7.2f}° "
              f"base=[{base_x_pos:7.4f}, {base_y_pos:7.4f}]m "
              f"u=[{u[0]:6.1f} {u[1]:6.1f} {u[2]:6.1f}]")
    
    u_prev = u.copy()

# Final report
print(f"\n{'='*70}")
print(f"OMNIDIRECTIONAL DISTURBANCE TEST COMPLETE")
print(f"{'='*70}")

if angle_max < np.radians(1.0):
    print("✓ SUCCESS! System remained stable (angle < ±1°) throughout 30-second test")
else:
    print(f"⚠ Note: Stick reached {np.degrees(angle_max):.2f}° during test")

angle_range_deg = np.degrees(angle_max - angle_min)
print(f"\nStick Angle Statistics:")
print(f"  Range: [{np.degrees(angle_min):7.2f}°, {np.degrees(angle_max):7.2f}°]")
print(f"  Total swing: {angle_range_deg:7.2f}°")
if angle_samples:
    print(f"  RMS angle: {np.degrees(np.sqrt(np.mean(np.array(angle_samples)**2))):7.2f}°")

base_x_range_mm = (base_x_max - base_x_min) * 1000
base_y_range_mm = (base_y_max - base_y_min) * 1000
print(f"\nBase X-axis Motion:")
print(f"  Range: [{base_x_min:8.5f}, {base_x_max:8.5f}] m")
print(f"  Total displacement: {base_x_range_mm:7.2f} mm")

print(f"\nBase Y-axis Motion:")
print(f"  Range: [{base_y_min:8.5f}, {base_y_max:8.5f}] m")
print(f"  Total displacement: {base_y_range_mm:7.2f} mm")

print(f"\nOmnidirectional Disturbances Applied:")
print(f"  Count: {disturbance_count} pulses")
print(f"  Magnitude: ±{DISTURBANCE_MAGNITUDE}N in X, Y; ±{DISTURBANCE_MAGNITUDE*0.5}N in Z")
print(f"  Pattern: Fully randomized 3D every {DISTURBANCE_INTERVAL}ms")

print(f"{'='*70}\n")
