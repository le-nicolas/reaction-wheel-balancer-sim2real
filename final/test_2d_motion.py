#!/usr/bin/env python3
"""Test 2D base motion tracking."""

import numpy as np
import mujoco
import mujoco.viewer
import time

# Load model
import os
os.chdir(os.path.dirname(__file__))  # Change to script directory
xml_path = 'final.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Get joint indices
jid_base_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x_slide")
jid_base_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y_slide")

# Get velocity indices (for base motion)
v_base_x = model.jnt_dofadr[jid_base_x]  # Velocity index for X slider
v_base_y = model.jnt_dofadr[jid_base_y]  # Velocity index for Y slider

print(f"Base X position index: {jid_base_x}")
print(f"Base Y position index: {jid_base_y}")
print(f"Base X velocity index: {v_base_x}")
print(f"Base Y velocity index: {v_base_y}")
print("\nRunning 10-second motion test with random disturbances...\n")

# Track motion
pos_x_values = []
pos_y_values = []
time_values = []

max_x = 0
min_x = 0
max_y = 0
min_y = 0

# Load LQR controller
import pickle
try:
    with open('lqr_K.pkl', 'rb') as f:
        K = pickle.load(f)
    print(f"Loaded LQR gain K with shape {K.shape}")
except:
    print("Warning: Could not load LQR gain, using zero control")
    K = np.zeros((3, 5))

for step in range(10000):  # 10 seconds
    # Apply disturbance every 100 steps
    if step % 100 == 0 and step > 0:
        # Random 20N push
        magnitude = 20
        angles = np.random.uniform(0, 2*np.pi, 2)  # XY plane disturbances
        data.xfrc_applied[0, 0] = magnitude * np.cos(angles[0])
        data.xfrc_applied[0, 1] = magnitude * np.sin(angles[1])
        data.xfrc_applied[0, 2] = 0  # Z force (minor perturbations)
    else:
        data.xfrc_applied[0, :] = 0
    
    # Simple control input (arbitrary non-zero)
    if step % 10 == 0:
        data.ctrl[0] = 1.0  # wheel spin
        data.ctrl[1] = 5.0  # X force
        data.ctrl[2] = 5.0  # Y force
    else:
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = 0
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Record positions
    if step % 10 == 0:  # Sample every 10ms
        pos_x = data.qpos[jid_base_x]
        pos_y = data.qpos[jid_base_y]
        
        pos_x_values.append(pos_x)
        pos_y_values.append(pos_y)
        time_values.append(step * 0.001)
        
        max_x = max(max_x, pos_x)
        min_x = min(min_x, pos_x)
        max_y = max(max_y, pos_y)
        min_y = min(min_y, pos_y)
        
        if step % 1000 == 0:
            print(f"Step {step:5d}: x_pos={pos_x:8.4f}m  y_pos={pos_y:8.4f}m  "
                  f"v_x={data.qvel[v_base_x]:7.4f}  v_y={data.qvel[v_base_y]:7.4f}")

print("\n" + "="*60)
print("2D BASE MOTION SUMMARY")
print("="*60)
print(f"X-axis motion:")
print(f"  Range: [{min_x:.6f}, {max_x:.6f}] m")
print(f"  Total displacement: {max_x - min_x:.6f} m = {(max_x - min_x)*1000:.2f} mm")
print(f"\nY-axis motion:")
print(f"  Range: [{min_y:.6f}, {max_y:.6f}] m")
print(f"  Total displacement: {max_y - min_y:.6f} m = {(max_y - min_y)*1000:.2f} mm")
print(f"\nBoth axes moving independently: {abs(max_x - min_x) > 0.001 and abs(max_y - min_y) > 0.001}")
print("="*60)
