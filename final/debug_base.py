import mujoco
import numpy as np

# ============================================================
# Load model and check structure
# ============================================================
model = mujoco.MjModel.from_xml_path("final/final.xml")
data  = mujoco.MjData(model)

print("=== MODEL STRUCTURE ===")
print(f"Total bodies: {model.nbody}")
print(f"Total joints: {model.njnt}")
print(f"Total DOF: {model.nv}")
print(f"Total qpos: {model.nq}")

print("\n=== JOINTS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = model.jnt_type[i]
    qposadr = model.jnt_qposadr[i]
    dofadr = model.jnt_dofadr[i]
    print(f"Joint {i}: '{name}' type={jtype} qposadr={qposadr} dofadr={dofadr}")

print("\n=== BODIES ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: '{name}'")

# Get joint IDs
def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

jid_base = jid("base_wheel")
print(f"\nBase 'base_wheel' joint ID: {jid_base}")
print(f"Base joint qposadr: {model.jnt_qposadr[jid_base]}")

# Initialize and test
data.qpos[:] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"\n=== INITIAL STATE ===")
print(f"data.qpos: {data.qpos}")
print(f"data.qvel: {data.qvel}")

# Manually set base position and step
q_base = model.jnt_qposadr[jid_base]
print(f"\n=== MANUAL TEST ===")
print(f"Setting qpos[{q_base}] = 1.0 manually...")
data.qpos[q_base] = 1.0
mujoco.mj_forward(model, data)
print(f"data.qpos[{q_base}] = {data.qpos[q_base]}")
print(f"data.xpos[1] = {data.xpos[1]}  (base body position)")

print(f"\nSetting vel to 0.5 m/s...")
data.qvel[model.jnt_dofadr[jid_base]] = 0.5
for _ in range(10):
    mujoco.mj_step(model, data)
    print(f"  qpos[{q_base}] = {data.qpos[q_base]:.6f}")
