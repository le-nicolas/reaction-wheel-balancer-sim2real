import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("final/final.xml")
model.opt.gravity[:] = 0.0  # No gravity for clarity
data  = mujoco.MjData(model)

# Test θ=0
print("=== Testing θ=0 ===")
data.qpos[1] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
print(f"At θ=0, stick_qpos[1]={data.qpos[1]:.6f}")
print(f"com position: {mujoco.mj_id2name(model, 2, 1)}")

# Get body positions
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name in ['stick', 'wheel']:
        pos = data.xpos[i]
        print(f"  {name}: pos={pos}")

# Test θ=π
print("\n=== Testing θ=π ===")
data.qpos[1] = np.pi
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
print(f"At θ=π, stick_qpos[1]={data.qpos[1]:.6f}")

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name in ['stick', 'wheel']:
        pos = data.xpos[i]
        print(f"  {name}: pos={pos}")
