import mujoco
import mujoco.viewer
import numpy as np

# Load with gravity disabled
model = mujoco.MjModel.from_xml_path("final/final.xml")
model.opt.gravity[:] = 0.0  # Disable gravity
data  = mujoco.MjData(model)

jid_stick = 1

# Start upright
data.qpos[jid_stick] = np.pi
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(2000):
        data.ctrl[:] = 0.0
        
        if step % 500 == 0:
            print(f"Step {step}: stick_pos={data.qpos[jid_stick]:.6f}")
        
        mujoco.mj_step(model, data)
        viewer.sync()
