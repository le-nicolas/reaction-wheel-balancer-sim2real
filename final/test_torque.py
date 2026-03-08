import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("final/final.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

# Find joint IDs
def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

aid_stick = aid("stick_hinge")

# Initial condition
data.qpos[1] = 0.0  # Stick at downright position
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        step += 1
        
        # Apply large constant torque to stick
        data.ctrl[:] = 0.0
        data.ctrl[aid_stick] = 100.0  # Large torque on stick hinge
        
        if step % 100 == 0:
            print(f"Step {step}: stick_pos={data.qpos[1]:.4f} stick_vel={data.qvel[1]:.4f}")
        
        mujoco.mj_step(model, data)
        viewer.sync()
