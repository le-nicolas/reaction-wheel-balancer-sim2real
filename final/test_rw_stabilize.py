import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("final/final.xml")
data  = mujoco.MjData(model)

def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

jid_stick = 1  # stick_hinge is joint 1
aid_rw = aid("wheel_spin")  # Reaction wheel

# Start with stick upright
data.qpos[jid_stick] = np.pi
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(5000):
        # Apply constant reaction wheel torque
        data.ctrl[aid_rw] = -50.0
        
        if step % 500 == 0:
            print(f"Step {step}: stick_pos={data.qpos[jid_stick]:.4f}, stick_vel={data.qvel[1]:.4f}")
        
        mujoco.mj_step(model, data)
        viewer.sync()
