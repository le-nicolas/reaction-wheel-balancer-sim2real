import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("final/final.xml")
data  = mujoco.MjData(model)

print(f"Total joints: {model.nq}")
print(f"Total actuators: {model.nu}")

print(f"\nAll joints:")
for i in range(model.nq):
    try:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  Joint {i}: '{name}'")
    except:
        pass

print(f"\nAll actuators:")
for i in range(model.nu):
    try:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            print(f"  Actuator {i}: '{name}'")
    except:
        pass

# Try to find stick_hinge
def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

print(f"\nSearching for specific joints/actuators:")
print(f"  stick_hinge joint ID: {jid('stick_hinge')}")
print(f"  stick_hinge actuator ID: {aid('stick_hinge')}")
print(f"  wheel_spin joint ID: {jid('wheel_spin')}")
print(f"  wheel_spin actuator ID: {aid('wheel_spin')}")
print(f"  base_wheel joint ID: {jid('base_wheel')}")
print(f"  base_wheel actuator ID: {aid('base_wheel')}")
