import mujoco

model = mujoco.MjModel.from_xml_path("final/test_simple.xml")
print(f"Joints: {model.nq}")
print(f"Actuators: {model.nu}")

for i in range(model.nq):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  Joint {i}: {name}")

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  Actuator {i}: {name}")

# Try to find the actuator
aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor1")
print(f"\nSearching for 'motor1': {aid}")
