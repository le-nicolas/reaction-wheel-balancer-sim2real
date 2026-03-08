# Reaction Wheel Balancer Sim-to-Real

This repository is the sim-to-real bring-up package for a reaction-wheel balancer with a driven base wheel.

The intended real robot architecture is:

- base wheel corrects pitch
- reaction wheel corrects roll
- ESP32 handles sensors, motor output, telemetry, and safety
- the PC-side stack stays available for live HIL tuning and dyno-style mapping

The point of this repo is not just to "flash firmware." It is to make first power-on practical:

- verify wiring and signs
- verify which control channel drives which motor
- verify ESTOP and timeout behavior
- save a mapping profile
- then tune the real robot toward the MuJoCo behavior

## Hardware Target

This package was prepared around the following build direction:

- `ESP32-WROOM` dev board
- `BMI088` IMU
- `AS5600` magnetic encoder on the reaction wheel BLDC
- `2804` hollow-shaft BLDC with `DRV8313`-class 3PWM driver board
- `110 RPM` geared base motor with `BTS7960` H-bridge
- `3S Li-Po 11.1 V 1500 mAh 25C`
- `12 V -> 5 V` buck converter for logic power

## Repository Layout

- `final/`
  - MuJoCo runtime/controller/export stack
  - HIL bridge
  - synthetic plug-and-play smoke tester
  - mapping profile and workflow docs
- `esp32_rw_base_platformio/`
  - PlatformIO firmware scaffold for the ESP32 rig
  - sensor readout, WiFi telemetry, motor output, and onboard conservative mode

## Current Status

What is already in place:

- real-time synthetic HIL smoke testing for plug-and-play bring-up
- live mapping profile support in the bridge
- timeout and tilt ESTOP behavior
- PlatformIO firmware scaffold for the ESP32 rig
- telemetry comparison tooling for sim vs HIL logs

What the current stack proves:

- the transport layer works
- the dyno-style mapping workflow exists
- the safety layer is usable for restrained bring-up

What still needs controller-side work:

- the runtime controller still does not fully match the intended split actuation
- pitch authority is not yet being driven into the base wheel the way the target robot needs
- this means the remaining gap is controller allocation, not the bridge/mapping infrastructure

## Install

Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Recommended extras:

- PlatformIO for ESP32 builds
- VS Code with PlatformIO extension

## Quick Start

### 1. PC-side synthetic bring-up

Run the real-time synthetic smoke test:

```powershell
python final/hil_plug_play_smoke.py --bridge-backend stub
```

This checks:

- idle
- pitch forward
- pitch backward
- roll right
- roll left
- tilt ESTOP
- packet timeout ESTOP

### 2. Live bridge for hardware

Start the bridge with a mapping profile:

```powershell
python final/hil_bridge.py --esp32-ip <ESP32_IP> --mapping-profile final/hardware_mapping_template.json --save-mapping-profile final/results/live_map_session.json --plot
```

This is the main dyno-style workflow. You bring the robot up physically restrained, observe telemetry, and adjust mapping values in real time.

### 3. ESP32 firmware

The ESP32 project lives in:

```text
esp32_rw_base_platformio/
```

Read:

- `esp32_rw_base_platformio/README.md`

Typical flow:

```powershell
cd esp32_rw_base_platformio
pio run -t upload
```

## Mapping Workflow

The mapping layer is the set of values you adjust during restrained bench bring-up so the hardware behaves like the simulation.

Main mapping knobs:

- `reaction_sign`
- `drive_sign`
- `pitch_rate_sign`
- `roll_rate_sign`
- `reaction_speed_sign`
- `accel_x_sign`
- `accel_y_sign`
- `accel_z_sign`
- `rw_cmd_scale`
- `drive_cmd_scale`
- `pitch_estop_deg`
- `roll_estop_deg`
- `comm_estop_s`

Supporting docs:

- `final/DYNO_MAPPING_WORKFLOW.md`
- `final/HIL_PLUG_AND_PLAY_MATRIX.md`

## Practical Bring-Up Order

1. Power the robot with the motors restrained or wheels lifted.
2. Start the PC HIL bridge first.
3. Verify the IMU signs at idle.
4. Verify roll commands go to the reaction wheel.
5. Verify pitch commands go to the base wheel.
6. Confirm large tilt forces ESTOP and zero outputs.
7. Confirm packet loss forces timeout ESTOP.
8. Save the mapping profile.
9. Only then start matching the real response to MuJoCo.

## Important Engineering Note

The HIL/mapping infrastructure is farther along than the runtime control allocation.

That means:

- the bridge and live-adjustment workflow are ready
- the safety path is ready for restrained testing
- the remaining work is to make the runtime/exported controller give the base wheel the right authority for pitch stabilization

## Files To Read First

- `README.md`
- `final/hil_bridge.py`
- `final/hil_plug_play_smoke.py`
- `final/hardware_mapping_template.json`
- `final/DYNO_MAPPING_WORKFLOW.md`
- `final/HIL_PLUG_AND_PLAY_MATRIX.md`
- `esp32_rw_base_platformio/README.md`

