# ESP32 Reaction Wheel + Base Wheel PlatformIO Scaffold

This folder is the current firmware scaffold for the real rig that matches the MuJoCo sim-to-real stack.

## Current Status

Authoritative status for this firmware as of `2026-03-08`:

- Default mode is still `HIL bridge`.
- The intended real actuator split is:
  - base wheel handles pitch
  - reaction wheel handles roll
- A base-wheel encoder is not required for the inner balance loop.
- The firmware supports an optional base encoder for later outer-loop improvements.
- Reaction-wheel FOC is implemented with SimpleFOC using:
  - `AS5600`
  - `BLDCDriver3PWM`
  - voltage-mode torque control
- The base wheel is a brushed/DC geared motor on `BTS7960`, so it is PWM-driven, not FOC-driven.

If you find older notes saying the base wheel had to wait for an encoder before doing meaningful pitch correction, treat those notes as historical. The current design supports IMU-only pitch stabilization for restrained bring-up.

## Hardware Direction

This scaffold is built around:

- `ESP32-WROOM` dev board
- `BMI088` IMU
- `AS5600` magnetic encoder on the reaction wheel BLDC
- `2804` hollow-shaft BLDC with `DRV8313`-class 3PWM stage
- `110 RPM` geared base motor with `BTS7960`
- `3S Li-Po 11.1 V 1500 mAh 25C`
- `12 V -> 5 V` buck for logic power

## Control Modes

Defined in `include/robot_config.h`:

1. `kHilBridge`
   - current default
   - ESP32 handles sensors, WiFi telemetry, safety, and motor output
   - the PC-side Python stack remains the controller
2. `kOnboardWheelOnly`
   - conservative local mode
   - useful for isolated reaction-wheel checks
3. `kOnboardExplicitSplit`
   - onboard split controller path
   - intended structure:
     - pitch -> base wheel
     - roll -> reaction wheel

Current checked-in defaults are conservative:

- `kControlMode = kHilBridge`
- `kHasBaseWheelEncoder = false`
- `kBaseWheelEnabledInHil = true`
- `kBaseWheelEnabledInOnboard = false`

That means the firmware is ready for live HIL bring-up immediately, while onboard base-wheel use still requires deliberate enabling and tuning.

## Folder Layout

- `platformio.ini`
  - PlatformIO project config
- `include/board_pins.h`
  - GPIO map for the current rig
- `include/robot_config.h`
  - control mode, WiFi, motor scaling, and safety settings
- `include/controller_params.h`
  - exported controller header from the MuJoCo stack
- `src/main.cpp`
  - firmware entry point
  - BMI088 readout
  - AS5600 readout
  - reaction-wheel FOC
  - BTS7960 base-wheel drive
  - telemetry and safety handling
- `lib/sim_parity/`
  - reused controller/estimator/guard code from the sim export path
- `tools/export_sim_controller.ps1`
  - refreshes `controller_params.h`

## Power Topology

Recommended power arrangement:

1. `3S Li-Po` to `BTS7960` motor power
2. the same battery rail to the `DRV8313` motor stage
3. `12 V -> 5 V buck` to ESP32 `5V/VIN`
4. `BMI088` and `AS5600` on `3.3 V`
5. common ground across battery, ESP32, BTS7960, DRV8313, BMI088, and AS5600

If you use the `MP1584`, assign it one clean job and verify the output before connecting sensors.

## Pin Map

Current default pins in `include/board_pins.h`:

- I2C SDA: `GPIO21`
- I2C SCL: `GPIO22`
- DRV8313 enable: `GPIO25`
- DRV8313 phase U/V/W PWM: `GPIO26`, `GPIO27`, `GPIO14`
- BTS7960 RPWM/LPWM: `GPIO32`, `GPIO33`
- BTS7960 REN/LEN: `GPIO18`, `GPIO19`
- base encoder A/B: `GPIO4`, `GPIO5`
- optional battery ADC: `GPIO34`

Adjust these if your actual wiring differs.

## What The Firmware Does Right Now

- reads the `BMI088`
- estimates pitch and roll from the IMU
- reads the reaction wheel via `AS5600`
- runs reaction-wheel FOC continuously
- drives the base motor through `BTS7960`
- streams live telemetry over WiFi/UDP
- accepts bridge commands over WiFi/UDP
- reports:
  - pitch/roll
  - gyro/accel
  - reaction angle/speed
  - optional base encoder state
  - battery voltage if enabled
  - fault and ESTOP state

## FOC Reality

The reaction-wheel FOC path is implemented, but it is still bring-up grade:

- sensor-based commutation is present
- `initFOC()` and `loopFOC()` are used
- control is `torque_controller = voltage`
- there is no current sensing yet
- motor constants and voltage limits still need real-hardware confirmation

So:

- reaction wheel FOC: yes
- base wheel FOC: not applicable
- fully tuned production motor control: not yet

## Bring-Up Order

1. Edit WiFi settings in `include/robot_config.h`.
2. Confirm the reaction wheel pole-pair count.
3. Keep the robot restrained or lifted.
4. Flash the firmware.
5. Start in `HIL bridge` mode first.
6. Start the PC bridge:

```powershell
python final/hil_bridge.py --esp32-ip <your-esp32-ip> --dashboard-telemetry --dashboard-port 9872
```

7. Start the dashboard:

```powershell
python final/sim_real_dashboard.py --sim-udp-port 9871 --real-udp-port 9872
```

8. Verify:
   - IMU signs
   - reaction motor polarity
   - base motor polarity
   - ESTOP behavior
   - timeout zeroing
9. Only then consider enabling onboard explicit split mode.

## Refreshing Controller Params

Refresh the exported controller header whenever the sim tuning changes:

```powershell
powershell -ExecutionPolicy Bypass -File .\esp32_rw_base_platformio\tools\export_sim_controller.ps1
```

## Known Gaps

1. No current sensing on the reaction-wheel power stage
2. battery sensing is scaffolded but disabled until the divider is wired and calibrated
3. base encoder outer loop is not active by default
4. onboard base-wheel control is intentionally conservative in the checked-in defaults

## Highest-Value Next Additions

1. current sensing for the reaction-wheel stage
2. calibrated battery sensing
3. base-wheel encoder for outer-loop position hold
4. measured motor constants for the real `2804`
