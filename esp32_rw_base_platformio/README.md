# ESP32 Reaction Wheel + Base Wheel PlatformIO Scaffold

This folder is the sim-to-real starting point for the real rig that matches your MuJoCo project direction:

- one reaction wheel on the upper body
- one driven base wheel at the bottom
- ESP32-WROOM dev board
- BMI088 IMU
- AS5600 magnetic encoder on the reaction wheel BLDC
- DRV8313-class 3PWM BLDC stage for the reaction wheel
- BTS7960 H-bridge for the geared base wheel
- 3S Li-Po power rail with buck conversion for logic

The firmware is intentionally split into two practical modes:

1. `HIL bridge` (default): ESP32 handles sensors, motor output, telemetry, and WiFi while your existing Python stack remains the controller.
2. `Onboard wheel-only`: uses the exported controller/estimator contract from `final/export_firmware_params.py` for conservative reaction-wheel bring-up.

## Why it is structured this way

Your current MuJoCo export path assumes richer base-state knowledge than the hardware list currently provides.
The big gap is that the BOM does not include a base-wheel encoder, so full state parity for `base_x/base_vx` is not available yet.

Because of that, the checked-in firmware defaults to:

- keep HIL mode ready now
- keep base-wheel driver ready now
- keep onboard controller conservative
- default onboard control to reaction-wheel-only parity until the base wheel is instrumented

That is the fastest path from sim to real without pretending the missing sensing problem is already solved.

## Folder layout

- `platformio.ini`: ESP32 Arduino PlatformIO project config.
- `include/board_pins.h`: pin map for the exact boards listed above.
- `include/robot_config.h`: control mode, WiFi, power, scaling, and safety settings.
- `include/controller_params.h`: generated controller header exported from the MuJoCo stack.
- `src/main.cpp`: firmware entry point, IMU readout, AS5600 readout, WiFi bridge, onboard loop, safety handling.
- `lib/sim_parity/`: reused C controller/estimator/guard code from `final/firmware/`.
- `tools/export_sim_controller.ps1`: refresh `include/controller_params.h` from the current MuJoCo model.

## Power topology

Recommended wiring topology for the listed hardware:

1. `3S Li-Po (11.1 V nominal, 12.6 V full)` to `BTS7960` motor power.
2. The same battery rail to `DRV8313` motor power input.
3. `12 V -> 5 V buck` to the ESP32 `5V/VIN` input.
4. `BMI088` and `AS5600` on `3.3 V`, not `5 V`.
5. Common ground between battery negative, ESP32 ground, BTS7960 ground, DRV8313 ground, BMI088 ground, and AS5600 ground.

If you use the `MP1584`, set it deliberately for one job only. The cleanest choice is usually a dedicated regulated sensor rail or a spare logic rail. Do not feed the BMI088 or AS5600 from an unknown 5 V rail.

## Pin map

Current default pin map:

- I2C SDA: `GPIO21`
- I2C SCL: `GPIO22`
- DRV8313 enable: `GPIO25`
- DRV8313 phase U/V/W PWM: `GPIO26`, `GPIO27`, `GPIO14`
- BTS7960 RPWM/LPWM: `GPIO32`, `GPIO33`
- BTS7960 REN/LEN: `GPIO18`, `GPIO19`
- Optional battery sense ADC: `GPIO34`

Adjust these in `include/board_pins.h` if your actual PCB/wiring differs.

## Current control contract

The firmware keeps the same basic controller contract as the existing repo:

- estimator state: `x[9]`
- measurement vector: `[pitch, roll, pitch_rate, roll_rate, wheel_rate]`
- controller output: `[reaction_wheel, base_x, base_y]`

On the real rig right now:

- `reaction_wheel -> real reaction wheel`
- `base_x -> real base wheel`
- `base_y -> not physically present in this scaffold`

So the onboard project is checked in with a wheel-only export by default.
That is deliberate.

The onboard code accepts the current exported measurement model and zero-fills the unavailable base-state channels until you add a real base-wheel encoder.

## First bring-up sequence

1. Edit WiFi settings in `include/robot_config.h`.
2. Confirm the reaction wheel pole-pair count in `include/robot_config.h`.
3. Keep the robot physically restrained or lifted.
4. Flash the project with `pio run -t upload`.
5. Start in HIL mode first.
6. Run the PC bridge:

```powershell
python final/hil_bridge.py --esp32-ip <your-esp32-ip> --plot
```

7. Verify IMU signs, reaction wheel polarity, base wheel polarity, and timeout zeroing.
8. Only then switch to onboard wheel-only mode in `include/robot_config.h`.

## Refreshing controller params from MuJoCo

The checked-in header is just a conservative starting point.
Refresh it whenever sim tuning changes:

```powershell
powershell -ExecutionPolicy Bypass -File .\esp32_rw_base_platformio\tools\export_sim_controller.ps1
```

For a future base-wheel-assisted export:

```powershell
powershell -ExecutionPolicy Bypass -File .\esp32_rw_base_platformio\tools\export_sim_controller.ps1 -Profile base_assist
```

## Known gaps you should expect

1. No base-wheel encoder is included in the BOM, so onboard base-state estimation is incomplete.
2. Reaction wheel motor constants are placeholders until you measure the real 2804 motor.
3. The DRV8313 path is using SimpleFOC voltage-mode torque approximation, not closed-loop current control.
4. Battery sensing is scaffolded but disabled until you wire the divider and calibrate the ADC scale.

## Recommended next hardware additions

If you want tighter sim-to-real parity, the highest-value additions are:

1. a base-wheel encoder
2. current sensing on the reaction wheel phase driver
3. a proper battery-voltage divider into the ESP32 ADC
