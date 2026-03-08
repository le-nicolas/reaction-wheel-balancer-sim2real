# Self-Balancing Unicycle Robot

![Status](https://img.shields.io/badge/status-active_research-orange)
![Scope](https://img.shields.io/badge/scope-sim--to--real_in_progress-blue)
![Maintained](https://img.shields.io/badge/maintained_paths-final%2F_+_esp32__rw__base__platformio-green)
![Evidence](https://img.shields.io/badge/claims-artifact_backed-brightgreen)

## Current Status

Authoritative status for this repo as of `2026-03-08`:

- This repo is no longer just simulation-only. It is a sim-first stack with an active HIL and ESP32 bring-up path.
- The maintained software paths are:
  - `final/`
  - `esp32_rw_base_platformio/`
- The intended real robot architecture is:
  - base wheel corrects pitch
  - reaction wheel corrects roll
- The current sim-to-real controller branch is `hardware_explicit_split`.
- For restrained bring-up, the base wheel does not require an encoder for the inner balance loop. IMU pitch angle and pitch rate are enough to drive meaningful pitch correction into the base wheel.
- A base encoder is still recommended later for position hold, drift suppression, and better outer-loop parity.

If you find older notes saying that pitch was not being driven into the base wheel correctly, treat those notes as historical. That problem was the earlier state, not the current one.

## What This Repo Contains

- `final/`
  - MuJoCo model and runtime controller
  - benchmark/evaluation tooling
  - HIL bridge
  - synthetic smoke tests
  - telemetry plotter
  - live sim-vs-real dashboard
- `esp32_rw_base_platformio/`
  - PlatformIO firmware scaffold for the ESP32 rig
  - BMI088 + AS5600 sensor path
  - reaction-wheel FOC path
  - base-wheel BTS7960 drive path
- `docs/`
  - workflow and analysis docs
- `archive/`
  - historical prototypes and investigations, not the recommended entry point

## What Is Working Now

- MuJoCo simulation and headless benchmarking
- `hardware_explicit_split` actuator routing:
  - pitch -> base wheel
  - roll -> reaction wheel
- HIL bridge with live mapping and ESTOP behavior
- live sim-vs-real dashboard with gauges, status lights, and side-by-side traces
- PlatformIO ESP32 scaffold with WiFi telemetry
- reaction-wheel FOC via SimpleFOC sensor-based commutation

## What Is Still Limited

- This is still a bring-up and research stack, not a finished hardware product.
- The reaction-wheel FOC path is voltage-mode torque control, not closed-loop current control.
- The base wheel can balance from IMU feedback alone, but it will drift without an encoder-based outer loop.
- The richer `current` family can still score better than `hardware_explicit_split` in some pure simulation benchmark modes.
- Real hardware still needs staged restraint testing, sign checks, and live gain adjustment.

## Architecture

Current intended hardware split:

- base wheel:
  - geared DC motor via `BTS7960`
  - handles pitch stabilization
- reaction wheel:
  - BLDC via `DRV8313`-class 3PWM stage
  - `AS5600` magnetic encoder
  - handles roll stabilization
- controller stack:
  - MuJoCo remains the reference environment
  - `hil_bridge.py` is the live mapping layer
  - `sim_real_dashboard.py` is the operator dashboard

## Quick Start

Run the simulator:

```powershell
python final/final.py --mode smooth
```

Run the sim telemetry plotter:

```powershell
python final/telemetry_plotter.py --source udp --udp-port 9871 --window-s 12
python final/final.py --mode smooth --telemetry --telemetry-transport udp --telemetry-udp-host 127.0.0.1 --telemetry-udp-port 9871
```

Run the cleaner sim-vs-real dashboard:

```powershell
python final/sim_real_dashboard.py --sim-udp-port 9871 --real-udp-port 9872
python final/hil_bridge.py --esp32-ip <ESP32_IP> --dashboard-telemetry --dashboard-port 9872
```

Run the one-command launcher:

```powershell
python final/play_everything.py --ui dashboard --with-hil --no-hil-plot --esp32-ip <ESP32_IP>
```

Run the synthetic HIL smoke test:

```powershell
python final/hil_plug_play_smoke.py --bridge-backend runtime
```

Run the benchmark used for the latest sim comparison:

```powershell
python final/benchmark.py --benchmark-profile fast_pr --episodes 100 --steps 2500 --trials 0 --controller-families current,hardware_explicit_split --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
```

## Latest Reference Artifacts

The current benchmark snapshot used in the latest README updates is:

- `final/results/benchmark_20260308_152510.csv`
- `final/results/benchmark_20260308_152510_pareto.png`
- `final/results/sim_vs_hardware_explicit_split_100ep.png`
- `final/results/sim_real_dashboard_demo.png`

High-level interpretation:

- both `current` and `hardware_explicit_split` survived all `100/100` nominal episodes
- `hardware_explicit_split` is the correct branch for your real split-actuation hardware
- `current` still performs better in some low-spin pure-sim cases

## Read This Next

- `final/README.md`
- `esp32_rw_base_platformio/README.md`
- `final/hil_bridge.py`
- `final/sim_real_dashboard.py`
- `final/hil_plug_play_smoke.py`
- `final/DYNO_MAPPING_WORKFLOW.md`
- `final/HIL_PLUG_AND_PLAY_MATRIX.md`

## Boundaries

- Do not treat archived files as the current implementation.
- Do not treat pure simulation results as automatic hardware proof.
- Do not flash unverified hardware changes without restrained bring-up.
- Use benchmark artifacts, smoke tests, and telemetry logs as the source of truth for claims.
