# Dyno-Style Mapping Workflow

This is the live hardware-mapping workflow for the balancer.
Think of it like a motorcycle dyno session:

1. start from a known map
2. plug the rig in
3. watch live telemetry
4. adjust only the map that matches the observed failure
5. save the map

## The maps

### 1. Sign Map

This is polarity and axis direction.
Touch this first.

- `reaction_sign`
- `drive_sign`
- `pitch_rate_sign`
- `roll_rate_sign`
- `reaction_speed_sign`
- `accel_x_sign`
- `accel_y_sign`
- `accel_z_sign`

If sign map is wrong, no amount of gain tuning will help.

### 2. Authority Map

This is output strength.
Touch this second.

- `rw_cmd_scale`
- `drive_cmd_scale`
- firmware `kReactionHilNormToVoltage`
- firmware `kBasePwmLimitHil`

If sign is correct but hardware is weak or violent, adjust authority map before controller gains.

### 3. Safety Map

This defines when the rig must stop.

- `pitch_estop_deg`
- `roll_estop_deg`
- `comm_estop_s`
- firmware timeout and tilt latch thresholds

This should be conservative first, then widened only after signs and authority are proven.

### 4. Controller Map

This is the balancing character.

- `base_pitch_kp`
- `base_pitch_kd`
- `base_roll_kp`
- `base_roll_kd`
- `wheel_momentum_k`
- `wheel_momentum_thresh_frac`
- `u_bleed`

Only touch this after the first three maps are correct.

### 5. Estimator Map

This is how much the controller trusts the sensors versus the model.

- `estimator_q_scale`
- `estimator_r_scale`
- complementary filter alpha on the ESP / bridge path

If the rig is noisy, laggy, or drifts despite correct signs, estimator map is where to look.

## Session order

1. Run the synthetic smoke:

```powershell
python final/hil_plug_play_smoke.py --bridge-backend stub
```

2. Start a real session from a map:

```powershell
python final/hil_bridge.py --esp32-ip <ESP32_IP> --mapping-profile final/hardware_mapping_template.json --save-mapping-profile final/results/live_map_session.json --plot
```

3. Tune in this order:

- sign map
- authority map
- safety map
- controller map
- estimator map

4. Save the session map when done.

The bridge now supports:

- `--mapping-profile path/to/map.json`
- `--save-mapping-profile path/to/output.json`

So each dyno-style session can start from a known map and end with a new one.

## Failure-to-map routing

| Symptom | Map to adjust first |
|---|---|
| robot reacts the wrong way immediately | sign map |
| robot reacts the right way but too weakly | authority map |
| robot reacts the right way but too violently | authority map |
| robot cuts out too early | safety map |
| robot never cuts out when it should | safety map |
| robot is stable but noisy/jittery | estimator map |
| robot is quiet but slow/sluggish | controller map after authority map |
| robot overshoots and hunts around upright | controller map and estimator map |

