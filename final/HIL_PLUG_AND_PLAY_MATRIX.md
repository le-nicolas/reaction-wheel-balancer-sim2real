# HIL Plug-And-Play Matrix

Use this before flashing real hardware.
The goal is to prove that transport, signs, ESTOP logic, and the first tuning knobs are already known.

## Situation Map

| Situation | Expected bridge response | If wrong, adjust here first |
|---|---|---|
| Idle upright | `rw_cmd` and `drive_cmd` stay near zero | BMI088 axis signs, complementary-filter signs, `--pitch-rate-sign`, `--roll-rate-sign`, `--accel-x-sign`, `--accel-y-sign` |
| Positive pitch | `drive_cmd` should oppose pitch with clear magnitude | `--drive-sign`, `kInvertBaseCommand`, then pitch sensing signs |
| Negative pitch | `drive_cmd` should flip sign relative to positive pitch | same as above; if it does not flip, signs are inconsistent |
| Positive roll | `rw_cmd` should oppose roll with clear magnitude | `--reaction-sign`, `kInvertReactionCommand`, then roll sensing signs |
| Negative roll | `rw_cmd` should flip sign relative to positive roll | same as above; if it does not flip, signs are inconsistent |
| Packet loss / timeout | commands go to zero and ESTOP becomes active | `--zero-on-timeout`, `--comm-estop-s`, firmware `kCommandTimeoutUs`, IP/port pairing |
| Large tilt | ESTOP becomes active and outputs stay zero | `--pitch-estop-deg`, `--roll-estop-deg`, firmware tilt thresholds |
| Response too weak | sign is correct but motion would be too soft | `kReactionHilNormToVoltage`, `kBasePwmLimitHil`, `--rw-cmd-scale`, `--drive-cmd-scale` |
| Response too aggressive | sign is correct but commands saturate too easily | lower the same command-scale/voltage-limit knobs first |
| Reaction wheel direction wrong | roll correction sign is reversed | `--reaction-sign`, `kInvertReactionCommand` |
| Base wheel direction wrong | pitch correction sign is reversed | `--drive-sign`, `kInvertBaseCommand` |
| Telemetry arrives but looks noisy | commands jitter at idle | BMI088 mounting/signs, `kComplementaryAlpha`, sensor wiring, grounding |

## Adjustment Order

1. Fix packet transport and timeout behavior.
2. Fix IMU sign conventions.
3. Fix actuator polarity.
4. Only then change command strength.
5. Only after that move from stub/HIL smoke into the real runtime backend.

## Scripts

Synthetic real-time smoke:

```powershell
python final/hil_plug_play_smoke.py --bridge-backend stub
```

Then, once transport/signs are clean, try the runtime backend:

```powershell
python final/hil_plug_play_smoke.py --bridge-backend runtime
```

The synthetic smoke does not prove physical stability.
It proves that the software stack already knows where the first real-time adjustments must happen.

