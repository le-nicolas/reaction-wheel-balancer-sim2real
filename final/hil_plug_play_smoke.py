from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


G_M_S2 = 9.81


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    evaluator: Callable[[list[dict[str, float | int | bool]]], dict[str, object]]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time synthetic HIL smoke runner for plug-and-play checks.")
    parser.add_argument("--pc-ip", type=str, default="127.0.0.1")
    parser.add_argument("--pc-port", type=int, default=56005)
    parser.add_argument("--esp-ip", type=str, default="127.0.0.1")
    parser.add_argument("--esp-port", type=int, default=56006)
    parser.add_argument("--loop-hz", type=float, default=120.0)
    parser.add_argument("--bridge-backend", choices=("stub", "runtime"), default="stub")
    parser.add_argument(
        "--drive-channel",
        choices=("bx", "by", "sum"),
        default="bx",
        help="Bridge-side runtime drive channel to test.",
    )
    parser.add_argument(
        "--mapping-profile",
        type=str,
        default=None,
        help="Optional bridge mapping profile JSON to preload before the smoke test.",
    )
    parser.add_argument(
        "--runtime-args",
        type=str,
        default="--mode robust --hardware-safe --control-hz 250 --controller-family hardware_explicit_split",
        help="Forwarded only when --bridge-backend runtime.",
    )
    parser.add_argument("--outdir", type=str, default="final/results")
    return parser.parse_args()


def _attitude_to_accel(pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
    pitch = np.radians(float(pitch_deg))
    roll = np.radians(float(roll_deg))
    ax = G_M_S2 * np.sin(pitch)
    ay = G_M_S2 * np.sin(roll)
    az = G_M_S2 * np.cos(pitch) * np.cos(roll)
    return float(ax), float(ay), float(az)


def _scenario_state(name: str, t_s: float) -> dict[str, float | bool]:
    if name == "idle":
        return {"pitch_deg": 0.0, "roll_deg": 0.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "pitch_forward":
        return {"pitch_deg": 8.0, "roll_deg": 0.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "pitch_backward":
        return {"pitch_deg": -8.0, "roll_deg": 0.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "roll_right":
        return {"pitch_deg": 0.0, "roll_deg": 8.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "roll_left":
        return {"pitch_deg": 0.0, "roll_deg": -8.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "pitch_estop":
        return {"pitch_deg": 55.0, "roll_deg": 0.0, "pitch_rate_dps": 0.0, "roll_rate_dps": 0.0, "send_packet": True}
    if name == "timeout":
        return {
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "pitch_rate_dps": 0.0,
            "roll_rate_dps": 0.0,
            "send_packet": bool(t_s < 0.8),
        }
    raise ValueError(f"Unknown scenario: {name}")


def _tail(records: list[dict[str, float | int | bool]], frac: float = 0.45) -> list[dict[str, float | int | bool]]:
    if not records:
        return []
    start = int(max(len(records) * (1.0 - frac), 0))
    return records[start:]


def _mean(records: list[dict[str, float | int | bool]], key: str) -> float:
    vals = [float(r[key]) for r in records if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def _max_abs(records: list[dict[str, float | int | bool]], key: str) -> float:
    vals = [abs(float(r[key])) for r in records if key in r]
    return float(np.max(vals)) if vals else float("nan")


def _estop_frac(records: list[dict[str, float | int | bool]]) -> float:
    vals = [1.0 if bool(r.get("estop", False)) else 0.0 for r in records]
    return float(np.mean(vals)) if vals else float("nan")


def _base_summary(records: list[dict[str, float | int | bool]]) -> dict[str, float]:
    tail = _tail(records)
    return {
        "samples": float(len(records)),
        "tail_samples": float(len(tail)),
        "rw_mean": _mean(tail, "rt"),
        "drive_mean": _mean(tail, "dt"),
        "rw_peak_abs": _max_abs(records, "rt"),
        "drive_peak_abs": _max_abs(records, "dt"),
        "rw_tail_peak_abs": _max_abs(tail, "rt"),
        "drive_tail_peak_abs": _max_abs(tail, "dt"),
        "estop_fraction": _estop_frac(records),
    }


def _result(pass_cond: bool, summary: dict[str, float], hints: list[str]) -> dict[str, object]:
    return {"pass": bool(pass_cond), "summary": summary, "adjust_if_wrong": hints}


def eval_idle(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = s["samples"] >= 20 and s["estop_fraction"] < 0.05 and abs(s["rw_mean"]) < 0.10 and abs(s["drive_mean"]) < 0.10
    return _result(
        ok,
        s,
        [
            "If idle commands are not near zero, check BMI088 axis mapping and complementary-filter signs first.",
            "If only base command drifts, adjust `--pitch-rate-sign`, `--accel-x-sign`, `--drive-sign`, or `kInvertBaseCommand`.",
            "If only reaction command drifts, adjust `--roll-rate-sign`, `--accel-y-sign`, `--reaction-sign`, or `kInvertReactionCommand`.",
        ],
    )


def eval_pitch_forward(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = s["samples"] >= 20 and s["estop_fraction"] < 0.05 and s["drive_mean"] < -0.20 and abs(s["rw_mean"]) < 0.20
    return _result(
        ok,
        s,
        [
            "If `drive_mean` has the wrong sign, invert base motor direction with `--drive-sign` or `kInvertBaseCommand`.",
            "If the sign still looks wrong, your pitch sensing sign is wrong: adjust `--pitch-rate-sign` and `--accel-x-sign`.",
            "If the sign is correct but too weak, increase `kBasePwmLimitHil` or the bridge-side `--drive-cmd-scale`.",
        ],
    )


def eval_pitch_backward(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = s["samples"] >= 20 and s["estop_fraction"] < 0.05 and s["drive_mean"] > 0.20 and abs(s["rw_mean"]) < 0.20
    return _result(
        ok,
        s,
        [
            "If `drive_mean` does not flip sign versus forward pitch, base direction or pitch-axis signs are inconsistent.",
            "Adjust `--drive-sign` / `kInvertBaseCommand`, then verify `--pitch-rate-sign` and `--accel-x-sign`.",
        ],
    )


def eval_roll_right(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = s["samples"] >= 20 and s["estop_fraction"] < 0.05 and s["rw_mean"] < -0.20 and abs(s["drive_mean"]) < 0.20
    return _result(
        ok,
        s,
        [
            "If `rw_mean` has the wrong sign, invert reaction-wheel command with `--reaction-sign` or `kInvertReactionCommand`.",
            "If the sign still looks wrong, adjust roll sensing signs: `--roll-rate-sign` and `--accel-y-sign`.",
            "If too weak, raise `kReactionHilNormToVoltage` or bridge-side `--rw-cmd-scale` after restraint testing.",
        ],
    )


def eval_roll_left(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = s["samples"] >= 20 and s["estop_fraction"] < 0.05 and s["rw_mean"] > 0.20 and abs(s["drive_mean"]) < 0.20
    return _result(
        ok,
        s,
        [
            "If reaction command does not flip sign versus right-roll, the reaction motor polarity or roll signs are wrong.",
            "Adjust `--reaction-sign` / `kInvertReactionCommand`, then verify `--roll-rate-sign` and `--accel-y-sign`.",
        ],
    )


def eval_pitch_estop(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = (
        s["samples"] >= 20
        and s["estop_fraction"] > 0.50
        and s["rw_tail_peak_abs"] < 0.10
        and s["drive_tail_peak_abs"] < 0.10
    )
    return _result(
        ok,
        s,
        [
            "If ESTOP does not trip on large pitch, lower `--pitch-estop-deg` in `hil_bridge.py` or firmware tilt limits.",
            "If ESTOP trips but commands are not zeroed, inspect timeout/ESTOP gating in the bridge and firmware output stage.",
        ],
    )


def eval_timeout(records: list[dict[str, float | int | bool]]) -> dict[str, object]:
    s = _base_summary(records)
    ok = (
        s["samples"] >= 20
        and s["estop_fraction"] > 0.20
        and s["rw_tail_peak_abs"] < 0.05
        and s["drive_tail_peak_abs"] < 0.05
    )
    return _result(
        ok,
        s,
        [
            "If commands do not zero on packet loss, verify `--zero-on-timeout`, `--comm-estop-s`, and firmware `kCommandTimeoutUs`.",
            "If timeout ESTOP happens too early or too late, tune `--comm-estop-s` and `kCommandTimeoutUs` together.",
            "If no packets are arriving, check IP/port pairing: bridge `--pc-port`/`--esp32-port` versus firmware telemetry/command ports.",
        ],
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("idle", 1.4, eval_idle, "Upright calm state. Commands should settle near zero."),
    Scenario("pitch_forward", 1.5, eval_pitch_forward, "Positive pitch should drive the base command in the corrective direction."),
    Scenario("pitch_backward", 1.5, eval_pitch_backward, "Negative pitch should flip the base command sign."),
    Scenario("roll_right", 1.5, eval_roll_right, "Positive roll should drive the reaction-wheel command in the corrective direction."),
    Scenario("roll_left", 1.5, eval_roll_left, "Negative roll should flip the reaction-wheel command sign."),
    Scenario("pitch_estop", 1.2, eval_pitch_estop, "Extreme tilt should force ESTOP and zero outputs."),
    Scenario("timeout", 1.6, eval_timeout, "Packet loss should zero commands and raise ESTOP."),
)


class FakeEsp32Rig:
    def __init__(self, *, pc_addr: tuple[str, int], listen_addr: tuple[str, int], loop_hz: float):
        self.pc_addr = pc_addr
        self.loop_hz = float(loop_hz)
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind(listen_addr)
        self.recv_sock.setblocking(False)
        self.seq = 0
        self.command_records: list[dict[str, float | int | bool]] = []

    def close(self) -> None:
        self.send_sock.close()
        self.recv_sock.close()

    def _drain_commands(self, now_s: float) -> None:
        while True:
            try:
                payload, _addr = self.recv_sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                frame = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            self.command_records.append(
                {
                    "t_s": now_s,
                    "rt": float(frame.get("rt", 0.0)),
                    "dt": float(frame.get("dt", 0.0)),
                    "estop": bool(frame.get("estop", 0)),
                    "seq": int(frame.get("seq", 0)),
                }
            )

    def run_scenario(self, scenario: Scenario) -> list[dict[str, float | int | bool]]:
        self.command_records = []
        start = time.perf_counter()
        period = 1.0 / max(self.loop_hz, 1e-6)
        next_tick = start
        while True:
            now = time.perf_counter()
            t_s = now - start
            if t_s >= scenario.duration_s:
                break
            state = _scenario_state(scenario.name, t_s)
            if bool(state["send_packet"]):
                ax, ay, az = _attitude_to_accel(float(state["pitch_deg"]), float(state["roll_deg"]))
                frame = {
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "gx": float(state["roll_rate_dps"]),
                    "gy": float(state["pitch_rate_dps"]),
                    "gz": 0.0,
                    "reaction_speed": 0.0,
                    "base_pos_m": 0.0,
                    "base_vel_m_s": 0.0,
                    "base_encoder_valid": 0,
                    "ts": int(t_s * 1_000_000.0),
                    "seq": self.seq,
                }
                self.seq += 1
                self.send_sock.sendto(json.dumps(frame, separators=(",", ":")).encode("utf-8"), self.pc_addr)
            self._drain_commands(t_s)
            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        settle_deadline = time.perf_counter() + 0.3
        while time.perf_counter() < settle_deadline:
            self._drain_commands(scenario.duration_s)
            time.sleep(0.01)
        return list(self.command_records)


def _build_bridge_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("hil_bridge.py")),
        "--pc-ip",
        args.pc_ip,
        "--pc-port",
        str(args.pc_port),
        "--esp32-ip",
        args.esp_ip,
        "--esp32-port",
        str(args.esp_port),
        "--loop-hz",
        str(int(max(args.loop_hz, 20.0))),
        "--imu-alpha",
        "0.98",
        "--zero-on-timeout",
        "--comm-estop-s",
        "0.25",
        "--pitch-estop-deg",
        "35",
        "--roll-estop-deg",
        "35",
        "--drive-channel",
        args.drive_channel,
        "--no-plot",
    ]
    if args.mapping_profile:
        cmd.extend(["--mapping-profile", args.mapping_profile])
    if args.bridge_backend == "stub":
        cmd.append("--stub-control")
    else:
        cmd.extend(["--runtime-args", args.runtime_args])
    return cmd


def run_bridge(args: argparse.Namespace, scenario: Scenario) -> dict[str, object]:
    proc = subprocess.Popen(
        _build_bridge_cmd(args),
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rig = FakeEsp32Rig(pc_addr=(args.pc_ip, args.pc_port), listen_addr=(args.esp_ip, args.esp_port), loop_hz=args.loop_hz)
    try:
        time.sleep(0.60)
        records = rig.run_scenario(scenario)
        time.sleep(0.15)
    finally:
        rig.close()
        bridge_exited_early = proc.poll() is not None
        if not bridge_exited_early:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
    output = proc.stdout.read() if proc.stdout is not None else ""
    result = scenario.evaluator(records)
    result["scenario"] = scenario.name
    result["description"] = scenario.description
    result["bridge_backend"] = args.bridge_backend
    result["bridge_returncode"] = int(proc.returncode)
    result["bridge_exited_early"] = bool(bridge_exited_early)
    result["bridge_output_tail"] = output.strip().splitlines()[-8:]
    return result


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = [run_bridge(args, scenario) for scenario in SCENARIOS]
    overall_pass = all(bool(r["pass"]) and (not bool(r["bridge_exited_early"])) for r in results)

    report = {
        "timestamp": timestamp,
        "bridge_backend": args.bridge_backend,
        "overall_pass": overall_pass,
        "pc_addr": f"{args.pc_ip}:{args.pc_port}",
        "esp_addr": f"{args.esp_ip}:{args.esp_port}",
        "scenarios": results,
    }
    out_json = outdir / f"hil_plug_play_smoke_{timestamp}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"bridge_backend={args.bridge_backend}")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['scenario']}: pass={int(result['pass'])} "
            f"rw_mean={summary['rw_mean']:+.3f} drive_mean={summary['drive_mean']:+.3f} "
            f"rw_tail_peak={summary['rw_tail_peak_abs']:.3f} drive_tail_peak={summary['drive_tail_peak_abs']:.3f} "
            f"estop_frac={summary['estop_fraction']:.3f} bridge_early={int(result['bridge_exited_early'])}"
        )
    print(f"overall_pass={int(overall_pass)}")
    print(f"report={out_json}")


if __name__ == "__main__":
    main()
