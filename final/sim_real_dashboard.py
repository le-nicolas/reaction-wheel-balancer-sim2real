from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, FancyBboxPatch, Wedge
except Exception as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError(
        "matplotlib is required for the live sim-vs-real dashboard. Install with: pip install matplotlib"
    ) from exc


SIM_COLOR = "#0f766e"
REAL_COLOR = "#d97706"
PITCH_COLOR = "#0f4c81"
ROLL_COLOR = "#b54708"
RATE_PITCH_COLOR = "#0ea5a4"
RATE_ROLL_COLOR = "#f97316"
RW_COLOR = "#7c3aed"
DRIVE_COLOR = "#2563eb"
WHEEL_COLOR = "#4b5563"
BG_COLOR = "#f4efe7"
CARD_COLOR = "#fffaf2"
GRID_COLOR = "#d8d2c4"
TEXT_COLOR = "#1f2937"
MUTED_COLOR = "#6b7280"
GOOD_COLOR = "#2e7d32"
WARN_COLOR = "#f59e0b"
BAD_COLOR = "#c62828"
OFF_COLOR = "#d6d3d1"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live sim-vs-real dashboard with gauges and side-by-side traces.")
    parser.add_argument("--sim-bind", type=str, default="0.0.0.0", help="UDP bind address for simulator telemetry.")
    parser.add_argument("--sim-udp-port", type=int, default=9871, help="UDP port for MuJoCo telemetry stream.")
    parser.add_argument("--real-bind", type=str, default="0.0.0.0", help="UDP bind address for bridge telemetry.")
    parser.add_argument("--real-udp-port", type=int, default=9872, help="UDP port for HIL bridge dashboard stream.")
    parser.add_argument("--window-s", type=float, default=12.0, help="Sliding plot window length in seconds.")
    parser.add_argument("--refresh-ms", type=int, default=60, help="UI refresh interval in milliseconds.")
    parser.add_argument("--max-drain", type=int, default=500, help="Max UDP frames to drain per source per refresh.")
    parser.add_argument(
        "--source-timeout-s",
        type=float,
        default=1.0,
        help="Mark a stream offline if no new frame arrives within this timeout.",
    )
    parser.add_argument(
        "--demo-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run with synthetic telemetry instead of UDP streams.",
    )
    parser.add_argument(
        "--snapshot-png",
        type=str,
        default=None,
        help="Optional path to save a dashboard PNG and exit after --snapshot-after-s.",
    )
    parser.add_argument(
        "--snapshot-after-s",
        type=float,
        default=3.0,
        help="When --snapshot-png is set, run this long before saving the frame.",
    )
    args = parser.parse_args(argv)
    args.sim_udp_port = int(np.clip(int(args.sim_udp_port), 1, 65535))
    args.real_udp_port = int(np.clip(int(args.real_udp_port), 1, 65535))
    args.refresh_ms = int(max(int(args.refresh_ms), 20))
    args.max_drain = int(max(int(args.max_drain), 1))
    args.window_s = float(max(float(args.window_s), 1.0))
    args.source_timeout_s = float(max(float(args.source_timeout_s), 0.2))
    args.snapshot_after_s = float(max(float(args.snapshot_after_s), 0.2))
    return args


def _safe_json_line_to_frame(raw: bytes) -> dict[str, Any] | None:
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        frame = json.loads(text)
    except json.JSONDecodeError:
        return None
    return frame if isinstance(frame, dict) else None


def _first_float(frame: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in frame:
            continue
        try:
            value = float(frame[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def _first_int(frame: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in frame:
            continue
        try:
            return int(frame[key])
        except (TypeError, ValueError):
            continue
    return 0


def _first_bool(frame: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in frame:
            continue
        value = frame[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
    return False


def _value_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _last_finite(values: deque[float]) -> float:
    for value in reversed(values):
        if math.isfinite(value):
            return float(value)
    return float("nan")


def _fmt(value: float, fmt: str) -> str:
    return fmt.format(value) if math.isfinite(value) else "--"


class _UdpReader:
    def __init__(self, bind_host: str, bind_port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, bind_port))
        self._sock.setblocking(False)

    def read_frames(self, max_frames: int) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for _ in range(max_frames):
            try:
                payload, _addr = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            for raw_line in payload.splitlines():
                frame = _safe_json_line_to_frame(raw_line)
                if frame is not None:
                    frames.append(frame)
        return frames

    def close(self) -> None:
        self._sock.close()


@dataclass
class StreamHistory:
    label: str
    t: deque[float]
    pitch_deg: deque[float]
    roll_deg: deque[float]
    pitch_rate_dps: deque[float]
    roll_rate_dps: deque[float]
    wheel_rpm: deque[float]
    rw_cmd: deque[float]
    drive_cmd: deque[float]
    frames_total: int = 0
    last_wall_s: float | None = None
    battery_v: float = float("nan")
    estop: bool = False
    fault: int = 0
    latched: bool = False
    base_encoder_valid: bool = False
    missed_packets: int = 0
    timeout_packets: int = 0
    controller_family: str = ""

    @classmethod
    def create(cls, label: str, maxlen: int = 30000) -> "StreamHistory":
        return cls(
            label=label,
            t=deque(maxlen=maxlen),
            pitch_deg=deque(maxlen=maxlen),
            roll_deg=deque(maxlen=maxlen),
            pitch_rate_dps=deque(maxlen=maxlen),
            roll_rate_dps=deque(maxlen=maxlen),
            wheel_rpm=deque(maxlen=maxlen),
            rw_cmd=deque(maxlen=maxlen),
            drive_cmd=deque(maxlen=maxlen),
        )

    def append_frame(self, frame: dict[str, Any], *, arrival_wall_s: float) -> None:
        time_s = _first_float(frame, "sim_time_s", "time_s", "wall_time_s", "t_s", "time")
        if not math.isfinite(time_s):
            time_s = (self.t[-1] + 0.01) if self.t else 0.0

        pitch_deg = _first_float(frame, "pitch_deg")
        if not math.isfinite(pitch_deg):
            pitch_rad = _first_float(frame, "pitch_rad", "pitch")
            pitch_deg = math.degrees(pitch_rad) if math.isfinite(pitch_rad) else float("nan")

        roll_deg = _first_float(frame, "roll_deg")
        if not math.isfinite(roll_deg):
            roll_rad = _first_float(frame, "roll_rad", "roll")
            roll_deg = math.degrees(roll_rad) if math.isfinite(roll_rad) else float("nan")

        pitch_rate_dps = _first_float(frame, "pitch_rate_dps")
        if not math.isfinite(pitch_rate_dps):
            pitch_rate_rad_s = _first_float(frame, "pitch_rate_rad_s", "pitch_rate")
            pitch_rate_dps = math.degrees(pitch_rate_rad_s) if math.isfinite(pitch_rate_rad_s) else float("nan")

        roll_rate_dps = _first_float(frame, "roll_rate_dps")
        if not math.isfinite(roll_rate_dps):
            roll_rate_rad_s = _first_float(frame, "roll_rate_rad_s", "roll_rate")
            roll_rate_dps = math.degrees(roll_rate_rad_s) if math.isfinite(roll_rate_rad_s) else float("nan")

        wheel_rpm = _first_float(frame, "reaction_speed_rpm", "wheel_rate_rpm")
        if not math.isfinite(wheel_rpm):
            reaction_speed_dps = _first_float(frame, "reaction_speed_dps", "reaction_speed")
            if math.isfinite(reaction_speed_dps):
                wheel_rpm = reaction_speed_dps / 6.0
            else:
                wheel_rate_rad_s = _first_float(frame, "reaction_speed_rad_s", "wheel_rate_rad_s", "wheel_rate")
                wheel_rpm = (wheel_rate_rad_s * 60.0 / (2.0 * math.pi)) if math.isfinite(wheel_rate_rad_s) else float("nan")

        rw_cmd = _first_float(frame, "rw_cmd_norm", "u_rw_cmd", "u_rw", "rt")
        drive_cmd = _first_float(frame, "drive_cmd_norm", "u_bx_cmd", "u_drive", "dt")

        self.t.append(time_s)
        self.pitch_deg.append(pitch_deg)
        self.roll_deg.append(roll_deg)
        self.pitch_rate_dps.append(pitch_rate_dps)
        self.roll_rate_dps.append(roll_rate_dps)
        self.wheel_rpm.append(wheel_rpm)
        self.rw_cmd.append(rw_cmd)
        self.drive_cmd.append(drive_cmd)

        self.frames_total += 1
        self.last_wall_s = arrival_wall_s
        self.battery_v = _first_float(frame, "battery_v")
        self.estop = _first_bool(frame, "estop")
        self.fault = _first_int(frame, "fault")
        self.latched = _first_bool(frame, "latched")
        self.base_encoder_valid = _first_bool(frame, "base_encoder_valid")
        self.missed_packets = _first_int(frame, "missed_packets")
        self.timeout_packets = _first_int(frame, "timeout_packets")
        self.controller_family = str(frame.get("controller_family", self.controller_family))

    def prune(self, window_s: float) -> None:
        if not self.t:
            return
        cutoff = self.t[-1] - max(window_s, 1e-3)
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.pitch_deg.popleft()
            self.roll_deg.popleft()
            self.pitch_rate_dps.popleft()
            self.roll_rate_dps.popleft()
            self.wheel_rpm.popleft()
            self.rw_cmd.popleft()
            self.drive_cmd.popleft()

    def link_alive(self, now_wall_s: float, timeout_s: float) -> bool:
        return self.last_wall_s is not None and (now_wall_s - self.last_wall_s) <= timeout_s


class DemoFeed:
    def __init__(self) -> None:
        self.t = 0.0

    def next_frames(self, dt_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.t += float(max(dt_s, 1e-3))

        pitch_sim = 6.0 * math.sin(0.75 * self.t) + 1.2 * math.sin(0.17 * self.t)
        roll_sim = 4.0 * math.sin(1.12 * self.t + 0.7)
        pitch_rate_sim = 6.0 * 0.75 * math.cos(0.75 * self.t) + 1.2 * 0.17 * math.cos(0.17 * self.t)
        roll_rate_sim = 4.0 * 1.12 * math.cos(1.12 * self.t + 0.7)
        rw_cmd_sim = float(np.clip(-(0.13 * roll_sim + 0.03 * roll_rate_sim), -1.0, 1.0))
        drive_cmd_sim = float(np.clip(-(0.12 * pitch_sim + 0.025 * pitch_rate_sim), -1.0, 1.0))
        wheel_rpm_sim = 420.0 * math.sin(1.45 * self.t) + 130.0 * math.sin(0.33 * self.t)

        pitch_real = 0.92 * pitch_sim + 0.45 * math.sin(1.9 * self.t + 0.3)
        roll_real = 1.06 * roll_sim - 0.35 * math.cos(1.6 * self.t)
        pitch_rate_real = 0.95 * pitch_rate_sim + 2.8 * math.sin(1.5 * self.t + 0.1)
        roll_rate_real = 1.05 * roll_rate_sim + 2.3 * math.cos(1.4 * self.t)
        rw_cmd_real = float(np.clip(rw_cmd_sim + 0.04 * math.sin(0.8 * self.t), -1.0, 1.0))
        drive_cmd_real = float(np.clip(drive_cmd_sim - 0.05 * math.cos(0.9 * self.t), -1.0, 1.0))
        wheel_rpm_real = 0.94 * wheel_rpm_sim + 45.0 * math.sin(0.95 * self.t)

        sim_frame = {
            "schema": "mujoco_telemetry_v1",
            "source": "sim",
            "controller_family": "hardware_explicit_split",
            "sim_time_s": self.t,
            "pitch_deg": pitch_sim,
            "roll_deg": roll_sim,
            "pitch_rate_dps": pitch_rate_sim,
            "roll_rate_dps": roll_rate_sim,
            "wheel_rate_rpm": wheel_rpm_sim,
            "u_rw_cmd": rw_cmd_sim,
            "u_bx_cmd": drive_cmd_sim,
        }
        real_frame = {
            "schema": "sim_real_dashboard_v1",
            "source": "real",
            "controller_family": "hardware_explicit_split",
            "time_s": self.t,
            "pitch_deg": pitch_real,
            "roll_deg": roll_real,
            "pitch_rate_dps": pitch_rate_real,
            "roll_rate_dps": roll_rate_real,
            "reaction_speed_rpm": wheel_rpm_real,
            "rw_cmd_norm": rw_cmd_real,
            "drive_cmd_norm": drive_cmd_real,
            "battery_v": 11.72 + 0.08 * math.sin(0.11 * self.t),
            "fault": 0,
            "latched": 0,
            "estop": 0,
            "base_encoder_valid": 0,
            "missed_packets": 0,
            "timeout_packets": 0,
        }
        return [sim_frame], [real_frame]


def _style_panel_axis(ax) -> None:
    ax.set_facecolor(CARD_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#c9c2b4")
        spine.set_linewidth(1.1)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(True, color=GRID_COLOR, alpha=0.75, linewidth=0.8)


def _autoscale_axis(ax, series: list[deque[float]], *, floor_span: float) -> None:
    values: list[float] = []
    for s in series:
        values.extend(v for v in s if math.isfinite(v))
    if not values:
        return
    arr = np.asarray(values, dtype=float)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return
    if hi <= lo:
        center = float(np.mean(arr))
        span = max(abs(center) * 0.3, floor_span)
        ax.set_ylim(center - span, center + span)
        return
    span = max(hi - lo, floor_span)
    pad = 0.14 * span
    ax.set_ylim(lo - pad, hi + pad)


class GaugeCard:
    def __init__(self, ax, *, title: str, lo: float, hi: float, unit: str, warn_frac: float = 0.62, bad_frac: float = 0.84):
        self.ax = ax
        self.title = title
        self.lo = float(lo)
        self.hi = float(hi)
        self.unit = unit
        self.warn_frac = float(np.clip(warn_frac, 0.0, 1.0))
        self.bad_frac = float(np.clip(bad_frac, self.warn_frac, 1.0))
        self._build()

    def _value_to_angle_deg(self, value: float) -> float:
        frac = (float(np.clip(value, self.lo, self.hi)) - self.lo) / max(self.hi - self.lo, 1e-6)
        return 180.0 - 180.0 * frac

    def _needle_xy(self, value: float, radius: float) -> tuple[float, float]:
        angle = math.radians(self._value_to_angle_deg(value))
        return radius * math.cos(angle), radius * math.sin(angle)

    def _build(self) -> None:
        ax = self.ax
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-0.42, 1.18)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                boxstyle="round,pad=0.018,rounding_size=0.045",
                linewidth=1.2,
                edgecolor="#d6cec1",
                facecolor=CARD_COLOR,
                transform=ax.transAxes,
                zorder=-20,
            )
        )
        ax.text(0.06, 0.92, self.title, transform=ax.transAxes, fontsize=12, fontweight="bold", color=TEXT_COLOR)

        green_lo = 180.0 - 180.0 * (1.0 - self.warn_frac)
        amber_lo = 180.0 - 180.0 * (1.0 - self.bad_frac)
        ax.add_patch(Wedge((0.0, 0.0), 0.94, 180.0, 180.0 - amber_lo, width=0.16, facecolor="#e6b44d", edgecolor="none", alpha=0.80))
        ax.add_patch(Wedge((0.0, 0.0), 0.94, 180.0 - amber_lo, amber_lo, width=0.16, facecolor="#6ba368", edgecolor="none", alpha=0.88))
        ax.add_patch(Wedge((0.0, 0.0), 0.94, amber_lo, 0.0, width=0.16, facecolor="#e6b44d", edgecolor="none", alpha=0.80))
        ax.add_patch(Wedge((0.0, 0.0), 0.94, 180.0, 180.0 - green_lo, width=0.16, facecolor="#ca5a46", edgecolor="none", alpha=0.75))
        ax.add_patch(Wedge((0.0, 0.0), 0.94, green_lo, 0.0, width=0.16, facecolor="#ca5a46", edgecolor="none", alpha=0.75))
        ax.add_patch(Wedge((0.0, 0.0), 0.74, 180.0, 0.0, width=0.01, facecolor="#ffffff", edgecolor="none", alpha=0.95))

        ticks = [self.lo, 0.0 if self.lo < 0.0 < self.hi else 0.5 * (self.lo + self.hi), self.hi]
        for value in ticks:
            angle = math.radians(self._value_to_angle_deg(value))
            x0, y0 = 0.75 * math.cos(angle), 0.75 * math.sin(angle)
            x1, y1 = 0.88 * math.cos(angle), 0.88 * math.sin(angle)
            ax.plot([x0, x1], [y0, y1], color="#6b7280", linewidth=1.0)
            xt, yt = 1.00 * math.cos(angle), 1.00 * math.sin(angle)
            ax.text(xt, yt - 0.02, f"{value:+.0f}", ha="center", va="center", fontsize=8, color=MUTED_COLOR)

        self.sim_needle, = ax.plot([], [], color=SIM_COLOR, linewidth=3.0, solid_capstyle="round")
        self.real_needle, = ax.plot([], [], color=REAL_COLOR, linewidth=3.0, solid_capstyle="round")
        self.hub = Circle((0.0, 0.0), 0.055, facecolor="#fffaf2", edgecolor="#9ca3af", linewidth=1.2, zorder=10)
        ax.add_patch(self.hub)

        ax.add_patch(Circle((0.12, 0.17), 0.018, transform=ax.transAxes, facecolor=SIM_COLOR, edgecolor="none"))
        ax.add_patch(Circle((0.44, 0.17), 0.018, transform=ax.transAxes, facecolor=REAL_COLOR, edgecolor="none"))
        ax.text(0.16, 0.16, "sim", transform=ax.transAxes, fontsize=8, color=MUTED_COLOR, va="center")
        ax.text(0.48, 0.16, "real", transform=ax.transAxes, fontsize=8, color=MUTED_COLOR, va="center")

        self.sim_text = ax.text(0.08, 0.05, "", transform=ax.transAxes, fontsize=12, fontweight="bold", color=SIM_COLOR)
        self.real_text = ax.text(0.42, 0.05, "", transform=ax.transAxes, fontsize=12, fontweight="bold", color=REAL_COLOR)
        self.delta_text = ax.text(0.78, 0.05, "", transform=ax.transAxes, fontsize=9, color=MUTED_COLOR, ha="right")

    def update(self, sim_value: float, real_value: float) -> None:
        sim_value = _value_or_zero(sim_value)
        real_value = _value_or_zero(real_value)
        sx, sy = self._needle_xy(sim_value, 0.79)
        rx, ry = self._needle_xy(real_value, 0.66)
        self.sim_needle.set_data([0.0, sx], [0.0, sy])
        self.real_needle.set_data([0.0, rx], [0.0, ry])
        self.sim_text.set_text(f"{sim_value:+.2f}{self.unit}")
        self.real_text.set_text(f"{real_value:+.2f}{self.unit}")
        self.delta_text.set_text(f"delta {real_value - sim_value:+.2f}")


class StatusPanel:
    def __init__(self, ax, *, timeout_s: float):
        self.ax = ax
        self.timeout_s = timeout_s
        self._build()

    def _build(self) -> None:
        ax = self.ax
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                boxstyle="round,pad=0.018,rounding_size=0.045",
                linewidth=1.2,
                edgecolor="#d6cec1",
                facecolor=CARD_COLOR,
                transform=ax.transAxes,
                zorder=-20,
            )
        )
        ax.text(0.06, 0.92, "Status", transform=ax.transAxes, fontsize=12, fontweight="bold", color=TEXT_COLOR)
        labels = [
            "Sim Link",
            "Real Link",
            "ESTOP",
            "Fault",
            "Base Encoder",
            "Packets",
        ]
        self.light_circles: dict[str, Circle] = {}
        for idx, label in enumerate(labels):
            row = idx % 3
            col = idx // 3
            x = 0.08 + col * 0.44
            y = 0.76 - row * 0.14
            circle = Circle((x, y), 0.028, transform=ax.transAxes, facecolor=OFF_COLOR, edgecolor="#9ca3af", linewidth=1.0)
            ax.add_patch(circle)
            ax.text(x + 0.06, y, label, transform=ax.transAxes, va="center", fontsize=9, color=TEXT_COLOR)
            self.light_circles[label] = circle

        self.summary_text = ax.text(0.06, 0.31, "", transform=ax.transAxes, fontsize=9, color=TEXT_COLOR, va="top")
        self.detail_text = ax.text(0.06, 0.11, "", transform=ax.transAxes, fontsize=9, color=MUTED_COLOR, va="bottom")

    def _set_light(self, label: str, *, active: bool, color: str) -> None:
        circle = self.light_circles[label]
        circle.set_facecolor(color if active else OFF_COLOR)
        circle.set_edgecolor("#374151" if active else "#9ca3af")

    def update(self, sim: StreamHistory, real: StreamHistory, now_wall_s: float) -> None:
        sim_alive = sim.link_alive(now_wall_s, self.timeout_s)
        real_alive = real.link_alive(now_wall_s, self.timeout_s)
        estop = bool(real.estop)
        fault_active = bool(real.fault) or bool(real.latched)
        packets_clean = real_alive and real.missed_packets == 0 and real.timeout_packets == 0
        encoder_ready = bool(real.base_encoder_valid)

        self._set_light("Sim Link", active=sim_alive, color=GOOD_COLOR)
        self._set_light("Real Link", active=real_alive, color=GOOD_COLOR)
        self._set_light("ESTOP", active=estop, color=BAD_COLOR)
        self._set_light("Fault", active=fault_active, color=BAD_COLOR)
        self._set_light("Base Encoder", active=encoder_ready, color=GOOD_COLOR if encoder_ready else WARN_COLOR)
        self._set_light("Packets", active=packets_clean, color=GOOD_COLOR if packets_clean else WARN_COLOR)

        pitch_delta = _last_finite(real.pitch_deg) - _last_finite(sim.pitch_deg)
        roll_delta = _last_finite(real.roll_deg) - _last_finite(sim.roll_deg)
        rw_delta = _last_finite(real.rw_cmd) - _last_finite(sim.rw_cmd)
        drive_delta = _last_finite(real.drive_cmd) - _last_finite(sim.drive_cmd)

        sim_age_ms = (now_wall_s - sim.last_wall_s) * 1000.0 if sim.last_wall_s is not None else float("nan")
        real_age_ms = (now_wall_s - real.last_wall_s) * 1000.0 if real.last_wall_s is not None else float("nan")

        self.summary_text.set_text(
            "\n".join(
                [
                    f"Pitch delta: {_fmt(pitch_delta, '{:+.2f} deg')}",
                    f"Roll delta: {_fmt(roll_delta, '{:+.2f} deg')}",
                    f"RW cmd delta: {_fmt(rw_delta, '{:+.3f}')}",
                    f"Drive delta: {_fmt(drive_delta, '{:+.3f}')}",
                    f"Battery: {_fmt(real.battery_v, '{:.2f} V')}",
                ]
            )
        )
        self.detail_text.set_text(
            "sim frames={sim_frames}  age={sim_age}\nreal frames={real_frames}  age={real_age}\n"
            "fault={fault} latched={latched}  missed={missed} timeout={timeout}\ncontroller={controller}".format(
                sim_frames=sim.frames_total,
                sim_age=_fmt(sim_age_ms, "{:.0f} ms"),
                real_frames=real.frames_total,
                real_age=_fmt(real_age_ms, "{:.0f} ms"),
                fault=int(real.fault),
                latched=int(real.latched),
                missed=int(real.missed_packets),
                timeout=int(real.timeout_packets),
                controller=real.controller_family or sim.controller_family or "--",
            )
        )


def _make_dashboard(args: argparse.Namespace):
    fig = plt.figure(figsize=(18, 10), constrained_layout=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.93, bottom=0.05, wspace=0.16, hspace=0.18)
    fig.suptitle("Sim vs Real Telemetry Dashboard", fontsize=21, fontweight="bold", color=TEXT_COLOR)
    fig.text(
        0.04,
        0.955,
        "Clean operator view for restrained bring-up: large gauges up top, live status on the right, side-by-side sim and HIL traces below.",
        fontsize=10,
        color=MUTED_COLOR,
    )

    outer = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0], width_ratios=[1.0, 1.0, 1.0, 1.25])
    top = outer[0, :].subgridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.35], wspace=0.12)
    ax_g_pitch = fig.add_subplot(top[0, 0])
    ax_g_roll = fig.add_subplot(top[0, 1])
    ax_g_rw = fig.add_subplot(top[0, 2])
    ax_g_drive = fig.add_subplot(top[0, 3])
    ax_status = fig.add_subplot(top[0, 4])

    ax_sim_att = fig.add_subplot(outer[1, 0:2])
    ax_real_att = fig.add_subplot(outer[1, 2:4])
    ax_sim_cmd = fig.add_subplot(outer[2, 0:2])
    ax_real_cmd = fig.add_subplot(outer[2, 2:4])

    ax_sim_rate = ax_sim_att.twinx()
    ax_real_rate = ax_real_att.twinx()
    ax_sim_wheel = ax_sim_cmd.twinx()
    ax_real_wheel = ax_real_cmd.twinx()

    for ax in (ax_sim_att, ax_real_att, ax_sim_cmd, ax_real_cmd, ax_sim_rate, ax_real_rate, ax_sim_wheel, ax_real_wheel):
        _style_panel_axis(ax)

    ax_sim_att.set_title("Simulation Attitude + Rates", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax_real_att.set_title("Real / HIL Attitude + Rates", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax_sim_cmd.set_title("Simulation Commands + Wheel", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax_real_cmd.set_title("Real / HIL Commands + Wheel", fontsize=12, fontweight="bold", color=TEXT_COLOR)

    ax_sim_att.set_ylabel("angle (deg)", color=TEXT_COLOR)
    ax_real_att.set_ylabel("angle (deg)", color=TEXT_COLOR)
    ax_sim_rate.set_ylabel("rate (deg/s)", color=MUTED_COLOR)
    ax_real_rate.set_ylabel("rate (deg/s)", color=MUTED_COLOR)
    ax_sim_cmd.set_ylabel("command (norm)", color=TEXT_COLOR)
    ax_real_cmd.set_ylabel("command (norm)", color=TEXT_COLOR)
    ax_sim_wheel.set_ylabel("wheel (rpm)", color=MUTED_COLOR)
    ax_real_wheel.set_ylabel("wheel (rpm)", color=MUTED_COLOR)
    ax_sim_cmd.set_xlabel("source time (s)", color=TEXT_COLOR)
    ax_real_cmd.set_xlabel("source time (s)", color=TEXT_COLOR)

    for ax in (ax_sim_att, ax_real_att, ax_sim_cmd, ax_real_cmd):
        ax.axhline(0.0, color="#8b8b8b", linewidth=1.0, alpha=0.7)

    sim_pitch_line, = ax_sim_att.plot([], [], color=PITCH_COLOR, linewidth=2.3, label="pitch")
    sim_roll_line, = ax_sim_att.plot([], [], color=ROLL_COLOR, linewidth=2.3, label="roll")
    sim_pr_line, = ax_sim_rate.plot([], [], color=RATE_PITCH_COLOR, linewidth=1.5, linestyle="--", label="pitch rate")
    sim_rr_line, = ax_sim_rate.plot([], [], color=RATE_ROLL_COLOR, linewidth=1.5, linestyle="--", label="roll rate")
    real_pitch_line, = ax_real_att.plot([], [], color=PITCH_COLOR, linewidth=2.3, label="pitch")
    real_roll_line, = ax_real_att.plot([], [], color=ROLL_COLOR, linewidth=2.3, label="roll")
    real_pr_line, = ax_real_rate.plot([], [], color=RATE_PITCH_COLOR, linewidth=1.5, linestyle="--", label="pitch rate")
    real_rr_line, = ax_real_rate.plot([], [], color=RATE_ROLL_COLOR, linewidth=1.5, linestyle="--", label="roll rate")

    sim_rw_line, = ax_sim_cmd.plot([], [], color=RW_COLOR, linewidth=2.2, label="rw cmd")
    sim_drive_line, = ax_sim_cmd.plot([], [], color=DRIVE_COLOR, linewidth=2.0, label="drive cmd")
    sim_wheel_line, = ax_sim_wheel.plot([], [], color=WHEEL_COLOR, linewidth=1.8, linestyle=":", label="wheel rpm")
    real_rw_line, = ax_real_cmd.plot([], [], color=RW_COLOR, linewidth=2.2, label="rw cmd")
    real_drive_line, = ax_real_cmd.plot([], [], color=DRIVE_COLOR, linewidth=2.0, label="drive cmd")
    real_wheel_line, = ax_real_wheel.plot([], [], color=WHEEL_COLOR, linewidth=1.8, linestyle=":", label="wheel rpm")

    ax_sim_att.legend(
        [sim_pitch_line, sim_roll_line, sim_pr_line, sim_rr_line],
        ["pitch", "roll", "pitch rate", "roll rate"],
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
    )
    ax_real_att.legend(
        [real_pitch_line, real_roll_line, real_pr_line, real_rr_line],
        ["pitch", "roll", "pitch rate", "roll rate"],
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
    )
    ax_sim_cmd.legend(
        [sim_rw_line, sim_drive_line, sim_wheel_line],
        ["rw cmd", "drive cmd", "wheel rpm"],
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
    )
    ax_real_cmd.legend(
        [real_rw_line, real_drive_line, real_wheel_line],
        ["rw cmd", "drive cmd", "wheel rpm"],
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
    )

    gauges = {
        "pitch": GaugeCard(ax_g_pitch, title="Pitch", lo=-20.0, hi=20.0, unit=" deg"),
        "roll": GaugeCard(ax_g_roll, title="Roll", lo=-20.0, hi=20.0, unit=" deg"),
        "rw": GaugeCard(ax_g_rw, title="Reaction Cmd", lo=-1.0, hi=1.0, unit=""),
        "drive": GaugeCard(ax_g_drive, title="Drive Cmd", lo=-1.0, hi=1.0, unit=""),
    }
    status = StatusPanel(ax_status, timeout_s=args.source_timeout_s)

    artists = {
        "sim_pitch_line": sim_pitch_line,
        "sim_roll_line": sim_roll_line,
        "sim_pr_line": sim_pr_line,
        "sim_rr_line": sim_rr_line,
        "real_pitch_line": real_pitch_line,
        "real_roll_line": real_roll_line,
        "real_pr_line": real_pr_line,
        "real_rr_line": real_rr_line,
        "sim_rw_line": sim_rw_line,
        "sim_drive_line": sim_drive_line,
        "sim_wheel_line": sim_wheel_line,
        "real_rw_line": real_rw_line,
        "real_drive_line": real_drive_line,
        "real_wheel_line": real_wheel_line,
        "ax_sim_att": ax_sim_att,
        "ax_real_att": ax_real_att,
        "ax_sim_rate": ax_sim_rate,
        "ax_real_rate": ax_real_rate,
        "ax_sim_cmd": ax_sim_cmd,
        "ax_real_cmd": ax_real_cmd,
        "ax_sim_wheel": ax_sim_wheel,
        "ax_real_wheel": ax_real_wheel,
    }
    return fig, gauges, status, artists


def main(argv=None) -> int:
    args = parse_args(argv)
    sim_history = StreamHistory.create("sim")
    real_history = StreamHistory.create("real")
    fig, gauges, status, artists = _make_dashboard(args)

    sim_reader = None if args.demo_mode else _UdpReader(args.sim_bind, args.sim_udp_port)
    real_reader = None if args.demo_mode else _UdpReader(args.real_bind, args.real_udp_port)
    demo_feed = DemoFeed() if args.demo_mode else None
    last_refresh_wall = time.perf_counter()

    def _ingest_frames(now_wall_s: float) -> None:
        nonlocal last_refresh_wall
        dt_s = max(now_wall_s - last_refresh_wall, args.refresh_ms / 1000.0)
        last_refresh_wall = now_wall_s
        if demo_feed is not None:
            sim_frames, real_frames = demo_feed.next_frames(dt_s)
        else:
            sim_frames = sim_reader.read_frames(args.max_drain) if sim_reader is not None else []
            real_frames = real_reader.read_frames(args.max_drain) if real_reader is not None else []

        for frame in sim_frames:
            sim_history.append_frame(frame, arrival_wall_s=now_wall_s)
        for frame in real_frames:
            real_history.append_frame(frame, arrival_wall_s=now_wall_s)

        sim_history.prune(args.window_s)
        real_history.prune(args.window_s)

    def _set_time_window(ax, x_values: deque[float]) -> None:
        if not x_values:
            return
        xmax = float(x_values[-1])
        xmin = max(0.0, xmax - args.window_s)
        ax.set_xlim(xmin, xmax if xmax > xmin else xmin + 1e-3)

    def _refresh(_frame_id: int):
        now_wall_s = time.perf_counter()
        _ingest_frames(now_wall_s)

        sim_x = list(sim_history.t)
        real_x = list(real_history.t)

        artists["sim_pitch_line"].set_data(sim_x, list(sim_history.pitch_deg))
        artists["sim_roll_line"].set_data(sim_x, list(sim_history.roll_deg))
        artists["sim_pr_line"].set_data(sim_x, list(sim_history.pitch_rate_dps))
        artists["sim_rr_line"].set_data(sim_x, list(sim_history.roll_rate_dps))
        artists["real_pitch_line"].set_data(real_x, list(real_history.pitch_deg))
        artists["real_roll_line"].set_data(real_x, list(real_history.roll_deg))
        artists["real_pr_line"].set_data(real_x, list(real_history.pitch_rate_dps))
        artists["real_rr_line"].set_data(real_x, list(real_history.roll_rate_dps))

        artists["sim_rw_line"].set_data(sim_x, list(sim_history.rw_cmd))
        artists["sim_drive_line"].set_data(sim_x, list(sim_history.drive_cmd))
        artists["sim_wheel_line"].set_data(sim_x, list(sim_history.wheel_rpm))
        artists["real_rw_line"].set_data(real_x, list(real_history.rw_cmd))
        artists["real_drive_line"].set_data(real_x, list(real_history.drive_cmd))
        artists["real_wheel_line"].set_data(real_x, list(real_history.wheel_rpm))

        _set_time_window(artists["ax_sim_att"], sim_history.t)
        _set_time_window(artists["ax_sim_cmd"], sim_history.t)
        _set_time_window(artists["ax_real_att"], real_history.t)
        _set_time_window(artists["ax_real_cmd"], real_history.t)

        _autoscale_axis(artists["ax_sim_att"], [sim_history.pitch_deg, sim_history.roll_deg], floor_span=6.0)
        _autoscale_axis(artists["ax_real_att"], [real_history.pitch_deg, real_history.roll_deg], floor_span=6.0)
        _autoscale_axis(artists["ax_sim_rate"], [sim_history.pitch_rate_dps, sim_history.roll_rate_dps], floor_span=25.0)
        _autoscale_axis(artists["ax_real_rate"], [real_history.pitch_rate_dps, real_history.roll_rate_dps], floor_span=25.0)
        artists["ax_sim_cmd"].set_ylim(-1.12, 1.12)
        artists["ax_real_cmd"].set_ylim(-1.12, 1.12)
        _autoscale_axis(artists["ax_sim_wheel"], [sim_history.wheel_rpm], floor_span=60.0)
        _autoscale_axis(artists["ax_real_wheel"], [real_history.wheel_rpm], floor_span=60.0)

        gauges["pitch"].update(_last_finite(sim_history.pitch_deg), _last_finite(real_history.pitch_deg))
        gauges["roll"].update(_last_finite(sim_history.roll_deg), _last_finite(real_history.roll_deg))
        gauges["rw"].update(_last_finite(sim_history.rw_cmd), _last_finite(real_history.rw_cmd))
        gauges["drive"].update(_last_finite(sim_history.drive_cmd), _last_finite(real_history.drive_cmd))
        status.update(sim_history, real_history, now_wall_s)

        return tuple(fig.axes)

    try:
        if args.snapshot_png:
            deadline = time.perf_counter() + args.snapshot_after_s
            while time.perf_counter() < deadline:
                _refresh(0)
                fig.canvas.draw()
                time.sleep(args.refresh_ms / 1000.0)
            out_path = Path(args.snapshot_png).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(f"Saved dashboard snapshot: {out_path}")
            return 0

        anim = FuncAnimation(fig, _refresh, interval=args.refresh_ms, blit=False, cache_frame_data=False)
        _ = anim
        plt.show()
        return 0
    finally:
        if sim_reader is not None:
            sim_reader.close()
        if real_reader is not None:
            real_reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
