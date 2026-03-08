import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRACE_EVENT_CONTROL_UPDATE = "control_update"


@dataclass(frozen=True)
class TraceRow:
    step: int
    sim_time_s: float
    pitch: float
    roll: float
    pitch_rate: float
    roll_rate: float
    wheel_rate: float
    base_x: float
    base_y: float
    u_rw: float
    u_bx: float
    u_by: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build residual training dataset from paired baseline/teacher traces.")
    parser.add_argument("--baseline-trace", type=str, required=True, help="Trace CSV from baseline controller run.")
    parser.add_argument("--teacher-trace", type=str, required=True, help="Trace CSV from teacher controller run.")
    parser.add_argument("--out", type=str, required=True, help="Output dataset .npz path.")
    parser.add_argument("--residual-gate-tilt-deg", type=float, default=8.0)
    parser.add_argument("--residual-gate-rate", type=float, default=3.0)
    parser.add_argument("--max-target-rw", type=float, default=6.0)
    parser.add_argument("--max-target-bx", type=float, default=1.0)
    parser.add_argument("--max-target-by", type=float, default=1.0)
    parser.add_argument(
        "--allow-outside-gate",
        action="store_true",
        help="Keep samples outside runtime residual gate region.",
    )
    return parser.parse_args()


def _f(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _i(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def load_control_rows(path: Path) -> dict[int, TraceRow]:
    out: dict[int, TraceRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = str(row.get("event", "")).strip()
            if event and event != TRACE_EVENT_CONTROL_UPDATE:
                continue
            tr = TraceRow(
                step=_i(row, "step"),
                sim_time_s=_f(row, "sim_time_s"),
                pitch=_f(row, "pitch"),
                roll=_f(row, "roll"),
                pitch_rate=_f(row, "pitch_rate"),
                roll_rate=_f(row, "roll_rate"),
                wheel_rate=_f(row, "wheel_rate"),
                base_x=_f(row, "base_x"),
                base_y=_f(row, "base_y"),
                u_rw=_f(row, "u_rw"),
                u_bx=_f(row, "u_bx"),
                u_by=_f(row, "u_by"),
            )
            if tr.step < 0:
                continue
            if not np.all(np.isfinite(np.array(
                [
                    tr.sim_time_s,
                    tr.pitch,
                    tr.roll,
                    tr.pitch_rate,
                    tr.roll_rate,
                    tr.wheel_rate,
                    tr.base_x,
                    tr.base_y,
                    tr.u_rw,
                    tr.u_bx,
                    tr.u_by,
                ],
                dtype=float,
            ))):
                continue
            out[tr.step] = tr
    if not out:
        raise RuntimeError(f"No usable control-update rows found in trace: {path}")
    return out


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_trace)
    teacher_path = Path(args.teacher_trace)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_control_rows(baseline_path)
    teacher_rows = load_control_rows(teacher_path)

    common_steps = sorted(set(baseline_rows.keys()) & set(teacher_rows.keys()))
    if len(common_steps) < 32:
        raise RuntimeError("Too few aligned samples between baseline and teacher traces.")

    gate_tilt_rad = float(np.radians(max(args.residual_gate_tilt_deg, 0.0)))
    gate_rate = float(max(args.residual_gate_rate, 0.0))
    target_limit = np.array(
        [
            max(float(args.max_target_rw), 0.0),
            max(float(args.max_target_bx), 0.0),
            max(float(args.max_target_by), 0.0),
        ],
        dtype=float,
    )

    features = []
    targets = []
    rows_used = 0
    rows_skipped_gate = 0
    rows_skipped_dt = 0

    prev_base: TraceRow | None = None
    prev_u_nominal: np.ndarray | None = None

    for step in common_steps:
        b = baseline_rows[step]
        t = teacher_rows[step]

        if prev_base is None:
            prev_base = b
            prev_u_nominal = np.array([b.u_rw, b.u_bx, b.u_by], dtype=float)
            continue

        dt = float(b.sim_time_s - prev_base.sim_time_s)
        if not np.isfinite(dt) or dt <= 1e-9:
            rows_skipped_dt += 1
            prev_base = b
            prev_u_nominal = np.array([b.u_rw, b.u_bx, b.u_by], dtype=float)
            continue

        base_vx = (b.base_x - prev_base.base_x) / dt
        base_vy = (b.base_y - prev_base.base_y) / dt
        x_est = np.array(
            [
                b.pitch,
                b.roll,
                b.pitch_rate,
                b.roll_rate,
                b.wheel_rate,
                b.base_x,
                b.base_y,
                base_vx,
                base_vy,
            ],
            dtype=float,
        )
        u_nominal = np.array([b.u_rw, b.u_bx, b.u_by], dtype=float)
        u_eff_applied = prev_u_nominal.copy() if prev_u_nominal is not None else u_nominal.copy()

        if not np.all(np.isfinite(x_est)):
            prev_base = b
            prev_u_nominal = u_nominal
            continue

        tilt_mag = max(abs(float(b.pitch)), abs(float(b.roll)))
        rate_mag = max(abs(float(b.pitch_rate)), abs(float(b.roll_rate)))
        if (not args.allow_outside_gate) and (tilt_mag > gate_tilt_rad or rate_mag > gate_rate):
            rows_skipped_gate += 1
            prev_base = b
            prev_u_nominal = u_nominal
            continue

        teacher_u = np.array([t.u_rw, t.u_bx, t.u_by], dtype=float)
        delta_u = teacher_u - u_nominal
        delta_u = np.clip(delta_u, -target_limit, target_limit)

        feat = np.concatenate([x_est, u_eff_applied, u_nominal]).astype(float, copy=False)
        if feat.size != 15:
            prev_base = b
            prev_u_nominal = u_nominal
            continue

        features.append(feat)
        targets.append(delta_u.astype(float, copy=False))
        rows_used += 1

        prev_base = b
        prev_u_nominal = u_nominal

    if rows_used < 32:
        raise RuntimeError("Not enough rows after filtering to build a useful dataset.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)

    np.savez_compressed(
        out_path,
        features=x,
        targets=y,
        feature_names=np.asarray(
            [
                "pitch",
                "roll",
                "pitch_rate",
                "roll_rate",
                "wheel_rate",
                "base_x",
                "base_y",
                "base_vx",
                "base_vy",
                "u_eff_rw",
                "u_eff_bx",
                "u_eff_by",
                "u_nom_rw",
                "u_nom_bx",
                "u_nom_by",
            ],
            dtype=object,
        ),
        target_names=np.asarray(["delta_rw", "delta_bx", "delta_by"], dtype=object),
        baseline_trace=str(baseline_path),
        teacher_trace=str(teacher_path),
        rows_total_aligned=len(common_steps),
        rows_used=rows_used,
        rows_skipped_gate=rows_skipped_gate,
        rows_skipped_dt=rows_skipped_dt,
    )

    print(f"Wrote dataset: {out_path}")
    print(f"Aligned rows: {len(common_steps)}")
    print(f"Used rows: {rows_used}")
    print(f"Skipped (gate): {rows_skipped_gate}")
    print(f"Skipped (dt): {rows_skipped_dt}")
    print(f"Feature shape: {x.shape}")
    print(f"Target shape: {y.shape}")


if __name__ == "__main__":
    main()
