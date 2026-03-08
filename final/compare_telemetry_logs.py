from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MIN_SAMPLES = 16
DEFAULT_GRID_SAMPLES = 400


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    sim_keys: tuple[str, ...]
    real_keys: tuple[str, ...]


CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("pitch_deg", ("pitch_deg", "pitch_rad", "pitch"), ("pitch_deg", "pitch_rad")),
    ChannelSpec("roll_deg", ("roll_deg", "roll_rad", "roll"), ("roll_deg", "roll_rad")),
    ChannelSpec(
        "pitch_rate_dps",
        ("pitch_rate_dps", "pitch_rate_rad_s", "pitch_rate"),
        ("pitch_rate_dps", "pitch_rate_rad_s"),
    ),
    ChannelSpec(
        "roll_rate_dps",
        ("roll_rate_dps", "roll_rate_rad_s", "roll_rate"),
        ("roll_rate_dps", "roll_rate_rad_s"),
    ),
    ChannelSpec(
        "wheel_rpm",
        ("wheel_rate_rpm", "wheel_rate_rad_s", "wheel_rate", "reaction_speed"),
        ("reaction_speed_rpm", "wheel_rate_rpm", "reaction_speed", "wheel_rate_rad_s"),
    ),
    ChannelSpec("u_rw", ("u_rw_cmd", "u_rw", "rw_cmd_norm", "rt"), ("rw_cmd_norm", "u_rw", "rt")),
    ChannelSpec("u_drive", ("u_bx_cmd", "u_bx", "drive_cmd_norm", "dt"), ("drive_cmd_norm", "u_drive", "dt")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MuJoCo telemetry/trace logs against hardware/HIL logs.")
    parser.add_argument("--sim-csv", required=True, help="Simulation trace or telemetry CSV.")
    parser.add_argument("--real-csv", required=True, help="Hardware/HIL CSV log.")
    parser.add_argument(
        "--align",
        choices=("time", "index"),
        default="time",
        help="Alignment method. 'time' interpolates on overlapping time support when possible.",
    )
    parser.add_argument(
        "--grid-samples",
        type=int,
        default=DEFAULT_GRID_SAMPLES,
        help="Interpolation grid size used by --align time.",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _first_finite(row: dict[str, str], keys: Iterable[str]) -> float:
    for key in keys:
        raw = row.get(key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _first_finite_key_value(row: dict[str, str], keys: Iterable[str]) -> tuple[str | None, float]:
    for key in keys:
        raw = row.get(key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return key, value
    return None, float("nan")


def _extract_time(rows: list[dict[str, str]]) -> np.ndarray:
    vals = []
    for row in rows:
        vals.append(
            _first_finite(
                row,
                ("sim_time_s", "time_s", "wall_time_s", "t_s", "time"),
            )
        )
    arr = np.asarray(vals, dtype=float)
    if np.count_nonzero(np.isfinite(arr)) < MIN_SAMPLES:
        return np.full(len(rows), np.nan, dtype=float)
    return arr


def _to_canonical_value(channel_name: str, key: str | None, value: float) -> float:
    if key is None or not np.isfinite(value):
        return float("nan")
    if channel_name in {"pitch_deg", "roll_deg"}:
        return value if key.endswith("_deg") else (value * 180.0 / np.pi)
    if channel_name in {"pitch_rate_dps", "roll_rate_dps"}:
        return value if key.endswith("_dps") else (value * 180.0 / np.pi)
    if channel_name == "wheel_rpm":
        if key.endswith("_rpm"):
            return value
        if key == "reaction_speed":
            return value / 6.0
        return value * 60.0 / (2.0 * np.pi)
    return value


def _extract_series(rows: list[dict[str, str]], channel_name: str, keys: tuple[str, ...]) -> np.ndarray:
    vals = []
    for row in rows:
        key, value = _first_finite_key_value(row, keys)
        vals.append(_to_canonical_value(channel_name, key, value))
    return np.asarray(vals, dtype=float)


def _clean_time_series(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(finite) < MIN_SAMPLES:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    t = t[finite]
    y = y[finite]
    order = np.argsort(t, kind="stable")
    t = t[order]
    y = y[order]
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    return t[keep], y[keep]


def _align_by_time(
    sim_t: np.ndarray,
    sim_y: np.ndarray,
    real_t: np.ndarray,
    real_y: np.ndarray,
    grid_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    sim_t, sim_y = _clean_time_series(sim_t, sim_y)
    real_t, real_y = _clean_time_series(real_t, real_y)
    if sim_t.size < MIN_SAMPLES or real_t.size < MIN_SAMPLES:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    start = max(float(sim_t[0]), float(real_t[0]))
    end = min(float(sim_t[-1]), float(real_t[-1]))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    grid_n = max(int(grid_samples), MIN_SAMPLES)
    grid = np.linspace(start, end, num=grid_n, dtype=float)
    return np.interp(grid, sim_t, sim_y), np.interp(grid, real_t, real_y)


def _align_by_index(sim_y: np.ndarray, real_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim_f = sim_y[np.isfinite(sim_y)]
    real_f = real_y[np.isfinite(real_y)]
    n = min(sim_f.size, real_f.size)
    if n < MIN_SAMPLES:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return sim_f[:n], real_f[:n]


def _metrics(sim_y: np.ndarray, real_y: np.ndarray) -> dict[str, float]:
    if sim_y.size < MIN_SAMPLES or real_y.size < MIN_SAMPLES:
        return {"samples": float(min(sim_y.size, real_y.size)), "rmse": np.nan, "nrmse": np.nan, "corr": np.nan, "consistency": np.nan}
    diff = sim_y - real_y
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    var = float(np.var(real_y))
    nrmse = float(np.sqrt(mse / max(var, 1e-9)))
    if np.std(sim_y) < 1e-12 or np.std(real_y) < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(sim_y, real_y)[0, 1])
    consistency = float(np.clip(1.0 - nrmse, 0.0, 1.0))
    return {
        "samples": float(sim_y.size),
        "rmse": rmse,
        "nrmse": nrmse,
        "corr": corr,
        "consistency": consistency,
    }


def compare_logs(args: argparse.Namespace) -> dict[str, object]:
    sim_path = Path(args.sim_csv)
    real_path = Path(args.real_csv)
    sim_rows = _load_csv(sim_path)
    real_rows = _load_csv(real_path)
    sim_t = _extract_time(sim_rows)
    real_t = _extract_time(real_rows)

    channel_results: dict[str, dict[str, float]] = {}
    for spec in CHANNEL_SPECS:
        sim_y = _extract_series(sim_rows, spec.name, spec.sim_keys)
        real_y = _extract_series(real_rows, spec.name, spec.real_keys)
        if args.align == "time":
            sim_aligned, real_aligned = _align_by_time(sim_t, sim_y, real_t, real_y, int(args.grid_samples))
            if sim_aligned.size < MIN_SAMPLES:
                sim_aligned, real_aligned = _align_by_index(sim_y, real_y)
        else:
            sim_aligned, real_aligned = _align_by_index(sim_y, real_y)
        channel_results[spec.name] = _metrics(sim_aligned, real_aligned)

    consistencies = np.asarray(
        [float(v["consistency"]) for v in channel_results.values() if np.isfinite(float(v["consistency"]))],
        dtype=float,
    )
    nrmses = np.asarray(
        [float(v["nrmse"]) for v in channel_results.values() if np.isfinite(float(v["nrmse"]))],
        dtype=float,
    )
    corrs = np.asarray(
        [float(v["corr"]) for v in channel_results.values() if np.isfinite(float(v["corr"]))],
        dtype=float,
    )

    return {
        "sim_csv": str(sim_path),
        "real_csv": str(real_path),
        "align": args.align,
        "channels": channel_results,
        "overall_consistency_mean": float(np.mean(consistencies)) if consistencies.size else np.nan,
        "overall_nrmse_mean": float(np.mean(nrmses)) if nrmses.size else np.nan,
        "overall_corr_mean": float(np.mean(corrs)) if corrs.size else np.nan,
    }


def main() -> None:
    args = parse_args()
    result = compare_logs(args)
    print(f"sim_csv={result['sim_csv']}")
    print(f"real_csv={result['real_csv']}")
    print(f"align={result['align']}")
    for name, metrics in result["channels"].items():
        print(
            f"{name}: samples={int(metrics['samples'])} "
            f"consistency={metrics['consistency']:.4f} "
            f"nrmse={metrics['nrmse']:.4f} "
            f"corr={metrics['corr']:.4f}"
        )
    print(
        "overall: "
        f"consistency_mean={result['overall_consistency_mean']:.4f} "
        f"nrmse_mean={result['overall_nrmse_mean']:.4f} "
        f"corr_mean={result['overall_corr_mean']:.4f}"
    )
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote_json={out_path}")


if __name__ == "__main__":
    main()
