import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Metrics:
    survival_rate: float
    crash_rate: float
    score_composite: float
    worst_tilt_deg: float
    wheel_over_hard_mean: float
    sim_real_consistency_mean: float
    sim_real_traj_nrmse_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sim-to-real sensitivity from benchmark baseline rows.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Benchmark CSV path. If omitted, newest final/results/benchmark_*.csv is used.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="final/results",
        help="Search directory used when --csv is omitted.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output prefix path (without suffix). Default: final/results/sim2real_sensitivity_<timestamp>.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Top factors to include in markdown summary.",
    )
    parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Include rows where both model variant and domain profile differ from nominal/default.",
    )
    return parser.parse_args()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except Exception:
        return float(default)


def _metric_from_row(row: dict[str, str]) -> Metrics:
    worst_pitch = abs(_as_float(row, "worst_pitch_deg"))
    worst_roll = abs(_as_float(row, "worst_roll_deg"))
    return Metrics(
        survival_rate=_as_float(row, "survival_rate"),
        crash_rate=_as_float(row, "crash_rate"),
        score_composite=_as_float(row, "score_composite"),
        worst_tilt_deg=max(worst_pitch, worst_roll),
        wheel_over_hard_mean=_as_float(row, "wheel_over_hard_mean"),
        sim_real_consistency_mean=_as_float(row, "sim_real_consistency_mean"),
        sim_real_traj_nrmse_mean=_as_float(row, "sim_real_traj_nrmse_mean"),
    )


def find_latest_benchmark_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("benchmark_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No benchmark CSV found under: {results_dir}")
    return candidates[0]


def load_baseline_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if _as_bool(row.get("is_baseline", False))]
    if not rows:
        raise RuntimeError("No baseline rows found in CSV. Ensure benchmark output includes run_id=baseline rows.")
    return rows


def _delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(a - b)


def _sensitivity_index(
    delta_crash_rate: float,
    delta_score: float,
    delta_worst_tilt: float,
    delta_wheel_over_hard: float,
    delta_sim_real_consistency: float,
) -> float:
    terms = []
    if np.isfinite(delta_crash_rate):
        terms.append(100.0 * abs(delta_crash_rate))
    if np.isfinite(delta_score):
        terms.append(abs(delta_score))
    if np.isfinite(delta_worst_tilt):
        terms.append(0.75 * abs(delta_worst_tilt))
    if np.isfinite(delta_wheel_over_hard):
        terms.append(80.0 * abs(delta_wheel_over_hard))
    if np.isfinite(delta_sim_real_consistency):
        terms.append(60.0 * abs(delta_sim_real_consistency))
    return float(np.sum(terms)) if terms else np.nan


def build_detail_rows(
    baseline_rows: Iterable[dict[str, str]],
    include_interactions: bool,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    anchor_by_key: dict[tuple[str, str], Metrics] = {}
    for row in baseline_rows:
        mode_id = str(row.get("mode_id", ""))
        family = str(row.get("controller_family", ""))
        variant = str(row.get("model_variant_id", ""))
        domain = str(row.get("domain_profile_id", ""))
        if variant == "nominal" and domain == "default":
            anchor_by_key[(mode_id, family)] = _metric_from_row(row)

    missing_anchor_keys = sorted(
        {
            (str(row.get("mode_id", "")), str(row.get("controller_family", "")))
            for row in baseline_rows
            if (str(row.get("mode_id", "")), str(row.get("controller_family", ""))) not in anchor_by_key
        }
    )

    details: list[dict[str, object]] = []
    for row in baseline_rows:
        mode_id = str(row.get("mode_id", ""))
        family = str(row.get("controller_family", ""))
        variant = str(row.get("model_variant_id", ""))
        domain = str(row.get("domain_profile_id", ""))
        anchor = anchor_by_key.get((mode_id, family))
        if anchor is None:
            continue

        if variant == "nominal" and domain == "default":
            continue
        if variant != "nominal" and domain == "default":
            factor_type = "model_variant"
            factor_name = variant
        elif variant == "nominal" and domain != "default":
            factor_type = "domain_profile"
            factor_name = domain
        elif include_interactions:
            factor_type = "interaction"
            factor_name = f"{variant}__{domain}"
        else:
            continue

        observed = _metric_from_row(row)
        d_survival = _delta(observed.survival_rate, anchor.survival_rate)
        d_crash = _delta(observed.crash_rate, anchor.crash_rate)
        d_score = _delta(observed.score_composite, anchor.score_composite)
        d_tilt = _delta(observed.worst_tilt_deg, anchor.worst_tilt_deg)
        d_wheel_hard = _delta(observed.wheel_over_hard_mean, anchor.wheel_over_hard_mean)
        d_consistency = _delta(observed.sim_real_consistency_mean, anchor.sim_real_consistency_mean)
        d_nrmse = _delta(observed.sim_real_traj_nrmse_mean, anchor.sim_real_traj_nrmse_mean)

        details.append(
            {
                "mode_id": mode_id,
                "controller_family": family,
                "factor_type": factor_type,
                "factor_name": factor_name,
                "model_variant_id": variant,
                "domain_profile_id": domain,
                "anchor_survival_rate": anchor.survival_rate,
                "anchor_crash_rate": anchor.crash_rate,
                "anchor_score_composite": anchor.score_composite,
                "anchor_worst_tilt_deg": anchor.worst_tilt_deg,
                "anchor_wheel_over_hard_mean": anchor.wheel_over_hard_mean,
                "anchor_sim_real_consistency_mean": anchor.sim_real_consistency_mean,
                "anchor_sim_real_traj_nrmse_mean": anchor.sim_real_traj_nrmse_mean,
                "survival_rate": observed.survival_rate,
                "crash_rate": observed.crash_rate,
                "score_composite": observed.score_composite,
                "worst_tilt_deg": observed.worst_tilt_deg,
                "wheel_over_hard_mean": observed.wheel_over_hard_mean,
                "sim_real_consistency_mean": observed.sim_real_consistency_mean,
                "sim_real_traj_nrmse_mean": observed.sim_real_traj_nrmse_mean,
                "delta_survival_rate": d_survival,
                "delta_crash_rate": d_crash,
                "delta_score_composite": d_score,
                "delta_worst_tilt_deg": d_tilt,
                "delta_wheel_over_hard_mean": d_wheel_hard,
                "delta_sim_real_consistency_mean": d_consistency,
                "delta_sim_real_traj_nrmse_mean": d_nrmse,
                "sensitivity_index": _sensitivity_index(
                    delta_crash_rate=d_crash,
                    delta_score=d_score,
                    delta_worst_tilt=d_tilt,
                    delta_wheel_over_hard=d_wheel_hard,
                    delta_sim_real_consistency=d_consistency,
                ),
            }
        )

    return details, missing_anchor_keys


def summarize_detail_rows(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        buckets[(str(row["factor_type"]), str(row["factor_name"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (factor_type, factor_name), rows in buckets.items():
        def vals(key: str) -> np.ndarray:
            arr = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
            return arr[np.isfinite(arr)]

        abs_survival = np.abs(vals("delta_survival_rate"))
        abs_crash = np.abs(vals("delta_crash_rate"))
        abs_score = np.abs(vals("delta_score_composite"))
        abs_tilt = np.abs(vals("delta_worst_tilt_deg"))
        abs_wheel = np.abs(vals("delta_wheel_over_hard_mean"))
        abs_cons = np.abs(vals("delta_sim_real_consistency_mean"))
        abs_nrmse = np.abs(vals("delta_sim_real_traj_nrmse_mean"))
        idx = vals("sensitivity_index")

        summary_rows.append(
            {
                "factor_type": factor_type,
                "factor_name": factor_name,
                "samples": len(rows),
                "mean_abs_delta_survival_rate": float(np.mean(abs_survival)) if abs_survival.size else np.nan,
                "mean_abs_delta_crash_rate": float(np.mean(abs_crash)) if abs_crash.size else np.nan,
                "mean_abs_delta_score_composite": float(np.mean(abs_score)) if abs_score.size else np.nan,
                "mean_abs_delta_worst_tilt_deg": float(np.mean(abs_tilt)) if abs_tilt.size else np.nan,
                "mean_abs_delta_wheel_over_hard_mean": float(np.mean(abs_wheel)) if abs_wheel.size else np.nan,
                "mean_abs_delta_sim_real_consistency_mean": float(np.mean(abs_cons)) if abs_cons.size else np.nan,
                "mean_abs_delta_sim_real_traj_nrmse_mean": float(np.mean(abs_nrmse)) if abs_nrmse.size else np.nan,
                "mean_sensitivity_index": float(np.mean(idx)) if idx.size else np.nan,
                "max_sensitivity_index": float(np.max(idx)) if idx.size else np.nan,
            }
        )

    summary_rows.sort(
        key=lambda r: (
            float(r.get("mean_sensitivity_index", np.nan))
            if np.isfinite(float(r.get("mean_sensitivity_index", np.nan)))
            else -np.inf
        ),
        reverse=True,
    )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _write_markdown_summary(
    path: Path,
    *,
    source_csv: Path,
    summary_rows: list[dict[str, object]],
    missing_anchor_keys: list[tuple[str, str]],
    top_n: int,
) -> None:
    lines = []
    lines.append("# Sim-to-Real Sensitivity Summary")
    lines.append("")
    lines.append(f"- Source benchmark CSV: `{source_csv.as_posix()}`")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Factors ranked: `{len(summary_rows)}`")
    lines.append("")
    if missing_anchor_keys:
        lines.append("## Missing Anchors")
        lines.append("")
        lines.append("Missing nominal/default baseline for:")
        for mode_id, family in missing_anchor_keys:
            lines.append(f"- `{mode_id} / {family}`")
        lines.append("")

    lines.append("## Top Factors")
    lines.append("")
    lines.append("| Rank | Factor | Type | Samples | Mean Index | Mean |dCrash| | Mean |dScore| | Mean |dTilt| |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(summary_rows[: max(1, top_n)], start=1):
        lines.append(
            f"| {i} | `{row['factor_name']}` | `{row['factor_type']}` | {row['samples']} | "
            f"{_fmt(row['mean_sensitivity_index'])} | {_fmt(row['mean_abs_delta_crash_rate'])} | "
            f"{_fmt(row['mean_abs_delta_score_composite'])} | {_fmt(row['mean_abs_delta_worst_tilt_deg'])} |"
        )
    lines.append("")
    lines.append("Higher index indicates stronger sensitivity relative to nominal/default baseline behavior.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_csv = Path(args.csv) if args.csv else find_latest_benchmark_csv(Path(args.results_dir))
    baseline_rows = load_baseline_rows(source_csv)

    detail_rows, missing_anchor_keys = build_detail_rows(
        baseline_rows=baseline_rows,
        include_interactions=bool(args.include_interactions),
    )
    if not detail_rows:
        raise RuntimeError("No sensitivity rows could be produced. Check model/domain variants in the benchmark CSV.")

    summary_rows = summarize_detail_rows(detail_rows)

    out_prefix = Path(args.out_prefix) if args.out_prefix else Path("final/results") / (
        f"sim2real_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    detail_path = Path(str(out_prefix) + "_detail.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    markdown_path = Path(str(out_prefix) + "_summary.md")

    _write_csv(detail_path, detail_rows)
    _write_csv(summary_path, summary_rows)
    _write_markdown_summary(
        markdown_path,
        source_csv=source_csv,
        summary_rows=summary_rows,
        missing_anchor_keys=missing_anchor_keys,
        top_n=max(1, int(args.top_n)),
    )

    print(f"Source CSV: {source_csv}")
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary MD: {markdown_path}")


if __name__ == "__main__":
    main()
