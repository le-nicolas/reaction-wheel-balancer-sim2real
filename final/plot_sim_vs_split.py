from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot current vs hardware_explicit_split benchmark comparison.")
    parser.add_argument("--csv", type=str, required=True, help="Benchmark CSV path.")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))

    keep = []
    for row in rows:
        family = str(row.get("controller_family", ""))
        if family in {"current", "hardware_explicit_split"}:
            keep.append(row)
    if not keep:
        raise RuntimeError("No current/hardware_explicit_split rows found in benchmark CSV.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required to generate the PNG.") from exc

    mode_order = []
    for row in keep:
        mode = str(row["mode_id"])
        if mode not in mode_order:
            mode_order.append(mode)

    family_order = ["current", "hardware_explicit_split"]
    metrics = [
        ("survival_rate", "Survival"),
        ("score_composite", "Score"),
        ("mean_control_energy", "Control Energy"),
        ("worst_pitch_deg", "Worst Pitch (deg)"),
        ("worst_roll_deg", "Worst Roll (deg)"),
    ]

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in keep:
        lookup[(str(row["mode_id"]), str(row["controller_family"]))] = row

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(mode_order), dtype=float)
    width = 0.32
    colors = {"current": "#2E86AB", "hardware_explicit_split": "#E07A5F"}

    for ax, (metric, title) in zip(axes, metrics):
        for idx, family in enumerate(family_order):
            offset = (-0.5 + idx) * width
            vals = []
            for mode in mode_order:
                row = lookup.get((mode, family))
                vals.append(float(row.get(metric, "nan")) if row is not None else float("nan"))
            ax.bar(x + offset, vals, width=width, label=family if ax is axes[0] else None, color=colors[family])
        ax.set_title(title)
        ax.set_xticks(x, [m.replace("mode_", "") for m in mode_order], rotation=20)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False)
    fig.suptitle("Sim Family vs Hardware Explicit Split (100-episode benchmark)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)


if __name__ == "__main__":
    main()
