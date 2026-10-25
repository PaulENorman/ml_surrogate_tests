"""Plot held-out Cd and sampled surface fields from saved experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180})


def save(fig, output: Path, name: str):
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def cd_plots(reports: list[dict], output: Path):
    colors = ["#2575ad", "#d57b24", "#23946a"]
    values = [v for d in reports for r in d["test"]["cases"]
              for v in (r["cd_true"], r["cd_direct"], r["cd_integrated"])]
    lo, hi = min(values), max(values)
    padding = (hi - lo) * 0.08
    limits = (lo - padding, hi + padding)
    fig, axes = plt.subplots(2, len(reports), figsize=(3.45 * len(reports), 6.8),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for col, report in enumerate(reports):
        n = report["training_vehicles"]
        rows = report["test"]["cases"]
        for row, key in enumerate(("direct", "integrated")):
            ax = axes[row, col]
            metrics = report["test"][f"{key}_cd"]
            ax.plot(limits, limits, color="0.55", linestyle="--", linewidth=1, zorder=0)
            ax.scatter([r["cd_true"] for r in rows], [r[f"cd_{key}"] for r in rows],
                       s=25, color=colors[col % len(colors)], alpha=0.82,
                       edgecolor="white", linewidth=0.35)
            ax.set(xlim=limits, ylim=limits, aspect="equal")
            ax.grid(alpha=0.17)
            ax.text(0.04, 0.96, f"MAE {metrics['mae']:.4f}\nR² {metrics['r2']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=10)
            if row == 0:
                ax.set_title(f"{n} training vehicles", weight="bold")
            if col == 0:
                ax.set_ylabel(f"{'Direct' if key == 'direct' else 'Integrated'} predicted Cd")
            if row == 1:
                ax.set_xlabel("CFD Cd")
    fig.suptitle("DrivAerML · Transolver-3 · same 50 test vehicles · sampled surface", fontsize=13)
    save(fig, output, "cd_scatter")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout="constrained")
    for key, label, color in (("direct", "Direct Cd", "#2575ad"),
                              ("integrated", "Integrated Cd", "#23946a")):
        n = np.array([d["training_vehicles"] for d in reports])
        metrics = [d["test"][f"{key}_cd"] for d in reports]
        mae = np.array([m["mae"] for m in metrics])
        ci = np.array([m["mae_bootstrap_95"] for m in metrics])
        axes[0].errorbar(n, mae, yerr=np.array([mae - ci[:, 0], ci[:, 1] - mae]),
                         marker="o", color=color, capsize=4, label=label)
        axes[1].plot(n, [m["r2"] for m in metrics], marker="o", color=color, label=label)
    axes[0].plot([d["training_vehicles"] for d in reports],
                 [d["test"]["train_mean_cd_mae"] for d in reports],
                 marker=".", linestyle=":", color="0.45", label="Training-mean Cd")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([d["training_vehicles"] for d in reports],
                      [str(d["training_vehicles"]) for d in reports])
        ax.set_xlabel("Training vehicles")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Test Cd MAE (95% bootstrap interval)")
    axes[1].set_ylabel("Test R²")
    axes[1].axhline(0, color="0.5", linestyle="--", linewidth=1)
    fig.suptitle("DrivAerML · effect of training-set size · Transolver-3", fontsize=13)
    save(fig, output, "data_efficiency")


def surface_plots(report: dict, field_dir: Path, output: Path):
    ordered = sorted(report["test"]["cases"],
                     key=lambda r: abs(r["cd_integrated"] - r["cd_true"]))
    selected = [("Lowest Cd error", ordered[0]),
                ("Median Cd error", ordered[len(ordered) // 2]),
                ("Highest Cd error", ordered[-1])]
    loaded = []
    for rank, row in selected:
        with np.load(field_dir / f"{row['run']}.npz") as f:
            loaded.append((rank, row, {k: f[k] for k in f.files}))
    metadata = {"selection": "lowest, median and highest absolute integrated Cd error on test set",
                "representation": "sampled native cell centers; not a full-mesh field prediction",
                "vehicles": [{"run": row["run"], "rank": rank,
                              "cd_abs_error": abs(row["cd_integrated"] - row["cd_true"])}
                             for rank, row, _ in loaded], "color_limits": {}}
    for field, title, elev, azim, cmap in (("cp", "Pressure coefficient Cp", 22, -120, "RdBu_r"),
                                          ("cf_x", "Streamwise shear coefficient Cf_x", -22, -120, "viridis")):
        def field_values(data, suffix):
            return data[f"cp_{suffix}"] if field == "cp" else data[f"cf_{suffix}"][:, 0]
        all_values = np.concatenate([field_values(data, suffix)
                                     for _, _, data in loaded for suffix in ("true", "pred")])
        lo, hi = (float(value) for value in np.quantile(all_values, [0.005, 0.995]))
        field_norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi) if field == "cp" and lo < 0 < hi else Normalize(lo, hi)
        errors = np.concatenate([field_values(d, "pred") - field_values(d, "true")
                                 for _, _, d in loaded])
        max_err = max(float(np.quantile(np.abs(errors), 0.99)), 1e-12)
        metadata["color_limits"][field] = {"field": [lo, hi], "error": [-max_err, max_err],
                                            "field_quantiles": [0.005, 0.995],
                                            "full_field_range": [float(all_values.min()), float(all_values.max())],
                                            "absolute_error_quantile": 0.99,
                                            "full_max_absolute_error": float(np.abs(errors).max()),
                                            "saturated_fraction": float(np.mean(np.abs(errors) > max_err))}
        azimuth, elevation = np.deg2rad([azim, elev])
        camera = np.array([np.cos(elevation) * np.cos(azimuth),
                           np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
        up = np.cross(camera, right)
        projected = []
        for _, _, data in loaded:
            xyz = data["xyz"]
            centered = xyz - (xyz.min(axis=0) + xyz.max(axis=0)) / 2
            projected.append((centered @ right, centered @ up,
                              np.argsort(centered @ camera)))
        all_u = np.concatenate([p[0] for p in projected])
        all_v = np.concatenate([p[1] for p in projected])
        margin = 0.06
        u_limits = (float(all_u.min() - margin), float(all_u.max() + margin))
        v_limits = (float(all_v.min() - margin), float(all_v.max() + margin))
        fig, axes = plt.subplots(3, 3, figsize=(12.0, 6.8), layout="constrained")
        for i, (rank, row, data) in enumerate(loaded):
            u, v, order = projected[i]
            truth, pred = field_values(data, "true"), field_values(data, "pred")
            for j, values in enumerate((truth, pred, pred - truth)):
                ax = axes[i, j]
                ax.scatter(u[order], v[order], c=values[order], s=1.1, linewidths=0,
                           cmap="coolwarm" if j == 2 else cmap,
                           norm=Normalize(-max_err, max_err) if j == 2 else field_norm,
                           rasterized=True)
                ax.set(xlim=u_limits, ylim=v_limits, aspect="equal")
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(("CFD", "Prediction", "Prediction − CFD")[j], weight="bold")
                if j == 0:
                    error = abs(row["cd_integrated"] - row["cd_true"])
                    ax.text(0.01, -0.11, f"{row['run']} · {rank}\n|Cd error| = {error:.4f}",
                            transform=ax.transAxes, fontsize=9)
        for j in range(3):
            norm = Normalize(-max_err, max_err) if j == 2 else field_norm
            sm = ScalarMappable(norm=norm, cmap="coolwarm" if j == 2 else cmap)
            fig.colorbar(sm, ax=list(axes[:, j]), shrink=0.75, fraction=0.025, pad=0.005,
                         extend="both")
        fig.suptitle(f"DrivAerML · {title} · {report['training_vehicles']} training vehicles\n"
                     "Area-weighted sampled surface; equal color scales for truth and prediction\n"
                     "Color limits cover 99% of selected sample points; extremes saturated", fontsize=11)
        save(fig, output, f"surface_{field}")
    (output / "surface_selection.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--training-vehicles", nargs="+", type=int, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if len(args.reports) != len(args.training_vehicles):
        parser.error("Specify one training-vehicle count per report")
    reports = []
    for path, n in zip(args.reports, args.training_vehicles):
        report = json.loads(path.read_text())
        report["training_vehicles"] = n
        reports.append(report)
    reports.sort(key=lambda d: d["training_vehicles"])
    expected_ids = {r["run"] for r in reports[0]["test"]["cases"]}
    if any({r["run"] for r in d["test"]["cases"]} != expected_ids
           or len(d["test"]["cases"]) != len(expected_ids) for d in reports):
        raise ValueError("Reports must use the same test vehicles, without duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    cd_plots(reports, args.output)
    rows = reports[-1]["test"]["cases"]
    with (args.output / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.fields:
        surface_plots(reports[-1], args.fields, args.output)


if __name__ == "__main__":
    main()
