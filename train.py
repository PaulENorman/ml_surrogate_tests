"""Surface-field + direct-Cd Transolver++ baseline and held-out reports."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

def load_case(path: Path, device: torch.device) -> dict:
    with np.load(path) as z:
        case = {key: torch.as_tensor(z[key], device=device, dtype=torch.float32)
                for key in z.files}
    length = case["l_ref"]
    case["position"] = (case["xyz"] - case["xyz"].mean(dim=0)) / length
    return case


def forward_case(model, case: dict) -> dict:
    # The fifth Transolver output is pooled into a directly supervised scalar.
    output = model(case["normals"][None], embedding=case["position"][None])[0]
    cp = output[:, 0]
    cf = output[:, 1:4] / 1000.0
    weights = case["area"] / case["area"].sum()
    direct = (output[:, 4] * weights).sum()
    factor = case["area"] / case["a_ref"]
    pressure = (-cp * case["normals"][:, 0] * factor).sum()
    friction = (-cf[:, 0] * factor).sum()
    return {"cp": cp, "cf": cf, "direct": direct, "pressure": pressure,
            "friction": friction, "integrated": pressure + friction}


def case_loss(pred: dict, case: dict, integrated_weight: float) -> torch.Tensor:
    import torch.nn.functional as F

    weights = case["area"] / case["area"].sum()
    cp_loss = torch.sum(weights * (pred["cp"] - case["cp"]) ** 2)
    cf_loss = torch.sum(weights[:, None] * ((pred["cf"] - case["cf"]) * 1000.0) ** 2) / 3
    direct_loss = F.smooth_l1_loss(pred["direct"], case["cd"], beta=0.02)
    integrated_loss = F.smooth_l1_loss(pred["integrated"], case["cd"], beta=0.02)
    return cp_loss + 0.25 * cf_loss + 2.0 * direct_loss + integrated_weight * integrated_loss


@torch.no_grad()
def evaluate(model, paths: list[Path], device: torch.device,
             baseline_cd: float | None = None) -> dict:
    model.eval()
    rows = []
    for path in paths:
        case = load_case(path, device)
        pred = forward_case(model, case)
        true_pressure = (-case["cp"] * case["normals"][:, 0]
                         * case["area"] / case["a_ref"]).sum()
        true_friction = (-case["cf"][:, 0] * case["area"] / case["a_ref"]).sum()
        weights = case["area"] / case["area"].sum()
        cp_error = pred["cp"] - case["cp"]
        cf_error = pred["cf"] - case["cf"]
        rows.append({"run": path.stem, "cd_true": case["cd"].item(),
                     "cd_direct": pred["direct"].item(),
                     "cd_integrated": pred["integrated"].item(),
                     "cd_pressure_pred": pred["pressure"].item(),
                     "cd_friction_pred": pred["friction"].item(),
                     "cd_pressure_cfd_sampled": true_pressure.item(),
                     "cd_friction_cfd_sampled": true_friction.item(),
                     "cp_mae": (weights * cp_error.abs()).sum().item(),
                     "cp_rmse": (weights * cp_error**2).sum().sqrt().item(),
                     "cf_mae": (weights[:, None] * cf_error.abs()).sum().div(3).item()})
    def score(key: str) -> dict:
        diffs = np.array([row[key] - row["cd_true"] for row in rows])
        truth = np.array([row["cd_true"] for row in rows])
        squared_total = float(np.sum((truth - truth.mean()) ** 2))
        rng = np.random.default_rng(42)
        boot = np.mean(np.abs(diffs[rng.integers(0, len(diffs), (500, len(diffs)))]), axis=1)
        return {"mae": float(np.mean(np.abs(diffs))),
                "rmse": float(np.sqrt(np.mean(diffs**2))),
                "bias": float(np.mean(diffs)),
                "median_abs": float(np.median(np.abs(diffs))),
                "max_abs": float(np.max(np.abs(diffs))),
                "r2": 1.0 - float(np.sum(diffs**2)) / squared_total if squared_total > 0 else None,
                "mae_bootstrap_95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
    result = {"n": len(rows), "direct_cd": score("cd_direct"),
            "integrated_cd": score("cd_integrated"),
            "mean_cp_mae": float(np.mean([r["cp_mae"] for r in rows])),
            "mean_cp_rmse": float(np.mean([r["cp_rmse"] for r in rows])),
            "mean_cf_mae": float(np.mean([r["cf_mae"] for r in rows])),
            "mean_pressure_cd_mae_vs_sampled_cfd": float(np.mean([
                abs(r["cd_pressure_pred"] - r["cd_pressure_cfd_sampled"]) for r in rows])),
            "mean_friction_cd_mae_vs_sampled_cfd": float(np.mean([
                abs(r["cd_friction_pred"] - r["cd_friction_cfd_sampled"]) for r in rows])),
            "mean_cd_disagreement": float(np.mean([abs(r["cd_direct"] - r["cd_integrated"])
                                               for r in rows])), "cases": rows}
    if baseline_cd is not None:
        result["train_mean_cd_baseline"] = baseline_cd
        result["train_mean_cd_mae"] = float(np.mean([
            abs(row["cd_true"] - baseline_cd) for row in rows]))
    return result


def train(prepared: Path, output: Path, split: str, max_per_split: int,
          epochs: int = 20, lr: float = 1e-4, closure_limit: float = 0.001,
          sampled_limit: float = 0.01, on_checkpoint=None,
          model_family: str = "physicsnemo++") -> dict:

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Modal training job")
    report = json.loads((prepared / split / "closure.json").read_text())
    if report["max_per_split"] != max_per_split:
        raise ValueError("Prepared sample limit differs from requested limit")
    closure = [abs(row["full_minus_source"]) for part in report["cases"].values()
               for row in part]
    sampling_error = [abs(row["sampled_minus_full"]) for part in report["cases"].values()
                      for row in part]
    if max(closure) > closure_limit:
        raise ValueError(f"Raw Cd closure failed: max |integrated-source|={max(closure):.5f}; "
                         "inspect closure.json before training")
    if max(sampling_error) > sampled_limit:
        raise ValueError(f"Sampled Cd quadrature failed: max |sampled-full|="
                         f"{max(sampling_error):.5f}; increase points_per_case")
    ids = {kind: [row["run"] for row in rows] for kind, rows in report["cases"].items()}
    paths = {kind: [prepared / split / f"{run}.npz" for run in runs]
             for kind, runs in ids.items()}
    if not all(paths.values()):
        raise ValueError("All train/val/test partitions need at least one case")
    baseline_cd = float(np.mean([row["cd_source"] for row in report["cases"]["train"]]))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    if model_family == "physicsnemo++":
        from physicsnemo.models.transolver import Transolver
        model = Transolver(functional_dim=3, embedding_dim=3, out_dim=5,
                           structured_shape=None, unified_pos=False,
                           n_layers=4, n_hidden=128, n_head=4, slice_num=32,
                           plus=True, use_te=False).to(device)
        model_name = "PhysicsNeMo Transolver++"
    elif model_family == "transolver3":
        from transolver3_model import SurfaceTransolver3
        model = SurfaceTransolver3().to(device)
        model_name = "Official Transolver-3 (matched capacity)"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        random.shuffle(paths["train"])
        losses = []
        for path in paths["train"]:
            case = load_case(path, device)
            optimizer.zero_grad(set_to_none=True)
            loss = case_loss(forward_case(model, case), case,
                             integrated_weight=0.0 if epoch < 5 else 2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val = evaluate(model, paths["val"], device)
        score = (val["direct_cd"]["mae"] + val["integrated_cd"]["mae"]
                 + 0.1 * val["mean_cp_mae"])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val": {key: value for key, value in val.items() if key != "cases"}})
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "split": split, "max_per_split": max_per_split,
                        "model_family": model_family}, output / "best.pt")
            if on_checkpoint is not None:
                on_checkpoint(epoch + 1)
        print(json.dumps({"epoch": epoch + 1, "train_loss": history[-1]["train_loss"],
                          "val_cd_direct_mae": val["direct_cd"]["mae"],
                          "val_cd_integrated_mae": val["integrated_cd"]["mae"],
                          "val_cp_mae": val["mean_cp_mae"]}), flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    result = {"model": model_name, "model_family": model_family, "split": split,
              "loss": "area-weighted Cp MSE + 0.25 Cf MSE (x1000) + 2 Huber(Cd_direct) + "
                      "2 Huber(Cd_integrated) after 5 warmup epochs",
              "max_per_split": max_per_split, "best_epoch": checkpoint["epoch"],
              "seconds": time.perf_counter() - start,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "closure_max_abs_cd": max(closure),
              "sampling_max_abs_cd": max(sampling_error), "history": history,
              "validation": evaluate(model, paths["val"], device, baseline_cd),
              "test": evaluate(model, paths["test"], device, baseline_cd)}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return {key: value for key, value in result.items() if key not in {"history", "test", "validation"}}
