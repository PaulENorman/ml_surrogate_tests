"""Native-cell force audit and deterministic, area-sampled training arrays."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from data import selected_ids
from forces import Q_INF, integrate_cd

PREP_VERSION = 3


def csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


def run_number(run: str) -> str:
    return run.removeprefix("run_")


def prepare_one(raw: Path, out: Path, run: str, points_per_case: int) -> dict:
    import pyvista as pv

    cached = out / f"{run}.json"
    if cached.is_file() and (out / f"{run}.npz").is_file():
        record = json.loads(cached.read_text())
        if record.get("preprocessing_version") == PREP_VERSION and record.get("requested_points") == points_per_case:
            return record
    # Official scarce/medium/full splits share validation and test runs.
    # Copy an already prepared case rather than re-reading its ~600 MB VTP.
    for prior in ("super_scarce", "scarce", "medium"):
        alternate = out.parent / prior / f"{run}.json"
        source_array = out.parent / prior / f"{run}.npz"
        if alternate == cached or not (alternate.is_file() and source_array.is_file()):
            continue
        record = json.loads(alternate.read_text())
        if record.get("preprocessing_version") == PREP_VERSION and record.get("requested_points") == points_per_case:
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_array, out / f"{run}.npz")
            shutil.copy2(alternate, cached)
            return record
    n = run_number(run)
    directory = raw / run
    mesh = pv.read(directory / f"boundary_{n}.vtp")
    area = np.load(directory / f"boundary_cell_area_{n}.npy").astype(np.float64)
    if area.shape != (mesh.n_cells,):
        raise ValueError(f"{run}: native area count does not match VTP cells")
    if not {"CpMeanTrim", "wallShearStressMeanTrim"} <= set(mesh.cell_data):
        raise KeyError(f"{run}: missing Cp or wall-shear cell field")
    cp = np.asarray(mesh.cell_data["CpMeanTrim"], dtype=np.float64)
    cf = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64) / Q_INF
    if cp.shape != (mesh.n_cells,) or cf.shape != (mesh.n_cells, 3):
        raise ValueError(f"{run}: unexpected surface-field shape")
    normals_mesh = mesh.compute_normals(cell_normals=True, point_normals=False,
                                        consistent_normals=True, auto_orient_normals=False,
                                        split_vertices=False, inplace=False)
    if normals_mesh.n_cells != mesh.n_cells:
        raise ValueError(f"{run}: normal computation changed cell order")
    normals = np.asarray(normals_mesh.cell_data["Normals"], dtype=np.float64)
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    centroid = np.asarray(mesh.points, dtype=np.float64).mean(axis=0)
    orientation = float(np.sum(np.einsum("ij,ij->i", centers - centroid, normals) * area))
    if abs(orientation) < 1e-8:
        raise ValueError(f"{run}: unable to orient surface normals")
    normals *= np.sign(orientation)
    ref = csv_row(directory / f"geo_ref_{n}.csv")
    a_ref, l_ref = float(ref["aRef"]), float(ref["lRef"])
    raw_cd = float(csv_row(directory / f"force_mom_{n}.csv")["Cd"])
    full = integrate_cd(cp, cf[:, 0], normals[:, 0], area, a_ref)
    if not np.isfinite([*full.values(), raw_cd, l_ref]).all():
        raise ValueError(f"{run}: non-finite data")

    # Stratify by streamwise position and normal direction. Cd is a small
    # difference of much larger fore/aft pressure forces; global random or
    # systematic sampling gives noisy integrals even at 65k points.
    seed = int.from_bytes(hashlib.sha256(run.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    x_edges = np.linspace(centers[:, 0].min(), centers[:, 0].max(), 13)[1:-1]
    x_bin = np.digitize(centers[:, 0], x_edges)
    normal_bin = np.digitize(normals[:, 0], [-0.5, -0.1, 0.1, 0.5])
    group = x_bin * 5 + normal_bin
    group_area = np.bincount(group, weights=area, minlength=60)
    sensitivity = 0.15 + np.bincount(group, weights=area * abs(normals[:, 0]),
                                    minlength=60) / np.maximum(group_area, 1e-30)
    budget = group_area * sensitivity
    strata = [(g, members, np.cumsum(area[members]))
              for g in np.flatnonzero(group_area)
              if (members := np.flatnonzero(group == g)).size]
    target_count = points_per_case
    while True:
        indices, measures = [], []
        for g, members, cumulative_area in strata:
            count = max(16, int(round(target_count * budget[g] / budget.sum())))
            targets = (np.arange(count) + rng.random()) * (group_area[g] / count)
            chosen = members[np.searchsorted(cumulative_area, targets, side="left")]
            indices.append(chosen)
            measures.append(np.full(count, group_area[g] / count, dtype=np.float32))
        ix = np.concatenate(indices)
        sampled_area = np.concatenate(measures)
        sample = integrate_cd(cp[ix], cf[ix, 0], normals[ix, 0], sampled_area, a_ref)
        if abs(sample["total"] - full["total"]) <= 0.01 or target_count >= 131072:
            break
        target_count *= 2
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{run}.npz", xyz=centers[ix].astype(np.float32),
                        normals=normals[ix].astype(np.float32), cp=cp[ix].astype(np.float32),
                        cf=cf[ix].astype(np.float32), area=sampled_area,
                        a_ref=np.float32(a_ref), l_ref=np.float32(l_ref),
                        cd=np.float32(raw_cd))
    record = {"run": run, "preprocessing_version": PREP_VERSION,
            "requested_points": points_per_case,
            "cells": mesh.n_cells, "native_area_m2": float(area.sum()),
            "sampled_cells": int(len(ix)),
            "cd_source": raw_cd, "cd_full": full, "cd_sampled": sample,
            "full_minus_source": full["total"] - raw_cd,
            "sampled_minus_full": sample["total"] - full["total"]}
    cached.write_text(json.dumps(record, indent=2))
    return record


def prepare_all(raw: Path, out: Path, split: str, max_per_split: int,
                points_per_case: int = 32768, progress=None) -> dict:
    manifest = json.loads((raw / "splits/manifest.json").read_text())
    ids = selected_ids(manifest, split, max_per_split)
    report = {"split": split, "max_per_split": max_per_split,
              "points_per_case": points_per_case, "cases": {}}
    completed = 0
    total = sum(map(len, ids.values()))
    by_run = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(prepare_one, raw, out / split, run, points_per_case): run
                   for runs in ids.values() for run in runs}
        for future in as_completed(futures):
            by_run[futures[future]] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, total, futures[future])
    for kind, runs in ids.items():
        report["cases"][kind] = [by_run[run] for run in runs]
    report_path = out / split / "closure.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def prepare_available(raw: Path, out: Path, split: str,
                      points_per_case: int = 32768, progress=None) -> int:
    """Preprocess already committed runs while ingestion continues."""
    manifest = json.loads((raw / "splits/manifest.json").read_text())
    ids = selected_ids(manifest, split, 0)
    ready = []
    for run in dict.fromkeys(run for runs in ids.values() for run in runs):
        n = run_number(run)
        if all((raw / run / name).is_file() for name in
               (f"boundary_{n}.vtp", f"boundary_cell_area_{n}.npy",
                f"geo_ref_{n}.csv", f"force_mom_{n}.csv")):
            ready.append(run)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(prepare_one, raw, out / split, run, points_per_case): run
                   for run in ready}
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if progress is not None:
                progress(done, len(ready), futures[future])
    return len(ready)
