"""Selective DrivAerML download; never fetches volume fields."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from urllib.request import urlretrieve

BASE = "https://huggingface.co/datasets/neashton/drivaerml/resolve/main"
KINDS = ("train", "val", "test")


def selected_ids(manifest: dict, split: str, max_per_split: int = 0) -> dict[str, list[str]]:
    if split not in {"super_scarce", "scarce", "medium", "full"}:
        raise ValueError(f"Unknown split: {split}")
    result = {}
    for kind in KINDS:
        key = f"{split}_{kind}"
        if key not in manifest:
            raise KeyError(key)
        result[kind] = manifest[key][:max_per_split] if max_per_split else manifest[key]
    return result


def download_file(relative: str, root: Path) -> Path:
    target = root / relative
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            urlretrieve(f"{BASE}/{relative}", partial)
            break
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    partial.replace(target)
    return target


def ingest(root: Path, split: str, max_per_split: int = 0,
           progress=None) -> dict[str, list[str]]:
    manifest_path = download_file("splits/manifest.json", root)
    selected = selected_ids(json.loads(manifest_path.read_text()), split, max_per_split)
    def one_case(run: str) -> None:
        n = run.removeprefix("run_")
        for name in (f"boundary_{n}.vtp", f"boundary_cell_area_{n}.npy",
                     f"geo_ref_{n}.csv", f"force_mom_{n}.csv"):
            download_file(f"{run}/{name}", root)

    completed = 0
    total = sum(map(len, selected.values()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one_case, run): (kind, run)
                   for kind in KINDS for run in selected[kind]}
        for future in as_completed(futures):
            future.result()
            kind, run = futures[future]
            completed += 1
            if progress is not None:
                progress(completed, total, kind, run)
    return selected
