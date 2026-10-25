"""Run selected DrivAerML data preparation and Transolver++ training on Modal."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/potential_flow_ml"
DATA = Path("/data")
volume = modal.Volume.from_name("drivaer-surface-ml", create_if_missing=True)
base = (modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy==2.2.6", "pyvista==0.46.4", "vtk==9.5.2")
        .workdir(REMOTE))
source = dict(remote_path=REMOTE, ignore=[".git", ".venv", "data", "__pycache__"])
cpu_image = base.add_local_dir(ROOT, **source)
gpu_image = base.pip_install("nvidia-physicsnemo==2.2.0", "torch==2.14.1",
                             "torchvision==0.29.1").add_local_dir(ROOT, **source)
# Official upstream implementation is fetched at build time rather than copied
# into this project. Keep the revision fixed for reproducibility.
TRANSOLVER3_COMMIT = "ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37"
gpu_image_t3 = (base.apt_install("git")
                .pip_install("torch==2.14.1", "einops==0.8.1")
                .run_commands("git clone https://github.com/thuml/Transolver-3.git /opt/transolver3",
                              f"git -C /opt/transolver3 checkout {TRANSOLVER3_COMMIT}")
                .env({"PYTHONPATH": "/opt/transolver3"})
                .add_local_dir(ROOT, **source))
app = modal.App("potential-flow-drivaer-surface")


@app.function(image=cpu_image, volumes={str(DATA): volume}, timeout=14400, cpu=4)
def ingest_job(split: str, max_per_split: int):
    from data import ingest

    def progress(done, total, kind, run):
        if done % 5 == 0 or done == total:
            volume.commit()
            print(f"Downloaded {done}/{total} surface cases ({kind}: {run})", flush=True)

    selected = ingest(DATA / "raw", split, max_per_split, progress=progress)
    volume.commit()
    return {kind: len(ids) for kind, ids in selected.items()}


@app.function(image=cpu_image, volumes={str(DATA): volume}, timeout=14400, cpu=8, memory=32768)
def prepare_job(split: str, max_per_split: int, points_per_case: int):
    from preprocess import prepare_all

    def progress(done, total, run):
        if done % 5 == 0 or done == total:
            volume.commit()
            print(f"Prepared {done}/{total} surface cases (last: {run})", flush=True)

    result = prepare_all(DATA / "raw", DATA / "prepared", split,
                         max_per_split, points_per_case, progress=progress)
    volume.commit()
    return {kind: {"n": len(rows),
                   "max_abs_cd_closure": max(abs(row["full_minus_source"]) for row in rows)}
            for kind, rows in result["cases"].items()}


@app.function(image=cpu_image, volumes={str(DATA): volume}, timeout=14400, cpu=8, memory=32768)
def prewarm_job(split: str, points_per_case: int):
    from preprocess import prepare_available

    def progress(done, total, run):
        if done % 5 == 0 or done == total:
            volume.commit()
            print(f"Prepared {done}/{total} available cases (last: {run})", flush=True)

    count = prepare_available(DATA / "raw", DATA / "prepared", split,
                              points_per_case, progress=progress)
    volume.commit()
    return {"ready_and_prepared": count}


@app.function(image=gpu_image, volumes={str(DATA): volume}, timeout=14400,
              gpu="A100-40GB", cpu=8, memory=32768)
def fit_job(split: str, max_per_split: int, epochs: int, run_tag: str):
    from train import train

    if run_tag and not all(c.isalnum() or c in "_-" for c in run_tag):
        raise ValueError("run_tag may contain only letters, digits, underscores, and dashes")
    name = f"{split}_{max_per_split or 'all'}"
    output = DATA / "runs" / (f"{name}_{run_tag}" if run_tag else name)
    result = train(DATA / "prepared", output, split, max_per_split, epochs,
                   on_checkpoint=lambda epoch: volume.commit())
    volume.commit()
    return result


@app.function(image=gpu_image_t3, volumes={str(DATA): volume}, timeout=14400,
              gpu="A100-40GB", cpu=8, memory=32768)
def fit_t3_job(split: str, max_per_split: int, epochs: int, run_tag: str):
    from train import train

    if run_tag and not all(c.isalnum() or c in "_-" for c in run_tag):
        raise ValueError("run_tag may contain only letters, digits, underscores, and dashes")
    name = f"transolver3_{split}_{max_per_split or 'all'}"
    output = DATA / "runs" / (f"{name}_{run_tag}" if run_tag else name)
    result = train(DATA / "prepared", output, split, max_per_split, epochs,
                   on_checkpoint=lambda epoch: volume.commit(),
                   model_family="transolver3")
    volume.commit()
    return result


@app.local_entrypoint()
def ingest(split: str = "super_scarce", max_per_split: int = 2):
    print(ingest_job.remote(split, max_per_split))


@app.local_entrypoint()
def prepare(split: str = "super_scarce", max_per_split: int = 2,
            points_per_case: int = 32768):
    print(prepare_job.remote(split, max_per_split, points_per_case))


@app.local_entrypoint()
def prewarm(split: str = "super_scarce", points_per_case: int = 32768):
    print(prewarm_job.remote(split, points_per_case))


@app.local_entrypoint()
def fit(split: str = "super_scarce", max_per_split: int = 2,
        epochs: int = 2, run_tag: str = ""):
    print(fit_job.remote(split, max_per_split, epochs, run_tag))


@app.local_entrypoint()
def fit_t3(split: str = "super_scarce", max_per_split: int = 2,
           epochs: int = 2, run_tag: str = ""):
    print(fit_t3_job.remote(split, max_per_split, epochs, run_tag))
