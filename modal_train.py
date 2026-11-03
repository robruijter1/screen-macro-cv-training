"""Modal launcher for the v2 training notebooks (paid training, Game Bot Creator's own account).

Runs train_object_detector{,_seg}_v2.ipynb unchanged through papermill on a Modal GPU, with
the notebook's parameters pointed at a per-job folder on the `gbc-training` Volume:

    /vol/jobs/<job_id>/dataset/      dataset .zip (uploaded by the caller)
    /vol/jobs/<job_id>/checkpoints/  rfdetr last.ckpt + best checkpoints (resume)
    /vol/jobs/<job_id>/output/       best.onnx + executed.ipynb (full log)

GPU functions on Modal are always preemptible; a preempted run restarts on the same input
(same job_id) and the notebook resumes from checkpoints/last.ckpt. Volumes commit in the
background every few seconds, plus once on shutdown.

Test from a dev PC (needs `modal token new` once):
    .venv-modal\\Scripts\\modal run modal_train.py --dataset path\\to\\x_dataset.zip [--model rfdetr_seg] [--job-id ID]
Re-running with the same --job-id resumes that job. Later, the gateway calls `train.spawn()`.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "rfdetr": "train_object_detector_v2.ipynb",
    "rfdetr_seg": "train_object_detector_seg_v2.ipynb",
}
VOL = "/vol"
# Fallback list: first type with capacity wins (L4 sat unscheduled for 30+ min on 2026-11-03).
# GBC_TRAIN_GPU="T4" (comma-separated) overrides. GPUs need a payment method on the account.
GPU = os.environ.get("GBC_TRAIN_GPU", "L4,A10,T4").split(",")
TIMEOUT_S = 6 * 3600  # Modal default is 5 min; max 24 h

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # cv2 runtime libs, in case a dep pulls full opencv
    .pip_install(
        # same pins as the notebooks' own install cell (that cell then finds them satisfied)
        "rfdetr[train,loggers,augment]==1.9.4", "onnx==1.22.0", "onnxruntime==1.29.0",
        "papermill", "ipykernel", "opencv-python-headless", "psutil",
    )
)
for _nb in NOTEBOOKS.values():
    image = image.add_local_file(HERE / _nb, f"/notebooks/{_nb}")

app = modal.App("gbc-training", image=image)
volume = modal.Volume.from_name("gbc-training", create_if_missing=True)


def job_paths(job_id: str, root: str = VOL) -> dict[str, str]:
    base = f"{root}/jobs/{job_id}"
    return {"dataset": f"{base}/dataset", "checkpoints": f"{base}/checkpoints", "output": f"{base}/output"}


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOL: volume})
def train(job_id: str, model_choice: str = "rfdetr") -> str:
    """Runs the notebook for `model_choice` on job `job_id`; returns best.onnx's Volume path."""
    import papermill as pm

    if model_choice not in NOTEBOOKS:
        raise ValueError(f"unknown model_choice {model_choice!r}")
    paths = job_paths(job_id)
    for key in ("checkpoints", "output"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)  # dataset_raw/, dataset_coco/ etc. stay off the Volume
    pm.execute_notebook(
        f"/notebooks/{NOTEBOOKS[model_choice]}",
        f"{paths['output']}/executed.ipynb",
        parameters={"DATASET_DIR": paths["dataset"], "OUTPUT_DIR": paths["output"],
                    "CHECKPOINT_DIR": paths["checkpoints"]},
        kernel_name="python3", cwd=str(work), log_output=True,
    )
    volume.commit()
    onnx = Path(paths["output"]) / "best.onnx"
    if not onnx.exists():
        raise RuntimeError("notebook finished without best.onnx")
    return str(onnx.relative_to(VOL))


@app.local_entrypoint()
def main(dataset: str, model: str = "rfdetr", job_id: str = "", out: str = ""):
    """Dev test: upload `dataset` zip, train, download best.onnx next to the zip (or to `out`)."""
    import time

    job_id = job_id or f"test-{uuid.uuid4().hex[:8]}"
    remote_dir = job_paths(job_id, root="")["dataset"]
    try:
        have = any(e.path.endswith("dataset.zip") for e in volume.listdir(remote_dir))
    except Exception:  # job folder doesn't exist yet
        have = False
    if have:
        print("dataset already on the Volume for this job -- skipping upload")
    else:
        t_up = time.time()
        with volume.batch_upload(force=True) as batch:
            batch.put_file(dataset, f"{remote_dir}/dataset.zip")
        print(f"uploaded dataset in {(time.time() - t_up) / 60:.1f} min")
    print(f"job_id={job_id} (re-run with --job-id {job_id} to resume)")
    t0 = time.time()
    onnx_rel = train.remote(job_id, model)
    print(f"train finished in {(time.time() - t0) / 60:.1f} min")
    dest = Path(out) if out else Path(dataset).with_name(f"{Path(dataset).stem}_modal.onnx")
    with open(dest, "wb") as f:
        for chunk in volume.read_file(onnx_rel):
            f.write(chunk)
    print(f"downloaded {onnx_rel} -> {dest}")
