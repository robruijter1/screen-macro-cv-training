"""Modal launcher for the v4 training notebooks (paid training, Game Bot Creator's own account).

Runs train_object_detector{,_seg}_v4.ipynb unchanged through papermill on a Modal GPU in the EU,
with the notebook's parameters pointed at a per-job folder on the `gbc-training` Volume:

    /vol/jobs/<job_id>/dataset/      dataset .zip (uploaded by the caller), or a shared
                                     /vol/datasets/<name>/ folder passed as dataset_dir
    /vol/jobs/<job_id>/checkpoints/  rfdetr last.ckpt + best checkpoints + metrics.csv (resume)
    /vol/jobs/<job_id>/output/       best.onnx + executed.ipynb (full log) + run_summary.json

GPU functions on Modal are always preemptible; a preempted run restarts on the same input
(same job_id) and the notebook resumes from checkpoints/last.ckpt. Volumes commit in the
background every few seconds, plus once on shutdown.

Dev PC (needs `modal token new` once):
    .venv-modal\\Scripts\\modal run modal_train.py --dataset path\\to\\x_dataset.zip [--model rfdetr_seg] [--job-id ID] [--gpu T4]
    .venv-modal\\Scripts\\modal run modal_train.py::bench --out results.json   (timing runs, see bench())
Re-running with the same --job-id resumes that job. Later, the gateway calls `train.spawn()`.
"""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "rfdetr": "train_object_detector_v4.ipynb",
    "rfdetr_seg": "train_object_detector_seg_v4.ipynb",
}
VOL = "/vol"
# Fallback list: first type with capacity wins. Cheapest per unit of work first (2026-11-14 bench:
# T4 costs 2-3x A10/L4 for box+kp). GBC_TRAIN_GPU="T4" (comma-separated) overrides; --gpu per call.
GPU = os.environ.get("GBC_TRAIN_GPU", "A10,L4,T4").split(",")
REGION = os.environ.get("GBC_TRAIN_REGION", "eu")  # all our services are EU; 1.15x price
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


def run_summary(executed_nb: Path, checkpoint_dir: Path) -> dict:
    """Timing + scores of one finished run: papermill cell durations, epochs and best val
    mAPs from rfdetr's metrics.csv. Pure (files only) so it is testable offline."""
    nb = json.loads(Path(executed_nb).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    durs = [float((c.get("metadata", {}).get("papermill") or {}).get("duration") or 0) for c in cells]
    train_s = sum(d for c, d in zip(cells, durs) if "model.train(" in "".join(c["source"]))
    out = {"total_cells_s": round(sum(durs), 1), "train_s": round(train_s, 1),
           "epochs": 0, "best": {}, "keypoint_mode": None}
    state = Path(checkpoint_dir) / "run_state.json"  # which model actually trained
    if state.exists():
        out["keypoint_mode"] = json.loads(state.read_text()).get("keypoint_mode")
    metrics = Path(checkpoint_dir) / "metrics.csv"
    if metrics.exists():
        with open(metrics, newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = {int(float(r["epoch"])) for r in rows if r.get("epoch") not in (None, "")}
        out["epochs"] = (max(epochs) + 1) if epochs else 0
        for r in rows:
            for k, v in r.items():
                if not (k and k.startswith("val/") and "mAP" in k and v not in (None, "")):
                    continue
                v = float(v)
                if k not in out["best"] or v > out["best"][k]["value"]:
                    out["best"][k] = {"value": round(v, 4), "epoch": int(float(r.get("epoch") or 0))}
    if out["epochs"]:
        out["s_per_epoch"] = round(train_s / out["epochs"], 2)
    return out


class UsageSampler:
    """Background thread: every `every_s`, CPU cores used by this process tree (papermill +
    kernel + DataLoader workers) and GPU utilisation %. Answers "GPU-bound or CPU-starved?"."""

    def __init__(self, every_s: float = 5.0):
        import threading

        self.every_s, self.rows = every_s, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _cpu_s(self):
        import psutil

        me = psutil.Process()
        total = 0.0
        for p in [me] + me.children(recursive=True):
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def _gpu(self):
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        try:
            util, mem = (float(x) for x in out.strip().split(","))
            return util, mem
        except ValueError:
            return None, None

    def _run(self):
        last_t, last_cpu = time.time(), self._cpu_s()
        while not self._stop.wait(self.every_s):
            now, cpu = time.time(), self._cpu_s()
            util, mem = self._gpu()
            self.rows.append({"t": round(now, 1), "cores": round((cpu - last_cpu) / (now - last_t), 2),
                              "gpu_util": util, "gpu_mem_mib": mem})
            last_t, last_cpu = now, cpu

    def start(self):
        self._t.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=30)
        return usage_stats(self.rows)


def usage_stats(rows: list[dict]) -> dict:
    """Mean/median CPU cores and GPU % over the samples where the GPU was in use (training)."""
    busy = [r for r in rows if r.get("gpu_mem_mib") and r["gpu_mem_mib"] > 1000]
    if not busy:
        return {"samples": len(rows)}

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    cores = [r["cores"] for r in busy]
    util = [r["gpu_util"] for r in busy if r["gpu_util"] is not None]
    return {"samples": len(rows), "busy_samples": len(busy),
            "cores_mean": round(sum(cores) / len(cores), 2), "cores_median": med(cores),
            "gpu_util_mean": round(sum(util) / len(util), 1) if util else None,
            "gpu_util_median": med(util) if util else None,
            "gpu_mem_max_mib": max(r["gpu_mem_mib"] for r in busy)}


@app.function(gpu=GPU, region=REGION, timeout=TIMEOUT_S, volumes={VOL: volume})
def train(job_id: str, model_choice: str = "rfdetr", dataset_dir: str = "", num_workers: int = -1,
          max_epochs: int = 0, extra_params: dict | None = None) -> dict:
    """Runs the notebook for `model_choice` on job `job_id`. `dataset_dir` (Volume-relative,
    e.g. "datasets/x") replaces the job's own dataset/ folder. Returns {"onnx": Volume path
    of best.onnx, "summary": run_summary() + gpu, wall time, train image count, usage}.
    `num_workers` >= 0 sets the notebook's DataLoader workers (-1 = notebook default);
    `max_epochs` > 0 caps the epochs (0 = notebook default, 200); `extra_params` = more notebook
    parameters (e.g. {"MODEL_SIZE": "small", "RESOLUTION": 384})."""
    import subprocess

    import papermill as pm

    t0 = time.time()
    if model_choice not in NOTEBOOKS:
        raise ValueError(f"unknown model_choice {model_choice!r}")
    paths = job_paths(job_id)
    if dataset_dir:
        paths["dataset"] = f"{VOL}/{dataset_dir.strip('/')}"
    for key in ("checkpoints", "output"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)  # dataset_raw/, dataset_coco/ etc. stay off the Volume
    executed = f"{paths['output']}/executed.ipynb"
    params = {"DATASET_DIR": paths["dataset"], "OUTPUT_DIR": paths["output"],
              "CHECKPOINT_DIR": paths["checkpoints"]}
    if num_workers >= 0:
        params["NUM_WORKERS"] = num_workers
    if max_epochs > 0:
        params["MAX_EPOCHS"] = max_epochs
    params.update(extra_params or {})
    sampler = UsageSampler().start()
    pm.execute_notebook(
        f"/notebooks/{NOTEBOOKS[model_choice]}", executed,
        parameters=params, kernel_name="python3", cwd=str(work), log_output=True,
    )
    usage = sampler.stop()
    Path(paths["output"], "usage.json").write_text(json.dumps(sampler.rows))
    summary = run_summary(Path(executed), Path(paths["checkpoints"]))
    summary["usage"] = usage
    summary["cpu_request"] = os.environ.get("MODAL_CPU_REQUEST", "")
    summary["num_workers"] = num_workers
    summary["max_epochs"] = max_epochs
    summary["extra_params"] = extra_params or {}
    summary["host_cpus"] = os.cpu_count()
    summary["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True).stdout.strip()
    summary["region"] = os.environ.get("MODAL_REGION", "")
    summary["wall_s"] = round(time.time() - t0, 1)
    coco = work / "dataset_coco"
    summary["images"] = {s: len(json.loads((coco / s / "_annotations.coco.json").read_text())["images"])
                         for s in ("train", "valid") if (coco / s / "_annotations.coco.json").exists()}
    Path(paths["output"], "run_summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    onnx = Path(paths["output"]) / "best.onnx"
    if not onnx.exists():
        raise RuntimeError("notebook finished without best.onnx")
    return {"onnx": str(onnx.relative_to(VOL)), "summary": summary}


@app.local_entrypoint()
def main(dataset: str, model: str = "rfdetr", job_id: str = "", out: str = "", gpu: str = ""):
    """Dev test: upload `dataset` zip, train, download best.onnx next to the zip (or to `out`)."""
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
    fn = train.with_options(gpu=gpu.split(",")) if gpu else train
    res = fn.remote(job_id, model)
    print(f"train finished in {(time.time() - t0) / 60:.1f} min: {json.dumps(res['summary'])}")
    dest = Path(out) if out else Path(dataset).with_name(f"{Path(dataset).stem}_modal.onnx")
    with open(dest, "wb") as f:
        for chunk in volume.read_file(res["onnx"]):
            f.write(chunk)
    print(f"downloaded {res['onnx']} -> {dest}")


@app.local_entrypoint()
def bench(out: str, datasets: str = "poe_mhb_20261103,goblins_20260914", models: str = "rfdetr,rfdetr_seg",
          gpus: str = "T4,L4,A10", tag: str = "", max_wait_h: float = 4.0, cpu_variants: str = "",
          max_epochs: int = 0, param_sets: str = ""):
    """Timing runs for the price table: every dataset x model x GPU (one GPU type each, no
    fallback), all spawned at once (Starter = 10 GPUs; the rest queue). Datasets must already
    be on the Volume under datasets/<name>/. Results are written to `out` as they come in;
    calls still unfinished after `max_wait_h` are cancelled and recorded as such.
    `cpu_variants` = "default,8:6" -> per GPU one run with Modal's default CPU and one with
    cpu=8 + NUM_WORKERS=6 ("<cores>:<workers>"). `param_sets` = JSON {"name": {notebook params}}:
    one run per set (job id gets -<name>), e.g. '{"kp384": {"RESOLUTION": 384}, "nano": {"MODEL_SIZE": "nano"}}'."""
    tag = tag or time.strftime("%Y%m%d-%H%M")
    sets = list(json.loads(param_sets).items()) if param_sets else [("", {})]
    cpus = cpu_variants.split(",") if cpu_variants else ["default"]
    calls = []
    for ds in datasets.split(","):
        for m in models.split(","):
            for g in gpus.split(","):
                for pname, extra in sets:
                    for v in cpus:
                        opts, workers, suffix = {"gpu": g}, -1, ""
                        if v != "default":
                            cores, workers = v.split(":")
                            opts["cpu"], workers, suffix = float(cores), int(workers), f"-cpu{cores}w{workers}"
                        suffix += f"-{pname}" if pname else ""
                        job = f"bench-{tag}-{ds}-{m}-{g}{suffix}".lower()
                        fc = train.with_options(**opts).spawn(job, m, f"datasets/{ds}", workers, max_epochs, extra)
                        calls.append(({"dataset": ds, "model": m, "gpu": g, "variant": v, "params": pname,
                                       "job_id": job, "call_id": fc.object_id}, fc))
                        print(f"spawned {job} ({fc.object_id})")
    deadline = time.time() + max_wait_h * 3600
    results = []
    for meta, fc in calls:
        try:
            meta = {**meta, **fc.get(timeout=max(1.0, deadline - time.time()))}
        except (TimeoutError, modal.exception.TimeoutError):
            fc.cancel()
            meta["error"] = f"not finished after {max_wait_h} h -- cancelled"
        except Exception as exc:  # noqa: BLE001 -- record and keep collecting the others
            meta["error"] = f"{type(exc).__name__}: {exc}"
        results.append(meta)
        Path(out).write_text(json.dumps(results, indent=1))
        print(json.dumps(meta))
