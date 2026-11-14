"""Offline check of modal_train.run_summary (timing + scores of a finished run).
Needs the `modal` package (import only, no account): run with .venv-modal's python."""
import json
from pathlib import Path

import pytest

pytest.importorskip("modal")
import modal_train  # noqa: E402


def _nb(path: Path, cells):
    path.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": [src], "metadata": {"papermill": {"duration": d}}}
        for src, d in cells] + [{"cell_type": "markdown", "source": ["x"], "metadata": {}}]}))


def test_run_summary_times_epochs_and_best_scores(tmp_path):
    _nb(tmp_path / "ex.ipynb", [("pip install", 5.0), ("model.train(dataset_dir=...)", 600.0), ("export", 20.5)])
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "run_state.json").write_text(json.dumps({"keypoint_mode": True, "dataset": "abc"}))
    (ck / "metrics.csv").write_text(
        "epoch,step,train/loss,val/mAP_50_95,val/ema_mAP_50_95\n"
        "0,10,1.0,,\n0,10,,0.30,0.20\n1,20,,0.62,0.50\n2,30,,0.55,0.69\n")
    s = modal_train.run_summary(tmp_path / "ex.ipynb", ck)
    assert s["total_cells_s"] == 625.5 and s["train_s"] == 600.0
    assert s["epochs"] == 3 and s["s_per_epoch"] == 200.0
    assert s["best"] == {"val/mAP_50_95": {"value": 0.62, "epoch": 1},
                         "val/ema_mAP_50_95": {"value": 0.69, "epoch": 2}}
    assert s["keypoint_mode"] is True


def test_run_summary_without_metrics(tmp_path):
    _nb(tmp_path / "ex.ipynb", [("model.train()", 3.0)])
    s = modal_train.run_summary(tmp_path / "ex.ipynb", tmp_path)
    assert s["epochs"] == 0 and "s_per_epoch" not in s and s["best"] == {} and s["keypoint_mode"] is None
