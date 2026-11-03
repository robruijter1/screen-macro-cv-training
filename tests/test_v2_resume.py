"""Offline check of the v2 notebooks' dataset/checkpoint/resume cells: runs the real cell
code with a fake `rfdetr` (no GPU, no torch). Needs numpy + opencv-python + pytest.
Run: python -m pytest tests"""
import json
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["train_object_detector_v2.ipynb", "train_object_detector_seg_v2.ipynb"]


def _cells(nb_name):
    nb = json.loads((REPO / nb_name).read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _code(cell):
    # IPython-only lines (!pip, x = !cmd, %magic) -> no-ops
    lines = "".join(cell["source"]).splitlines()
    return "\n".join("pass" if re.match(r"\s*(!|%)|.*= !", ln) else ln for ln in lines)


def _cell(nb_name, marker):
    hits = [c for c in _cells(nb_name) if marker in "".join(c["source"])]
    assert len(hits) == 1, marker
    return _code(hits[0])


class FakeModel:
    calls = []
    fail_keypoint = False

    def __init__(self, kind):
        self.kind = kind

    def train(self, **kw):
        FakeModel.calls.append((self.kind, kw))
        if self.kind == "keypoint" and FakeModel.fail_keypoint:
            Path(kw["output_dir"], "last.ckpt").write_text("kp")  # partial keypoint run
            raise RuntimeError("keypoint training failed")
        Path(kw["output_dir"], "last.ckpt").write_text(self.kind)


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = types.ModuleType("rfdetr")
    fake.RFDETRKeypointPreview = lambda: FakeModel("keypoint")
    fake.RFDETRNano = lambda: FakeModel("box")
    fake.RFDETRSegNano = lambda: FakeModel("seg")
    monkeypatch.setitem(sys.modules, "rfdetr", fake)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", None)  # ceiling-stop hook -> its own except
    FakeModel.calls, FakeModel.fail_keypoint = [], False
    monkeypatch.chdir(tmp_path)
    ds = tmp_path / "upload" / "images"
    ds.mkdir(parents=True)
    labels = []
    for i in range(4):
        cv2.imwrite(str(ds / f"f{i}.png"), np.full((40, 60, 3), 90, np.uint8))
        obj = {"box": [5, 5, 20, 10], "keypoint": [15, 10], "polygon": [[5, 5], [25, 5], [25, 15], [5, 15]]}
        labels.append({"image": f"f{i}.png", "objects": [obj]})
    (ds.parent / "labels.json").write_text(json.dumps({"images_dir": "images", "labels": labels}))
    return tmp_path


def _run(nb_name, tmp_path, train=True):
    g = {}
    exec(_cell(nb_name, "DATASET_DIR = None"), g)
    g.update(DATASET_DIR=str(tmp_path / "upload"), OUTPUT_DIR=str(tmp_path / "out"),
             CHECKPOINT_DIR=str(tmp_path / "ckpt"))
    exec(_cell(nb_name, "RAW_DIR = Path"), g)
    exec(_cell(nb_name, "VALID_FRACTION"), g)
    exec(_cell(nb_name, "RUN_STATE_PATH"), g)
    if train:
        exec(_cell(nb_name, "EARLY_STOPPING_PATIENCE ="), g)
    return g


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_fresh_then_resume(env, nb_name):
    _run(nb_name, env)
    kind, kw = FakeModel.calls[-1]
    assert kw["resume"] is None
    assert kw["output_dir"] == str(env / "ckpt") and kw["checkpoint_interval"] == 100_000
    _run(nb_name, env)  # second run = interrupted run restarted
    assert FakeModel.calls[-1][1]["resume"] == str(env / "ckpt" / "last.ckpt")


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_other_dataset_clears_checkpoints(env, nb_name):
    _run(nb_name, env)
    labels_path = env / "upload" / "labels.json"
    data = json.loads(labels_path.read_text())
    data["labels"].pop()
    labels_path.write_text(json.dumps(data))
    g = _run(nb_name, env, train=False)
    assert not (env / "ckpt" / "last.ckpt").exists() and g["run_state"] == {}


def test_keypoint_fallback_resets_and_stays_box_only(env):
    nb = NOTEBOOKS[0]
    FakeModel.fail_keypoint = True
    _run(nb, env)
    assert [c[0] for c in FakeModel.calls] == ["keypoint", "box"]
    assert "resume" not in FakeModel.calls[-1][1]  # fallback never resumes keypoint checkpoints
    assert (env / "ckpt" / "last.ckpt").read_text() == "box"
    state = json.loads((env / "ckpt" / "run_state.json").read_text())
    assert state["keypoint_mode"] is False
    FakeModel.calls = []
    _run(nb, env)  # resume: straight to box-only, from the box checkpoint
    assert [c[0] for c in FakeModel.calls] == ["box"]
    assert FakeModel.calls[0][1]["resume"] == str(env / "ckpt" / "last.ckpt")


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_dataset_zip_and_no_colab(env, nb_name, tmp_path):
    import shutil
    shutil.make_archive(str(tmp_path / "zipped" / "ds"), "zip", env / "upload")
    shutil.rmtree(env / "upload")
    (env / "upload").mkdir()
    shutil.move(str(tmp_path / "zipped" / "ds.zip"), env / "upload" / "ds.zip")
    g = _run(nb_name, env, train=False)
    assert (g["RAW_DIR"] / "labels.json").exists()
    text = (REPO / nb_name).read_text(encoding="utf-8")
    assert "google.colab" not in text and "files.upload" not in text
