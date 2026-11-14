"""v4 notebooks: MAX_EPOCHS + AMP_DTYPE parameters reach model.train(); fp16 on GPUs without
real bf16 (T4: torch says bf16 is "supported" only through emulation). Runs the real cells with
a fake rfdetr + fake torch (no GPU). Reuses test_v2_resume's helpers."""
import sys
import types

import pytest

from test_v2_resume import FakeModel, _cell, env  # noqa: F401 -- env is a fixture

NOTEBOOKS = ["train_object_detector_v4.ipynb", "train_object_detector_seg_v4.ipynb"]


def _fake_torch(monkeypatch, cuda=True, native_bf16=False):
    def is_bf16_supported(including_emulation=True):
        return cuda and (native_bf16 or including_emulation)

    cuda_mod = types.SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=is_bf16_supported)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda_mod))
    # v3+ checkpoint cell: Lightning callback + rfdetr.training.build_trainer hook
    monkeypatch.setitem(sys.modules, "pytorch_lightning", types.SimpleNamespace(Callback=object))
    training = types.ModuleType("rfdetr.training")
    training.build_trainer = lambda *a, **k: types.SimpleNamespace(callbacks=[])
    monkeypatch.setitem(sys.modules, "rfdetr.training", training)
    sys.modules["rfdetr"].training = training


def _train(nb_name, tmp_path, **params):
    g = {}
    exec(_cell(nb_name, "DATASET_DIR = None"), g)
    g.update(DATASET_DIR=str(tmp_path / "upload"), OUTPUT_DIR=str(tmp_path / "out"),
             CHECKPOINT_DIR=str(tmp_path / "ckpt"), **params)
    for marker in ("RAW_DIR = Path", "VALID_FRACTION", "RUN_STATE_PATH", "EARLY_STOPPING_PATIENCE ="):
        exec(_cell(nb_name, marker), g)
    return FakeModel.calls[-1][1]


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_parameters_default_to_none(nb_name):
    g = {}
    exec(_cell(nb_name, "DATASET_DIR = None"), g)
    assert g["MAX_EPOCHS"] is None and g["AMP_DTYPE"] is None


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_t4_like_gpu_trains_fp16_and_200_epochs(env, monkeypatch, nb_name):
    _fake_torch(monkeypatch, native_bf16=False)  # emulated bf16 only
    kw = _train(nb_name, env)
    assert kw["amp_dtype"] == "fp16" and kw["epochs"] == 200


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_ampere_gpu_keeps_bf16(env, monkeypatch, nb_name):
    _fake_torch(monkeypatch, native_bf16=True)
    assert _train(nb_name, env)["amp_dtype"] == "bf16"


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_max_epochs_and_explicit_amp(env, monkeypatch, nb_name):
    _fake_torch(monkeypatch, native_bf16=False)
    kw = _train(nb_name, env, MAX_EPOCHS=3, AMP_DTYPE="bf16")
    assert kw["epochs"] == 3 and kw["amp_dtype"] == "bf16"


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_no_gpu_still_runs(env, monkeypatch, nb_name):
    _fake_torch(monkeypatch, cuda=False)  # CPU: rfdetr trains 32-bit regardless of amp_dtype
    assert _train(nb_name, env)["amp_dtype"] == "fp16"
