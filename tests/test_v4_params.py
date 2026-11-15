"""v4 notebooks: MAX_EPOCHS + AMP_DTYPE parameters reach model.train(); fp16 on GPUs without
real bf16 (T4: torch says bf16 is "supported" only through emulation). Runs the real cells with
a fake rfdetr + fake torch (no GPU). Reuses test_v2_resume's helpers."""
import json
import sys
import types
import zipfile

import pytest

from test_v2_resume import FakeModel, _cell, env  # noqa: F401 -- env is a fixture

NOTEBOOKS = ["train_object_detector_v4.ipynb", "train_object_detector_seg_v4.ipynb"]
BOX_NB = NOTEBOOKS[0]
BUILT = []


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
    # v4 MODEL_SIZE / RESOLUTION: constructors take kwargs, model.model.resolution exists
    defaults = {"keypoint": 576, "box": 384, "small": 512, "medium": 576}

    def make(kind):
        def ctor(**kw):
            m = FakeModel(kind)
            m.kw = kw
            m.model = types.SimpleNamespace(resolution=kw.get("resolution", defaults[kind]))
            BUILT.append(m)
            return m
        return ctor

    BUILT.clear()
    fake = sys.modules["rfdetr"]
    fake.RFDETRKeypointPreview, fake.RFDETRNano = make("keypoint"), make("box")
    fake.RFDETRSmall, fake.RFDETRMedium = make("small"), make("medium")


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


@pytest.mark.parametrize("size,kind", [(None, "small"), ("keypoint", "keypoint"), ("nano", "box"),
                                       ("small", "small"), ("medium", "medium")])
def test_model_size_and_resolution(env, monkeypatch, size, kind):
    _fake_torch(monkeypatch)
    _train(BOX_NB, env, MODEL_SIZE=size, RESOLUTION=384)
    assert FakeModel.calls[-1][0] == kind and BUILT[-1].kw == {"resolution": 384}
    state = json.loads((env / "ckpt" / "run_state.json").read_text())
    assert state["keypoint_mode"] is (size == "keypoint") and state["model_size"] == (size or "small")
    assert state["resolution"] == 384


def test_defaults_build_box_small_at_its_own_resolution(env, monkeypatch):
    _fake_torch(monkeypatch)
    _train(BOX_NB, env)
    assert FakeModel.calls[-1][0] == "small" and BUILT[-1].kw == {}


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_labels_without_keypoint(env, monkeypatch, nb_name):
    """v4: the app no longer asks for a click point; keypoint missing = box centre (box nb)."""
    labels_path = env / "upload" / "labels.json"
    data = json.loads(labels_path.read_text())
    for e in data["labels"]:
        for o in e["objects"]:
            del o["keypoint"]
    labels_path.write_text(json.dumps(data))
    _fake_torch(monkeypatch)
    _train(nb_name, env)
    ann = json.loads((env / "dataset_coco" / "train" / "_annotations.coco.json").read_text())["annotations"]
    if nb_name == BOX_NB:
        assert all(a["keypoints"] == [15, 10, 2] for a in ann)  # box [5, 5, 20, 10] centre
    assert ann


def test_other_size_never_resumes(env, monkeypatch):
    _fake_torch(monkeypatch)
    _train(BOX_NB, env, MODEL_SIZE="small")
    with zipfile.ZipFile(env / "ckpt" / "last.ckpt", "w") as z:  # a real (zip) checkpoint passes ckpt_ok
        z.writestr("data.pkl", b"x")
    kw = _train(BOX_NB, env, MODEL_SIZE="small")  # same choice: resumes
    assert kw["resume"] == str(env / "ckpt" / "last.ckpt")
    kw = _train(BOX_NB, env, MODEL_SIZE="medium")  # other size: fresh start
    assert kw["resume"] is None and FakeModel.calls[-1][0] == "medium"


def test_unknown_size_is_an_error(env, monkeypatch):
    _fake_torch(monkeypatch)
    with pytest.raises(ValueError, match="MODEL_SIZE"):
        _train(BOX_NB, env, MODEL_SIZE="huge")


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
@pytest.mark.parametrize("value,expected", [(None, "auto"), (4, 4), ("4", 4)])
def test_batch_size_parameter(env, monkeypatch, nb_name, value, expected):
    """v4: local Windows runs pass a fixed BATCH_SIZE (auto overfills 8 GB via shared memory)."""
    _fake_torch(monkeypatch, native_bf16=True)
    kw = _train(nb_name, env, BATCH_SIZE=value, MODEL_SIZE="small") if nb_name == BOX_NB else         _train(nb_name, env, BATCH_SIZE=value)
    assert kw["batch_size"] == expected
