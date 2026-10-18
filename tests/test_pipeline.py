# tests/test_pipeline.py
"""Prueba de humo del pipeline completo con datos sintéticos.  Ejecutar: pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from make_synthetic_mitdb import make_dataset            # noqa: E402
from segmentacion_ecg import normalize, cut_windows, WINDOW_LEN   # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_normalize_and_cut():
    sig = np.sin(np.linspace(0, 50, 2000)).astype(np.float32)
    wins, valid = cut_windows(sig, np.array([5, 500, 1000, 1995]))
    assert valid.tolist() == [False, True, True, False]
    assert wins.shape == (2, WINDOW_LEN)
    assert np.allclose(wins.mean(1), 0, atol=1e-5) and np.allclose(wins.std(1), 1, atol=1e-4)
    assert np.isfinite(normalize(np.ones(10))).all()      # señal plana no produce NaN


def test_end_to_end(tmp_path):
    data = make_dataset(tmp_path / "mitdb")
    out, model_dir = tmp_path / "data", tmp_path / "output"

    run("ingesta_mitdb.py", "--records", data, "--outdir", out)
    splits = json.loads((out / "splits.json").read_text())
    assert "102" not in sum(splits.values(), [])                    # paced excluido
    assert set(splits["test"]) == {"100", "103"}                    # DS2
    assert not set(splits["train"]) & set(splits["val"])            # sin fuga entre splits

    run("segmentacion_ecg.py", "--records", data, "--beats", out / "beats.parquet", "--outdir", out)
    npz = np.load(out / "windows.npz")
    assert npz["X"].shape[1:] == (1, WINDOW_LEN) and set(npz["y"]) <= {0, 1, 2, 3, 4}
    assert "114" in set(npz["groups"])                              # MLII hallado en canal 1

    run("modelo_cnn.py", "--windows", out / "windows.npz", "--splits", out / "splits.json",
        "--outdir", model_dir, "--epochs", "3", "--batch", "256")
    assert (model_dir / "modelo_arritmias.pt").exists()
    assert (model_dir / "metricas_test.json").exists()

    txt = run("inferencia.py", "--model", model_dir / "modelo_arritmias.pt",
              "--record", data / "100")
    assert "latidos clasificados" in txt
