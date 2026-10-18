# tests/make_synthetic_mitdb.py
"""
Genera registros ECG SINTÉTICOS con el mismo formato que MIT-BIH (.hea/.dat/.atr).
Sirve solo para probar que el pipeline corre de punta a punta sin descargar PhysioNet.
Las métricas obtenidas con estos datos NO significan nada clínico.
"""
from pathlib import Path
import numpy as np
import wfdb

FS = 360


def _beat(kind: str, n: int = 216) -> np.ndarray:
    t = (np.arange(n) - n // 2) / FS
    g = lambda mu, s, a: a * np.exp(-0.5 * ((t - mu) / s) ** 2)
    if kind == "V":      # ancho, grande, onda T invertida
        return g(0, 0.045, 1.6) - g(0.02, 0.05, 0.4) - g(0.2, 0.06, 0.5)
    if kind == "F":      # intermedio
        return g(0, 0.03, 1.1) + g(0.06, 0.04, -0.5) + g(0.2, 0.05, 0.2)
    if kind == "Q":      # espiga de marcapasos + QRS ancho pequeño
        return g(-0.05, 0.004, 1.2) + g(0.01, 0.04, 0.7) + g(0.2, 0.05, 0.1)
    if kind == "S":      # QRS normal sin onda P
        return g(0, 0.012, 1.0) + g(0.2, 0.04, 0.3)
    return g(-0.16, 0.02, 0.12) + g(0, 0.012, 1.0) + g(0.2, 0.04, 0.3)   # N con onda P


SYMBOL = {"N": "N", "S": "A", "V": "V", "F": "F", "Q": "/"}


def make_record(name: str, out_dir: Path, seed: int, mlii_channel: int = 0,
                seconds: int = 120) -> None:
    rng = np.random.default_rng(seed)
    n = seconds * FS
    sig = 0.05 * rng.standard_normal(n)
    sig += 0.3 * np.sin(2 * np.pi * 0.3 * np.arange(n) / FS)          # deriva de línea base
    pos = int(0.8 * FS)
    samples, symbols = [], []
    while pos < n - FS:
        kind = rng.choice(list("NSVFQ"), p=[0.72, 0.09, 0.1, 0.04, 0.05])
        w = _beat(kind)
        a, b = pos - 108, pos + 108
        sig[a:b] += w
        samples.append(pos); symbols.append(SYMBOL[kind])
        pos += int(FS * rng.uniform(0.7, 0.95) * (0.7 if kind == "S" else 1.0))
    other = 0.5 * sig + 0.05 * rng.standard_normal(n)
    chans = [sig, other] if mlii_channel == 0 else [other, sig]
    names = ["MLII", "V5"] if mlii_channel == 0 else ["V5", "MLII"]
    p = np.stack(chans, axis=1)
    wfdb.wrsamp(name, fs=FS, units=["mV", "mV"], sig_name=names, p_signal=p,
                fmt=["16", "16"], write_dir=str(out_dir))
    wfdb.wrann(name, "atr", np.array(samples), np.array(symbols), write_dir=str(out_dir))


def make_dataset(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # DS1 (train/val), DS2 (test) y un registro (114) con MLII en el canal 1
    for i, rec in enumerate(["101", "106", "108", "109", "112", "100", "103"]):
        make_record(rec, out_dir, seed=i)
    make_record("114", out_dir, seed=99, mlii_channel=1)
    make_record("102", out_dir, seed=55)       # con marcapasos: debe excluirse en dechazal
    return out_dir


if __name__ == "__main__":
    import sys
    make_dataset(Path(sys.argv[1] if len(sys.argv) > 1 else "data_sintetica"))
