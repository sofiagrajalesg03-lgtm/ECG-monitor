# inferencia.py
"""
Paso 4: usa el modelo entrenado para clasificar latidos de una señal ECG.

Dos modos de entrada:
  --record data/mitdb/100          registro MIT-BIH (usa las anotaciones .atr como picos R
                                   y, además, reporta qué tan bien coinciden las predicciones)
  --signal mi_ecg.csv --fs 250     señal propia (1 columna, o la columna --col). Los picos R
                                   se detectan automáticamente y la señal se re-muestrea a 360 Hz.

Uso rápido:
  python inferencia.py --model output/modelo_arritmias.pt --record data/mitdb/100
  python inferencia.py --model output/modelo_arritmias.pt --signal ecg.csv --fs 250
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks, resample_poly
from math import gcd

from modelo_cnn import ArrhythmiaNet, DEVICE
from segmentacion_ecg import bandpass, cut_windows, pick_lead, TARGET_FS


def load_model(path: Path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt["config"]
    model = ArrhythmiaNet(cfg["num_classes"], cfg["base_ch"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["label_names"]


def detect_r_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    Detector simple de picos R (energía de la derivada + umbral adaptativo).
    Suficiente para una demo; para uso clínico habría que validar un detector robusto
    (Pan-Tompkins, neurokit2) contra anotaciones.
    """
    d = np.diff(sig, prepend=sig[0]) ** 2
    win = max(1, int(0.08 * fs))
    env = np.convolve(d, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(env, height=0.3 * np.percentile(env, 99.5),
                          distance=int(0.25 * fs))
    # refinar al máximo absoluto local de la señal filtrada
    half = int(0.05 * fs)
    refined = []
    for p in peaks:
        a, b = max(0, p - half), min(len(sig), p + half)
        refined.append(a + int(np.argmax(np.abs(sig[a:b]))))
    return np.unique(refined)


def resample_to_target(sig: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - TARGET_FS) < 1e-6:
        return sig
    g = gcd(int(round(fs)), TARGET_FS)
    return resample_poly(sig, TARGET_FS // g, int(round(fs)) // g)


@torch.no_grad()
def classify(model, sig_360: np.ndarray, r_samples: np.ndarray):
    """sig_360: señal YA filtrada a 360 Hz. Devuelve (r_validos, clase_idx, probabilidades)."""
    wins, valid = cut_windows(sig_360, r_samples)
    if len(wins) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.zeros((0, 5))
    x = torch.from_numpy(wins[:, None, :]).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()
    return np.asarray(r_samples)[valid], probs.argmax(1), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", help="Ruta MIT-BIH sin extensión, p.ej. data/mitdb/100")
    src.add_argument("--signal", help="CSV/NPY con la señal ECG")
    ap.add_argument("--fs", type=float, help="Frecuencia de muestreo de --signal")
    ap.add_argument("--col", type=int, default=0, help="Columna del CSV con el ECG")
    args = ap.parse_args()

    model, names = load_model(Path(args.model))

    if args.record:
        import wfdb
        r = wfdb.rdrecord(args.record)
        ann = wfdb.rdann(args.record, "atr")
        ch = max(pick_lead(r, "MLII"), 0)
        sig = bandpass(r.p_signal[:, ch].astype(np.float64), float(r.fs))
        from ingesta_mitdb import AAMI_MAP
        from segmentacion_ecg import LABEL2IDX
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI_MAP]
        r_s = ann.sample[keep]
        truth = np.array([LABEL2IDX[AAMI_MAP[ann.symbol[i]]] for i in keep])
        r_ok, pred, probs = classify(model, sig, r_s)
        valid = np.isin(r_s, r_ok)
        truth = truth[valid]
        print(f"Registro {args.record}: {len(pred)} latidos clasificados")
        print(f"Coincidencia con anotaciones: {100 * (pred == truth).mean():.1f}% "
              "(accuracy bruta; para evaluar usa los reportes de modelo_cnn.py)")
    else:
        if args.fs is None:
            raise SystemExit("--fs es obligatorio con --signal")
        p = Path(args.signal)
        raw = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, delimiter=",", ndmin=2)
        raw = raw if raw.ndim == 1 else raw[:, args.col]
        sig = bandpass(resample_to_target(raw.astype(np.float64), args.fs), TARGET_FS)
        r_s = detect_r_peaks(sig, TARGET_FS)
        r_ok, pred, probs = classify(model, sig, r_s)
        print(f"{len(r_s)} picos R detectados, {len(pred)} latidos clasificados")

    counts = np.bincount(pred, minlength=len(names))
    print("\nLatidos por clase predicha:")
    for n, c in zip(names, counts):
        print(f"  {n:24s} {c:6d}  ({100 * c / max(len(pred), 1):.1f}%)")
    print("\nAviso: proyecto académico, no es un dispositivo médico ni apto para diagnóstico.")


if __name__ == "__main__":
    main()
