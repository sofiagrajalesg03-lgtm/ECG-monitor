# segmentacion_ecg.py
"""
Paso 2: segmenta ventanas de ECG centradas en cada pico R.
Entrada : beats.parquet + carpeta con .hea/.dat/.atr (MIT-BIH)
Salida  : windows.npz  -> X (N, 1, L), y (N,), groups (N,)

Las funciones bandpass() y normalize() se reutilizan en inferencia.py para que el
preprocesamiento en entrenamiento y en producción sea idéntico.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

# ── Parámetros de ventana ────────────────────────────────
WINDOW_SEC = 0.6                            # segundos totales por latido (±0.3 s del pico R)
TARGET_FS  = 360                            # fs nativa de MIT-BIH
WINDOW_LEN = int(TARGET_FS * WINDOW_SEC)    # 216 muestras
PRE        = WINDOW_LEN // 2                # muestras antes del pico R

LABEL2IDX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}


# ── Filtrado y normalización ─────────────────────────────

def bandpass(signal: np.ndarray, fs: float,
             low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Butterworth bandpass 0.5-40 Hz (cero-fase) para quitar baseline y ruido."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def normalize(windows: np.ndarray) -> np.ndarray:
    """Z-score por ventana. Acepta (L,) o (N, L)."""
    w = np.atleast_2d(windows).astype(np.float32)
    mean = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True)
    out = (w - mean) / np.where(std < 1e-6, 1.0, std)
    return out[0] if windows.ndim == 1 else out


def cut_windows(sig: np.ndarray, r_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta ventanas de WINDOW_LEN muestras centradas en cada r_sample.
    Retorna (ventanas normalizadas (M, L), máscara booleana de los r_samples válidos).
    """
    r_samples = np.asarray(r_samples, dtype=int)
    start = r_samples - PRE
    valid = (start >= 0) & (start + WINDOW_LEN <= len(sig))
    idx = start[valid][:, None] + np.arange(WINDOW_LEN)[None, :]
    return normalize(sig[idx]), valid


# ── Segmentación ─────────────────────────────────────────

def pick_lead(record, lead) -> int:
    """
    'MLII' (default) busca el canal por nombre: en MIT-BIH el canal 0 NO siempre es MLII
    (p.ej. el registro 114 lo trae en el canal 1). Un entero fuerza ese índice.
    """
    if isinstance(lead, int):
        return lead
    names = [n.strip().upper() for n in record.sig_name]
    return names.index(lead.upper()) if lead.upper() in names else -1


def extract_windows(beats: pd.DataFrame, records_dir: Path, lead="MLII"):
    """
    Retorna:
        X      : (N, 1, WINDOW_LEN) float32
        y      : (N,)               int64
        groups : (N,)               str  — id de paciente para splits
    """
    X_list, y_list, g_list = [], [], []

    for rec, grp in beats.groupby("record"):
        r = wfdb.rdrecord(str(records_dir / rec))
        if abs(float(r.fs) - TARGET_FS) > 1e-6:
            print(f"  ! {rec}: fs={r.fs} != {TARGET_FS}, se omite")
            continue
        ch = pick_lead(r, lead)
        if ch < 0:
            print(f"  ! {rec}: no tiene canal {lead} (canales: {r.sig_name}), se omite")
            continue

        sig = bandpass(r.p_signal[:, ch].astype(np.float64), float(r.fs)).astype(np.float32)
        wins, valid = cut_windows(sig, grp["r_sample"].to_numpy())
        if len(wins) == 0:
            continue
        X_list.append(wins)
        y_list.append(grp["label_aami"].map(LABEL2IDX).to_numpy()[valid])
        g_list.append(np.full(len(wins), str(rec)))

    if not X_list:
        raise SystemExit("No se extrajo ninguna ventana: revisa --records y --lead.")
    X = np.concatenate(X_list)[:, np.newaxis, :]
    y = np.concatenate(y_list).astype(np.int64)
    g = np.concatenate(g_list)
    return X, y, g


# ── Main ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Carpeta MIT-BIH (.hea/.dat/.atr)")
    ap.add_argument("--beats",   required=True, help="beats.parquet de ingesta_mitdb.py")
    ap.add_argument("--outdir",  required=True, help="Carpeta de salida")
    ap.add_argument("--lead",    default="MLII",
                    help="Nombre del canal (MLII) o índice entero")
    args = ap.parse_args()

    lead = int(args.lead) if args.lead.lstrip("-").isdigit() else args.lead
    records_dir = Path(args.records)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    beats = pd.read_parquet(args.beats)
    print(f"Latidos cargados: {len(beats)}")

    X, y, groups = extract_windows(beats, records_dir, lead=lead)
    print(f"Ventanas extraídas: {X.shape}  |  clases: {np.unique(y, return_counts=True)}")

    np.savez_compressed(outdir / "windows.npz", X=X, y=y, groups=groups)
    print(f"✔ Guardado en {outdir / 'windows.npz'}")

    labels = ["N", "S", "V", "F", "Q"]
    print("\nDistribución de clases:")
    for i, lbl in enumerate(labels):
        n = int((y == i).sum())
        if n > 0:
            print(f"  {lbl}: {n} ({100 * n / len(y):.1f}%)")


if __name__ == "__main__":
    main()
