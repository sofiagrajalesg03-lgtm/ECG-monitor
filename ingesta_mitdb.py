# ingesta_mitdb.py
"""
Paso 1: extrae todos los latidos anotados de MIT-BIH y crea los splits por paciente.

Salida: beats.parquet  (record, patient, fs, r_sample, label_sym, label_aami)
        splits.json    ({"train": [...], "val": [...], "test": [...]})

Modos de split:
  dechazal (default) : protocolo inter-paciente estándar de De Chazal et al. (2004).
                       DS1 -> train/val, DS2 -> test. Excluye los 4 registros con
                       marcapasos (102, 104, 107, 217). Es el que permite comparar
                       tus resultados con la literatura.
  random             : mezcla aleatoria de pacientes (15 % val, 15 % test). Con solo
                       48 registros, el test puede quedar sin clases F o S.
"""
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
import wfdb

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # Normal/BBB/escapes
    "A": "S", "a": "S", "J": "S", "S": "S",            # Supraventricular
    "V": "V", "E": "V",                                # Ventricular
    "F": "F",                                          # Fusión
    "/": "Q", "f": "Q", "Q": "Q",                      # Otros/paced
}

# Protocolo De Chazal (inter-paciente)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
PACED = ["102", "104", "107", "217"]


def build_beats(records_dir: Path, exclude: tuple = ()) -> pd.DataFrame:
    rows = []
    for hea in sorted(records_dir.glob("*.hea")):
        rec = hea.stem
        if rec in exclude:
            continue
        base = str(hea.with_suffix(""))
        if not hea.with_suffix(".atr").exists():
            print(f"  ! {rec}: sin archivo .atr, se omite")
            continue
        r = wfdb.rdheader(base)             # solo header: no hace falta leer la señal aquí
        fs = float(r.fs)
        ann = wfdb.rdann(base, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            aami = AAMI_MAP.get(sym)
            if aami is None:                # anotaciones que no son latidos ('+', '~', '|', ...)
                continue
            rows.append({
                "record": rec,
                "patient": rec,             # en MIT-BIH cada record es un sujeto
                "fs": fs,
                "r_sample": int(s),
                "label_sym": sym,
                "label_aami": aami,
            })
    return pd.DataFrame(rows)


def write_splits(beats: pd.DataFrame, out_json: Path, mode: str = "dechazal",
                 seed: int = 7, val_frac: float = 0.2):
    patients = beats["patient"].drop_duplicates().to_list()
    rng = np.random.default_rng(seed)

    if mode == "dechazal":
        test = [p for p in patients if p in DS2]
        ds1 = [p for p in patients if p in DS1]
        rng.shuffle(ds1)
        n_val = max(1, int(round(val_frac * len(ds1)))) if ds1 else 0
        val, train = ds1[:n_val], ds1[n_val:]
        unused = [p for p in patients if p not in DS1 and p not in DS2]
        if unused:
            print(f"  ! registros fuera de DS1/DS2 (no usados): {unused}")
        if not (train and val and test):
            raise SystemExit("El modo dechazal necesita los registros estándar de MIT-BIH "
                             "(100-124, 200-234). Usa --split_mode random si tienes menos.")
    else:
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, int(round(0.15 * n)))
        n_val = max(1, int(round(0.15 * n)))
        test = patients[:n_test]
        val = patients[n_test:n_test + n_val]
        train = patients[n_test + n_val:]

    out_json.write_text(json.dumps({"train": sorted(train), "val": sorted(val),
                                    "test": sorted(test)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Carpeta con .hea/.dat/.atr")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida (p.ej. data/)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split_mode", choices=["dechazal", "random"], default="dechazal")
    args = ap.parse_args()

    records_dir = Path(args.records)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    exclude = tuple(PACED) if args.split_mode == "dechazal" else ()
    beats = build_beats(records_dir, exclude=exclude)
    if beats.empty:
        raise SystemExit(f"No se encontraron latidos en {records_dir}. "
                         "¿La carpeta contiene .hea/.dat/.atr?")
    beats.to_parquet(outdir / "beats.parquet", index=False)
    write_splits(beats, outdir / "splits.json", mode=args.split_mode, seed=args.seed)

    print("✔ beats:", outdir / "beats.parquet", f"({len(beats)} filas, "
          f"{beats['record'].nunique()} registros)")
    print("✔ splits:", outdir / "splits.json", f"(modo: {args.split_mode})")
    print(beats["label_aami"].value_counts())


if __name__ == "__main__":
    main()
