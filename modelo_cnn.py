# modelo_cnn.py
"""
Paso 3: CNN 1D con bloques residuales para clasificación de arritmias (5 clases AAMI).
Entrada : windows.npz + splits.json
Salida  : modelo_arritmias.pt, metricas_test.json, matriz_confusion.png
"""

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABEL_NAMES = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusión)", "Q (Otro)"]
NUM_CLASSES = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Arquitectura ─────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Bloque residual 1D con BatchNorm y Dropout."""
    def __init__(self, channels: int, kernel: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class ArrhythmiaNet(nn.Module):
    """CNN 1D con bloques residuales para ECG de entrada (1, 216)."""
    def __init__(self, num_classes: int = NUM_CLASSES,
                 base_ch: int = 32, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),          # → (32, 108)
        )
        self.block1 = ResBlock1D(base_ch, kernel=7, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=1),
            nn.MaxPool1d(2),          # → (64, 54)
        )
        self.block2 = ResBlock1D(base_ch * 2, kernel=5, dropout=dropout)
        self.down2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=1),
            nn.MaxPool1d(2),          # → (128, 27)
        )
        self.block3 = ResBlock1D(base_ch * 4, kernel=3, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.head(x)


# ── Dataset helpers ──────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(npz_path: Path, splits_json: Path, split: str):
    """Carga X, y filtrados por pacientes del split indicado."""
    data = np.load(npz_path)
    X, y, groups = data["X"], data["y"], data["groups"]
    splits = json.loads(splits_json.read_text())
    mask = np.isin(groups, splits[split])
    return X[mask].astype(np.float32), y[mask]


def class_inverse_freq(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    return 1.0 / np.where(counts == 0, 1, counts)


def make_loader(X, y, batch_size: int, balance: bool = False, shuffle: bool = True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    sampler = None
    if balance:
        sample_weights = torch.from_numpy(class_inverse_freq(y)[y]).float()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, shuffle=shuffle, num_workers=0)


# ── Entrenamiento / evaluación ───────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct += (logits.argmax(1) == y_b).sum().item()
        n += len(y_b)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for X_b, y_b in loader:
        logits = model(X_b.to(device))
        loss_sum += ce(logits, y_b.to(device)).item()
        n += len(y_b)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(trues), loss_sum / n


def macro_f1(y_true, y_pred) -> float:
    """F1 macro sobre las clases presentes en y_true (accuracy engaña con N >> resto)."""
    return f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro",
                    zero_division=0)


def save_confusion_png(cm, names, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    short = [n.split()[0] for n in names]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(short)), short); ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ── Main ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows.npz de segmentacion_ecg.py")
    ap.add_argument("--splits",  required=True, help="splits.json de ingesta_mitdb.py")
    ap.add_argument("--outdir",  required=True, help="Carpeta de salida para el modelo")
    ap.add_argument("--epochs",  type=int,   default=30)
    ap.add_argument("--batch",   type=int,   default=128)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base_ch", type=int,   default=32)
    ap.add_argument("--seed",    type=int,   default=7)
    ap.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler",
                    help="Cómo tratar el desbalance. Usar sampler Y pesos en la loss a la vez "
                         "sobre-compensa, por eso son excluyentes.")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    wpz, sjson = Path(args.windows), Path(args.splits)

    # — datos —
    X_tr, y_tr = load_split(wpz, sjson, "train")
    X_va, y_va = load_split(wpz, sjson, "val")
    X_te, y_te = load_split(wpz, sjson, "test")
    print(f"Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}")
    for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        print(f"  {name:5s} por clase [N,S,V,F,Q]: {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    tr_loader = make_loader(X_tr, y_tr, args.batch, balance=(args.balance == "sampler"))
    va_loader = make_loader(X_va, y_va, args.batch, shuffle=False)
    te_loader = make_loader(X_te, y_te, args.batch, shuffle=False)

    # — modelo —
    model = ArrhythmiaNet(NUM_CLASSES, args.base_ch, args.dropout).to(DEVICE)
    weight = None
    if args.balance == "loss":
        w = class_inverse_freq(y_tr)
        weight = torch.tensor(w / w.sum() * NUM_CLASSES, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nEntrenando en {DEVICE}  |  Épocas: {args.epochs}  |  balance: {args.balance}\n")
    best_f1, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        p_va, t_va, va_loss = predict(model, va_loader, DEVICE)
        va_acc, va_f1 = float((p_va == t_va).mean()), macro_f1(t_va, p_va)
        scheduler.step()

        flag = ""
        if va_f1 > best_f1:     # se elige por F1 macro, no por accuracy
            best_f1, best_epoch = va_f1, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = "  ← mejor"

        if epoch % 5 == 0 or epoch == 1 or flag:
            print(f"Época {epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}  val_f1={va_f1:.3f}{flag}")

    # — evaluación final en test —
    model.load_state_dict(best_state)
    all_preds, all_true, _ = predict(model, te_loader, DEVICE)
    present = sorted(set(all_true.tolist()))
    p_names = [LABEL_NAMES[i] for i in present]

    print("\n── Reporte de clasificación (Test) ──────────────────")
    report_txt = classification_report(all_true, all_preds, labels=present,
                                       target_names=p_names, digits=3, zero_division=0)
    print(report_txt)

    print("── Matriz de confusión ──────────────────────────────")
    cm = confusion_matrix(all_true, all_preds, labels=present)
    short = [n.split()[0] for n in p_names]
    print("      " + "".join(f"{s:>7}" for s in short) + "   (filas = real, columnas = predicho)")
    for i, row in enumerate(cm):
        print(f"  {short[i]:>3} " + "".join(f"{v:>7}" for v in row))
    save_confusion_png(cm, p_names, outdir / "matriz_confusion.png")

    # — guardar —
    report = classification_report(all_true, all_preds, labels=present, target_names=p_names,
                                   digits=4, zero_division=0, output_dict=True)
    (outdir / "metricas_test.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "test_accuracy": float((all_preds == all_true).mean()),
        "report": report, "confusion_matrix": cm.tolist(), "classes": p_names,
        "args": vars(args),
    }, indent=2, ensure_ascii=False))

    ckpt_path = outdir / "modelo_arritmias.pt"
    torch.save({
        "model_state": best_state,
        "config": {"num_classes": NUM_CLASSES, "base_ch": args.base_ch,
                   "dropout": args.dropout, "window_len": 216, "target_fs": 360},
        "label_names": LABEL_NAMES,
    }, ckpt_path)
    print(f"\n✔ Modelo guardado en {ckpt_path}")
    print(f"✔ Mejor val F1 macro: {best_f1:.3f} (época {best_epoch})")
    print(f"✔ Métricas: {outdir / 'metricas_test.json'}")


if __name__ == "__main__":
    main()
