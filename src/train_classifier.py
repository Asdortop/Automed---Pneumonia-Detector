"""
train_classifier.py — Phase 2
Full training loop for the XRayClassifier.

Usage:
    python src/train_classifier.py               # full training
    python src/train_classifier.py --epochs 2    # quick test
    python src/train_classifier.py --debug       # 1 epoch, small batch
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers/Colab)
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from classifier import build_model
from dataset    import get_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def compute_class_weight(dataset) -> torch.Tensor:
    """
    Returns BCEWithLogitsLoss pos_weight tensor to handle class imbalance.
    pos_weight = count(NORMAL) / count(PNEUMONIA)
    """
    counts = dataset.class_counts()
    w = counts["NORMAL"] / counts["PNEUMONIA"]
    return torch.tensor([w], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, list, list]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    avg_loss = running_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def plot_history(train_losses, val_losses, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, "g-o", label="Val Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  📊  Training curves saved → {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  AutoMed — Phase 2: ResNet-50 Classifier Training")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    loaders  = get_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
    train_ds = loaders["train"].dataset

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(device)

    # ── Loss: BCEWithLogitsLoss with pos_weight for class imbalance ──────
    pos_weight = compute_class_weight(train_ds).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\n  pos_weight (PNEUMONIA): {pos_weight.item():.4f}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>9} {'Time':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_accs.append(vl_acc)

        print(f"{epoch:>6} {tr_loss:>12.4f} {vl_loss:>10.4f} "
              f"{vl_acc*100:>8.2f}% {elapsed:>6.1f}s")

        # ── Save best model ───────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   vl_loss,
                "val_acc":    vl_acc,
            }, MODELS_DIR / "classifier.pth")
            print(f"  💾  Saved best model (val_loss={vl_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Final Evaluation on Test Set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(MODELS_DIR / "classifier.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_preds, test_labels = evaluate(
        model, loaders["test"], criterion, device
    )
    f1 = f1_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  F1 Score      : {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(f"  {'':12s}  Pred NORMAL  Pred PNEUMONIA")
    print(f"  True NORMAL   {cm[0,0]:^11d}  {cm[0,1]:^13d}")
    print(f"  True PNEUMONIA{cm[1,0]:^11d}  {cm[1,1]:^13d}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['NORMAL','PNEUMONIA'])}")

    # ── Save training curves ──────────────────────────────────────────────
    plot_history(train_losses, val_losses, val_accs,
                 OUTPUTS_DIR / "training_curves.png")

    print("\n✅  Training complete!")
    print(f"  Best model saved → {MODELS_DIR / 'classifier.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMed Classifier Training")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--debug",      action="store_true",
                        help="Run 1 epoch with batch_size=8 for quick test")
    args = parser.parse_args()

    if args.debug:
        args.epochs     = 1
        args.batch_size = 8
        args.workers    = 0
        print("⚡  Debug mode: 1 epoch, batch=8")

    main(args)
