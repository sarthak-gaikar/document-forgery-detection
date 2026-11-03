"""
scripts/plot_curves.py
Generate training/validation learning curves from metrics CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csv
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_metrics(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    train = {int(r["epoch"]): r for r in rows if r["split"] == "train"}
    val   = {int(r["epoch"]): r for r in rows if r["split"] == "val"}
    return train, val


def plot_model(model_name: str, out_dir: Path):
    csv_path = ROOT / "outputs" / "logs" / model_name / "metrics.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return

    train, val = load_metrics(csv_path)
    epochs = sorted(set(list(train.keys()) + list(val.keys())))

    def get(d, k, field):
        return float(d[k][field]) if k in d else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Learning Curves", fontsize=13, fontweight="bold")

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, [get(train, e, "loss") for e in epochs], "b-o", ms=4, label="Train")
    ax.plot(epochs, [get(val,   e, "loss") for e in epochs], "r-o", ms=4, label="Val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # --- Dice ---
    ax = axes[1]
    ax.plot(epochs, [get(train, e, "dice") for e in epochs], "b-o", ms=4, label="Train")
    ax.plot(epochs, [get(val,   e, "dice") for e in epochs], "r-o", ms=4, label="Val")
    ax.set_title("Dice Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # --- Classification Accuracy or Recall ---
    ax = axes[2]
    has_cls = any("cls_acc" in r for r in train.values())
    metric_key = "cls_acc" if has_cls else "recall"
    metric_label = "Classification Accuracy" if has_cls else "Recall"
    ax.plot(epochs, [get(train, e, metric_key) for e in epochs], "b-o", ms=4, label="Train")
    ax.plot(epochs, [get(val,   e, metric_key) for e in epochs], "r-o", ms=4, label="Val")
    ax.set_title(metric_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{model_name}_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    out_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in ["dual_branch_forgery_net", "unet_resnet34", "deeplabv3plus_efficientnetb3"]:
        print(f"Plotting {model}...")
        plot_model(model, out_dir)

    print("Done.")
