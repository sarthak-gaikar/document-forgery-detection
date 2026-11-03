"""
scripts/benchmark.py
Benchmarking script to compare all trained models on the test set.

Usage:
  python scripts/benchmark.py --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import sys
import time
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.utils import load_config, resolve_device, get_logger
from src.data.dataloader import build_dataloaders
from src.metrics.segmentation import SegmentationMetrics
from src.metrics.classification import ClassificationMetrics

logger = get_logger("benchmark")


def evaluate_checkpoint(ckpt_path: Path, config_path: Path, test_loader, device):
    cfg = load_config(config_path)
    if hasattr(cfg, "model") and hasattr(cfg.model, "rgb_encoder_weights"):
        cfg.model.rgb_encoder_weights = "none"
    if hasattr(cfg, "model") and hasattr(cfg.model, "encoder_weights"):
        cfg.model.encoder_weights = "none"

    from scripts.train import build_model
    model = build_model(cfg)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    seg_metrics = SegmentationMetrics(threshold=0.5)
    cls_metrics = ClassificationMetrics(threshold=0.5)

    t0 = time.time()
    n_images = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            n_images += images.shape[0]
            try:
                if amp_enabled:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                model.to("cpu")
                outputs = model(images.to("cpu"))

            if "seg_logits" in outputs:
                seg_metrics.update(outputs["seg_logits"], batch["mask"].to(device))
            if "cls_logits" in outputs and "label" in batch:
                cls_metrics.update(outputs["cls_logits"], batch["label"].to(device))

    elapsed = time.time() - t0
    fps = n_images / max(elapsed, 1e-4)

    seg_res = seg_metrics.compute() if seg_metrics._n > 0 else {}
    cls_res = cls_metrics.compute() if len(cls_metrics._labels) > 0 else {}

    return {
        "model_name": getattr(cfg.model, "name", ckpt_path.parent.name),
        "arch": getattr(cfg.model, "arch", getattr(cfg.model, "type", "segmentation")),
        "dice": seg_res.get("dice", 0.0),
        "iou": seg_res.get("iou", 0.0),
        "precision": seg_res.get("precision", 0.0),
        "recall": seg_res.get("recall", 0.0),
        "cls_acc": cls_res.get("cls_acc", 0.0),
        "cls_f1": cls_res.get("cls_f1", 0.0),
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained forgery models")
    parser.add_argument("--data-dir", default=None, help="Dataset directory (defaults to RealText-V2)")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--output", default="outputs/results")
    args = parser.parse_args()

    device = resolve_device("auto")
    if args.data_dir:
        data_dir = ROOT / args.data_dir
    elif (ROOT / "data/realtext_v2/manifest.json").exists():
        data_dir = ROOT / "data/realtext_v2"
    else:
        data_dir = ROOT / "data/synthetic"

    ckpt_root = ROOT / args.checkpoints_dir
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "manifest.json").exists():
        logger.error(f"Dataset not found at {data_dir}. Run scripts/download_realtext.py first.")
        sys.exit(1)

    _, _, test_loader = build_dataloaders(data_dir=data_dir, batch_size=1, num_workers=0)
    logger.info(f"Loaded test set with {len(test_loader.dataset)} samples from {data_dir}.")

    # Find model checkpoints (including subdirs like realtext/)
    checkpoints = list(ckpt_root.rglob("best_model.pth"))
    if not checkpoints:
        logger.warning(f"No trained checkpoints found in {ckpt_root}. Train models first using scripts/train.py")
        sys.exit(0)

    results = []
    config_candidates = {
        "dual_branch_forgery_net": ["configs/realtext_dual_branch.yaml", "configs/dual_branch.yaml"],
        "unet_resnet34": ["configs/realtext_unet.yaml", "configs/unet_resnet34.yaml"],
        "deeplabv3plus_efficientnetb3": ["configs/realtext_deeplabv3plus.yaml", "configs/deeplabv3plus_efficientnet.yaml"],
        "segformer_b2": ["configs/segformer_b2.yaml"],
    }

    for ckpt in checkpoints:
        model_name = ckpt.parent.name
        # Find matching config
        cand_list = config_candidates.get(model_name, ["configs/realtext_dual_branch.yaml", "configs/dual_branch.yaml"])
        cfg_path = None
        for cand in cand_list:
            if (ROOT / cand).exists():
                cfg_path = ROOT / cand
                break
        if cfg_path is None or not cfg_path.exists():
            continue

        label = f"{ckpt.parent.parent.name}/{model_name}" if ckpt.parent.parent.name in ("realtext", "checkpoints") else model_name
        logger.info(f"Evaluating {label} ({ckpt})...")
        res = evaluate_checkpoint(ckpt, cfg_path, test_loader, device)
        res["model_name"] = label
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = out_dir / "benchmark_comparison.csv"
    md_path = out_dir / "benchmark_comparison.md"

    df.to_csv(csv_path, index=False)
    try:
        md_table = df.to_markdown(index=False)
    except Exception:
        # Fallback without tabulate
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]
        md_table = "\n".join([header, sep] + rows)

    with open(md_path, "w") as f:
        f.write("# Model Benchmark Comparison\n\n")
        f.write(md_table)

    print("\n" + "=" * 60)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved results to:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
