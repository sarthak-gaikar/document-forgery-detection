"""
scripts/infer.py
CLI inference entry point for document forgery detection.

Usage:
  python scripts/infer.py --input path/to/document.pdf --output outputs/results/
  python scripts/infer.py --input path/to/image.png --checkpoint outputs/checkpoints/dual_branch_forgery_net/best_model.pth --config configs/dual_branch.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, get_logger, resolve_device

logger = get_logger("infer")


def main():
    parser = argparse.ArgumentParser(description="Run forgery detection inference")
    parser.add_argument("--input", required=True, help="Path to input document (PDF or image)")
    parser.add_argument("--output", default="outputs/results", help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="configs/dual_branch.yaml", help="Config YAML")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    device = resolve_device(args.device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        model_name = getattr(cfg.model, "name", "model")
        checkpoint_path = ROOT / cfg.paths.checkpoint_dir / model_name / "best_model.pth"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}. Train a model first with scripts/train.py")
        sys.exit(1)

    from src.inference.pipeline import ForgeryDetector, load_model
    model = load_model(checkpoint_path, cfg, device=device)

    detector = ForgeryDetector(
        model=model,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )

    result = detector.detect(args.input, output_dir=ROOT / args.output)

    print(f"\n{'='*50}")
    print(f"Input:       {result['input_path']}")
    print(f"Is Forged:   {result['is_forged']}")
    print(f"Confidence:  {result['confidence']:.4f}")
    print(f"Regions:     {len(result['forgery_regions'])} detected")
    print(f"{'='*50}")
    print("Forensic Assistance Tool - NOT a Legal Verdict")


if __name__ == "__main__":
    main()
