"""
scripts/generate_synthetic.py
Phase 3: Synthetic dataset generation entry point.
Usage:
  python scripts/generate_synthetic.py --config configs/synthetic_gen.yaml --n-samples 1000 --output data/synthetic
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, get_logger
from src.data.synthetic_gen import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic forgery dataset")
    parser.add_argument("--config", default="configs/synthetic_gen.yaml",
                        help="Path to synthetic_gen.yaml (relative to project root)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Total samples to generate")
    parser.add_argument("--output", default="data/synthetic",
                        help="Output directory (relative to project root)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    logger = get_logger("generate_synthetic")
    cfg = load_config(ROOT / args.config)

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.n_samples} samples -> {output_dir}")
    generate_dataset(
        output_dir=output_dir,
        n_samples=args.n_samples,
        config=cfg,
        seed=args.seed,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
