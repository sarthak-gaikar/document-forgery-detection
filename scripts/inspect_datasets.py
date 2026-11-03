"""
scripts/inspect_datasets.py
Phase 2: Dataset inspection - shows stats about the generated synthetic dataset.
Usage: python scripts/inspect_datasets.py --data-dir data/synthetic
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a generated synthetic forgery dataset."
    )
    parser.add_argument("--data-dir", default="data/synthetic",
                        help="Path to dataset directory (relative to project root)")
    args = parser.parse_args()

    logger = get_logger("inspect_datasets")

    data_dir = ROOT / args.data_dir
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(
            f"manifest.json not found in {data_dir}. "
            "Run generate_synthetic.py first."
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    logger.info(f"Total samples: {len(samples)}")

    # Per-split counts
    split_counts = Counter(s["split"] for s in samples)
    for split, count in sorted(split_counts.items()):
        logger.info(f"  {split}: {count}")

    # Label distribution
    label_counts = Counter(s["is_forged"] for s in samples)
    logger.info(
        f"  Authentic: {label_counts[False]}, Forged: {label_counts[True]}"
    )

    # Tamper type distribution
    tamper_counts = Counter(s.get("tamper_type", "none") for s in samples)
    logger.info("Tamper type distribution:")
    for t, c in tamper_counts.most_common():
        logger.info(f"  tamper_type={t}: {c}")

    # Doc type distribution
    doc_type_counts = Counter(s["doc_type"] for s in samples)
    logger.info("Document type distribution:")
    for t, c in doc_type_counts.most_common():
        logger.info(f"  doc_type={t}: {c}")

    logger.info("Inspection complete.")


if __name__ == "__main__":
    main()
