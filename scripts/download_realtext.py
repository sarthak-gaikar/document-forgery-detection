"""
scripts/download_realtext.py
Download and index the RealText-V2 document forgery dataset from Hugging Face.

Usage:
  # Download full RealText-V2 dataset and build manifest:
  python scripts/download_realtext.py --output-dir data/realtext_v2

  # Download the synthetic extension (RealText-V2-Syn25k):
  python scripts/download_realtext.py --repo-id vankey/RealText-V2-Syn25k --output-dir data/realtext_v2

  # Re-index an already downloaded folder:
  python scripts/download_realtext.py --output-dir data/realtext_v2 --build-manifest-only

  # Create a mock RealText-V2 dataset for offline testing/verification:
  python scripts/download_realtext.py --output-dir data/realtext_v2 --mock
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger("download_realtext")


def download_hf_dataset(repo_id: str, output_dir: Path, max_samples: int | None = None) -> bool:
    """Download RealText-V2 from Hugging Face Hub using snapshot_download or datasets."""
    logger.info(f"Downloading {repo_id} to {output_dir}...")
    ensure_dir(output_dir)

    try:
        from huggingface_hub import snapshot_download
        logger.info("Using huggingface_hub snapshot_download...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"Successfully downloaded files to {output_dir}")
        return True
    except Exception as e:
        logger.warning(f"snapshot_download failed or unavailable: {e}")

    try:
        import datasets
        logger.info("Attempting download via HuggingFace datasets library...")
        ds = datasets.load_dataset(repo_id)
        logger.info(f"Loaded dataset: {ds}")
        return True
    except Exception as e2:
        logger.error(f"Failed to download dataset: {e2}")
        return False


def build_realtext_manifest(
    data_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Scan data_dir for document images and corresponding masks,
    then generate data/realtext_v2/manifest.json.
    """
    logger.info(f"Indexing RealText-V2 images in {data_dir}...")

    # Look for common image formats
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    # Separate images from masks
    images = []
    masks_dict = {}  # key -> Path

    for p in all_files:
        rel = p.relative_to(data_dir)
        parts = [part.lower() for part in rel.parts]
        stem = p.stem.lower()

        is_mask = "mask" in parts or stem.endswith("_mask") or "masks" in parts
        if is_mask:
            # Clean key for pairing
            clean_key = stem.replace("_mask", "").replace("mask_", "")
            masks_dict[clean_key] = rel
        else:
            images.append(p)

    if not images:
        logger.warning(f"No document images found in {data_dir} to index.")
        return data_dir / "manifest.json"

    logger.info(f"Found {len(images)} document images and {len(masks_dict)} mask files.")

    # Pair images with masks
    random.seed(seed)
    random.shuffle(images)

    samples = []
    n_forged = 0
    n_authentic = 0

    for idx, img_path in enumerate(images):
        rel_img = img_path.relative_to(data_dir)
        stem = img_path.stem.lower()
        clean_key = stem.replace("_mask", "").replace("mask_", "")

        mask_rel = masks_dict.get(clean_key)
        if mask_rel is None:
            # Check if mask exists with same relative path under 'mask' folder instead of 'image'
            alt_parts = list(rel_img.parts)
            for i, part in enumerate(alt_parts):
                if part.lower() == "image" or part.lower() == "images":
                    alt_parts[i] = "mask"
                    break
            alt_path = data_dir.joinpath(*alt_parts)
            if alt_path.exists():
                mask_rel = alt_path.relative_to(data_dir)

        is_forged = mask_rel is not None
        if is_forged:
            n_forged += 1
        else:
            n_authentic += 1

        samples.append({
            "sample_id": f"realtext_{idx:06d}",
            "image_path": str(rel_img).replace("\\", "/"),
            "mask_path": str(mask_rel).replace("\\", "/") if mask_rel else None,
            "is_forged": is_forged,
            "doc_type": "document",
            "tamper_type": "text_manipulation" if is_forged else "authentic",
            "split": "train",  # assigned below
        })

    # Partition splits
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    for i, s in enumerate(samples):
        if i < n_train:
            s["split"] = "train"
        elif i < n_train + n_val:
            s["split"] = "val"
        else:
            s["split"] = "test"

    manifest = {
        "dataset_name": "RealText-V2",
        "n_samples": n_total,
        "n_authentic": n_authentic,
        "n_forged": n_forged,
        "splits": {
            "train": sum(1 for s in samples if s["split"] == "train"),
            "val": sum(1 for s in samples if s["split"] == "val"),
            "test": sum(1 for s in samples if s["split"] == "test"),
        },
        "samples": samples,
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated {manifest_path} with {n_total} samples "
                f"({n_forged} forged, {n_authentic} authentic).")
    logger.info(f"Splits -> Train: {manifest['splits']['train']} | "
                f"Val: {manifest['splits']['val']} | Test: {manifest['splits']['test']}")
    return manifest_path


def create_mock_realtext_dataset(output_dir: Path, n_samples: int = 100) -> Path:
    """Create a mock RealText-V2 dataset structure for offline testing and verification."""
    import cv2
    import numpy as np

    logger.info(f"Generating mock RealText-V2 dataset ({n_samples} samples) at {output_dir}...")
    img_dir = output_dir / "train" / "image" / "part000"
    mask_dir = output_dir / "train" / "mask" / "part000"
    ensure_dir(img_dir)
    ensure_dir(mask_dir)

    for i in range(n_samples):
        # Create a document-like image
        h, w = 700, 500
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 248
        # Add text lines
        for y in range(80, h - 80, 45):
            cv2.line(canvas, (40, y), (w - 40, y), (210, 210, 210), 1)
            cv2.putText(canvas, f"Official Document Line {y//45} Record No {1000+i}", (50, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

        is_forged = (i % 2 == 1)
        img_name = f"RealText_doc_{i:04d}.jpg"
        mask_name = f"RealText_doc_{i:04d}.png"

        if is_forged:
            # Tamper a region
            mask = np.zeros((h, w), dtype=np.uint8)
            tx, ty, tw, th = 150, 200 + (i % 5) * 50, 180, 35
            # Draw altered text on canvas
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (255, 255, 240), -1)
            cv2.putText(canvas, "MODIFIED VALUE $9999", (tx + 5, ty + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            mask[ty:ty+th, tx:tx+tw] = 255
            cv2.imwrite(str(mask_dir / mask_name), mask)

        cv2.imwrite(str(img_dir / img_name), canvas)

    # Build manifest
    return build_realtext_manifest(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and index RealText-V2 dataset")
    parser.add_argument("--repo-id", default="vankey/RealText-V2", help="Hugging Face Dataset repo ID")
    parser.add_argument("--output-dir", default="data/realtext_v2", help="Target output directory")
    parser.add_argument("--build-manifest-only", action="store_true", help="Skip download, just index existing files")
    parser.add_argument("--mock", action="store_true", help="Generate mock RealText-V2 dataset for offline testing")
    parser.add_argument("--n-mock", type=int, default=100, help="Number of mock samples to generate")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir

    if args.mock:
        create_mock_realtext_dataset(out_dir, n_samples=args.n_mock)
        print("\n[OK] Mock RealText-V2 dataset generated successfully.")
        return

    if not args.build_manifest_only:
        success = download_hf_dataset(args.repo_id, out_dir)
        if not success:
            logger.warning("Download failed. If offline, use --mock or run download outside sandbox.")
            sys.exit(1)

    build_realtext_manifest(out_dir)
    print(f"\n[OK] RealText-V2 indexed successfully at {out_dir}")


if __name__ == "__main__":
    main()

