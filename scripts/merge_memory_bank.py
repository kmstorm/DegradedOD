"""
Merge / Convert memory bank outputs for use with MemoryBank class.

Converts the raw output from build_memory_bank.py into the format
expected by src.models.memory_bank.MemoryBank.load().

Also supports merging multiple partial banks into one.

Usage:
    # Convert single bank
    python scripts/merge_memory_bank.py \
        --input-dirs outputs/memory_bank \
        --output-dir outputs/memory_bank_final

    # Merge multiple partial banks
    python scripts/merge_memory_bank.py \
        --input-dirs outputs/memory_bank_p1 outputs/memory_bank_p2 \
        --output-dir outputs/memory_bank_final
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_memory_bank")


def load_raw_bank(bank_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """
    Load features and metadata from build_memory_bank.py output.
    
    Supports two cases:
      1. Final output: bank_dir/features.npy + metadata.json
      2. Checkpoint: bank_dir/checkpoints/features_NNNNN.npy (latest)
    """
    # Case 1: final output
    if (bank_dir / "features.npy").exists():
        features = np.load(bank_dir / "features.npy")
        with open(bank_dir / "metadata.json") as f:
            data = json.load(f)
        metadata = data.get("metadata", data.get("entries", []))
        logger.info(f"Loaded {len(metadata)} entries from {bank_dir}/features.npy")
        return features, metadata

    # Case 2: load latest checkpoint
    ckpt_dir = bank_dir / "checkpoints"
    if ckpt_dir.exists():
        feat_files = sorted(ckpt_dir.glob("features_*.npy"))
        if feat_files:
            latest_feat = feat_files[-1]  # Highest index = latest
            ckpt_idx = latest_feat.stem.split("_")[1]
            meta_file = ckpt_dir / f"metadata_{ckpt_idx}.json"

            features = np.load(latest_feat)
            with open(meta_file) as f:
                data = json.load(f)
            metadata = data.get("metadata", [])
            logger.info(
                f"Loaded {len(metadata)} entries from checkpoint "
                f"{latest_feat.name} (build still in progress)"
            )
            return features, metadata

    raise FileNotFoundError(f"No features found in {bank_dir} or {bank_dir}/checkpoints/")


def merge_and_convert(
    input_dirs: list[Path],
    output_dir: Path,
    max_entries: int | None = None,
) -> None:
    """Merge multiple raw banks and convert to MemoryBank format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_metadata = []
    class_counts: dict[str, int] = {}

    for bank_dir in input_dirs:
        try:
            features, metadata = load_raw_bank(bank_dir)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {bank_dir}: {e}")
            continue
        all_features.append(features)
        all_metadata.extend(metadata)

        for entry in metadata:
            cls = entry.get("class", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if not all_features:
        logger.error("No features found in any input directory!")
        return

    merged_features = np.concatenate(all_features, axis=0).astype(np.float32)
    logger.info(f"Total merged: {merged_features.shape[0]} features, dim={merged_features.shape[1]}")

    if max_entries and merged_features.shape[0] > max_entries:
        logger.info(f"Truncating to {max_entries} entries")
        merged_features = merged_features[:max_entries]
        all_metadata = all_metadata[:max_entries]

    # Convert metadata to MemoryBank format
    entries = []
    for i, m in enumerate(all_metadata):
        entries.append({
            "class_name": m.get("class", "unknown"),
            "score": m.get("score", 0.0),
            "source_image": m.get("image_id", ""),
        })

    # Rebuild class counts from converted data
    class_counts = {}
    for e in entries:
        cls = e["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Save in MemoryBank.load() format
    meta_out = {
        "feature_dim": int(merged_features.shape[1]),
        "max_entries_per_class": 10000,
        "max_total_entries": merged_features.shape[0],
        "similarity": "cosine",
        "entries": entries,
        "class_counts": class_counts,
        "num_entries": len(entries),
        "num_classes": len(class_counts),
    }

    np.save(output_dir / "features.npy", merged_features)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Build FAISS index
    try:
        import faiss
        faiss.normalize_L2(merged_features)
        index = faiss.IndexFlatIP(merged_features.shape[1])
        index.add(merged_features)
        faiss.write_index(index, str(output_dir / "index.faiss"))
        logger.info(f"FAISS index built: {index.ntotal} vectors")
    except ImportError:
        logger.warning("faiss not available, skipping index build")

    logger.info(f"Memory bank saved to {output_dir}")
    logger.info(f"  Entries: {len(entries):,}")
    logger.info(f"  Classes: {len(class_counts)}")
    logger.info(f"  Top classes:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {cls:20s}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Merge/convert memory banks")
    parser.add_argument("--input-dirs", nargs="+", type=str, required=True,
                        help="Input directories from build_memory_bank.py")
    parser.add_argument("--output-dir", type=str, default="./outputs/memory_bank_final",
                        help="Output directory for merged bank")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Maximum total entries to keep")
    args = parser.parse_args()

    merge_and_convert(
        input_dirs=[Path(d) for d in args.input_dirs],
        output_dir=Path(args.output_dir),
        max_entries=args.max_entries,
    )


if __name__ == "__main__":
    main()
