#!/usr/bin/env python3
"""Feature extraction script for Disaster AI benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_dataset
from src.features.clip_extractor import CLIPFeatureExtractor
from src.features.text_extractor import TextFeatureExtractor
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen embeddings for disaster datasets.")
    parser.add_argument("--dataset", type=str, default="crisismmd", help="Dataset name in registry")
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory containing datasets")
    parser.add_argument("--cache-dir", type=str, default="./cache/features", help="Output feature cache directory")
    parser.add_argument("--image-model", type=str, default="ViT-B-32", help="OpenCLIP vision backbone")
    parser.add_argument("--image-pretrained", type=str, default="openai", help="Pretrained weights")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Text backbone")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    logger.info(f"Loading dataset '{args.dataset}' from {data_root}")
    try:
        dataset = get_dataset(args.dataset, data_root=data_root)
        samples = dataset.load_samples()
        logger.info(f"Loaded {len(samples)} samples from {args.dataset}")
    except Exception as e:
        logger.error(f"Could not load real dataset '{args.dataset}': {e}")
        logger.info("Make sure dataset is downloaded to data_root using scripts/download_datasets.sh")
        return

    sample_ids = [s.sample_id for s in samples]

    # Vision feature extraction
    image_paths = [s.image_path for s in samples if s.image_path is not None]
    if image_paths:
        logger.info(f"Extracting vision features with {args.image_model} ({args.image_pretrained})...")
        vision_extractor = CLIPFeatureExtractor(
            model_name=args.image_model,
            pretrained=args.image_pretrained,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
        vision_extractor.extract_image_features(
            image_paths=[s.image_path for s in samples],
            dataset_name=args.dataset,
            sample_ids=sample_ids,
        )

    # Text feature extraction
    texts = [s.text for s in samples if s.text is not None]
    if texts:
        logger.info(f"Extracting text features with {args.text_model}...")
        text_extractor = TextFeatureExtractor(
            model_name=args.text_model,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
        text_extractor.extract_features(
            texts=[s.text for s in samples],
            dataset_name=args.dataset,
            sample_ids=sample_ids,
        )

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
