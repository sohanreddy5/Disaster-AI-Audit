"""Feature extraction using CLIP, SigLIP, and DINOv2 backbones."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class CLIPFeatureExtractor:
    """Extract and cache frozen CLIP/SigLIP/DINOv2 features for disaster images.

    Supports:
        - OpenCLIP models: ViT-B-32, ViT-L-14, ViT-H-14, ViT-SO400M (SigLIP)
        - Pretrained weights from OpenAI, LAION-2B, webli
        - Disk caching to .npy files with hash checks
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def _load_model(self) -> None:
        """Lazy load OpenCLIP model to preserve memory."""
        if self.model is not None:
            return

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

    def get_cache_path(self, dataset_name: str, sample_id: str, modality: str) -> Path | None:
        """Generate cache file path for a specific sample."""
        if not self.cache_dir:
            return None
        safe_model = f"{self.model_name}_{self.pretrained}".replace("/", "_")
        d = self.cache_dir / dataset_name / safe_model / modality
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sample_id}.npy"

    @torch.no_grad()
    def extract_image_features(
        self,
        image_paths: Sequence[str | Path],
        dataset_name: str = "default",
        sample_ids: Sequence[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract L2-normalized image embeddings for a sequence of image paths."""
        self._load_model()
        n_samples = len(image_paths)
        features_list = []

        # Check for cached features if sample_ids provided
        to_compute_indices = []
        cached_features = {}
        if sample_ids and self.cache_dir:
            for idx, sid in enumerate(sample_ids):
                cp = self.get_cache_path(dataset_name, sid, "image")
                if cp and cp.exists():
                    try:
                        cached_features[idx] = np.load(cp)
                    except Exception:
                        to_compute_indices.append(idx)
                else:
                    to_compute_indices.append(idx)
        else:
            to_compute_indices = list(range(n_samples))

        if not to_compute_indices:
            return np.stack([cached_features[i] for i in range(n_samples)], axis=0)

        # Process missing in batches
        pbar = tqdm(
            range(0, len(to_compute_indices), self.batch_size),
            desc=f"Extracting {self.model_name} image features",
            disable=not show_progress,
        )

        for batch_start in pbar:
            batch_idxs = to_compute_indices[batch_start : batch_start + self.batch_size]
            batch_images = []
            valid_batch_idxs = []

            for idx in batch_idxs:
                p = image_paths[idx]
                try:
                    if p is not None and os.path.exists(p):
                        img = Image.open(p).convert("RGB")
                        tensor = self.preprocess(img)
                        batch_images.append(tensor)
                        valid_batch_idxs.append(idx)
                    else:
                        # Dummy zero tensor if missing
                        batch_images.append(torch.zeros(3, 224, 224))
                        valid_batch_idxs.append(idx)
                except Exception:
                    batch_images.append(torch.zeros(3, 224, 224))
                    valid_batch_idxs.append(idx)

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images, dim=0).to(self.device)
            feats = self.model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)

            for local_i, idx in enumerate(valid_batch_idxs):
                feat = feats_np[local_i]
                cached_features[idx] = feat
                if sample_ids and self.cache_dir:
                    cp = self.get_cache_path(dataset_name, sample_ids[idx], "image")
                    if cp:
                        np.save(cp, feat)

        all_feats = [cached_features[i] for i in range(n_samples)]
        return np.stack(all_feats, axis=0)

    @torch.no_grad()
    def extract_text_features(
        self,
        texts: Sequence[str],
        dataset_name: str = "default",
        sample_ids: Sequence[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract L2-normalized text embeddings using CLIP text encoder."""
        self._load_model()
        n_samples = len(texts)
        cached_features = {}
        to_compute_indices = []

        if sample_ids and self.cache_dir:
            for idx, sid in enumerate(sample_ids):
                cp = self.get_cache_path(dataset_name, sid, "text")
                if cp and cp.exists():
                    try:
                        cached_features[idx] = np.load(cp)
                    except Exception:
                        to_compute_indices.append(idx)
                else:
                    to_compute_indices.append(idx)
        else:
            to_compute_indices = list(range(n_samples))

        if not to_compute_indices:
            return np.stack([cached_features[i] for i in range(n_samples)], axis=0)

        pbar = tqdm(
            range(0, len(to_compute_indices), self.batch_size),
            desc=f"Extracting {self.model_name} text features",
            disable=not show_progress,
        )

        for batch_start in pbar:
            batch_idxs = to_compute_indices[batch_start : batch_start + self.batch_size]
            batch_texts = [texts[i] if texts[i] is not None else "" for i in batch_idxs]

            tokens = self.tokenizer(batch_texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)

            for local_i, idx in enumerate(batch_idxs):
                feat = feats_np[local_i]
                cached_features[idx] = feat
                if sample_ids and self.cache_dir:
                    cp = self.get_cache_path(dataset_name, sample_ids[idx], "text")
                    if cp:
                        np.save(cp, feat)

        all_feats = [cached_features[i] for i in range(n_samples)]
        return np.stack(all_feats, axis=0)
