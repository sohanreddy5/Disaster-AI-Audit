"""Text feature extraction using SentenceTransformers backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm


class TextFeatureExtractor:
    """Extract and cache sentence embeddings for disaster tweets and text data.

    Supported models:
        - sentence-transformers/all-MiniLM-L6-v2 (fast 384-d baseline)
        - sentence-transformers/all-mpnet-base-v2 (768-d high-accuracy)
        - BAAI/bge-base-en-v1.5
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def get_cache_path(self, dataset_name: str, sample_id: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe_model = self.model_name.replace("/", "_")
        d = self.cache_dir / dataset_name / safe_model / "text"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sample_id}.npy"

    def extract_features(
        self,
        texts: Sequence[str],
        dataset_name: str = "default",
        sample_ids: Sequence[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract L2-normalized embeddings for input texts."""
        self._load_model()
        n_samples = len(texts)
        clean_texts = [t if (t is not None and len(t.strip()) > 0) else "[EMPTY]" for t in texts]

        cached_features = {}
        to_compute_indices = []

        if sample_ids and self.cache_dir:
            for idx, sid in enumerate(sample_ids):
                cp = self.get_cache_path(dataset_name, sid)
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
            desc=f"Extracting {self.model_name} embeddings",
            disable=not show_progress,
        )

        for batch_start in pbar:
            batch_idxs = to_compute_indices[batch_start : batch_start + self.batch_size]
            sub_texts = [clean_texts[i] for i in batch_idxs]

            embeddings = self.model.encode(
                sub_texts,
                batch_size=len(sub_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for local_i, idx in enumerate(batch_idxs):
                feat = embeddings[local_i].astype(np.float32)
                cached_features[idx] = feat
                if sample_ids and self.cache_dir:
                    cp = self.get_cache_path(dataset_name, sample_ids[idx])
                    if cp:
                        np.save(cp, feat)

        all_feats = [cached_features[i] for i in range(n_samples)]
        return np.stack(all_feats, axis=0)
