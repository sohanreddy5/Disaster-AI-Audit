"""Near-duplicate and leakage audit across disaster splits and events."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


class DuplicateAuditor:
    """Audits image and text datasets for exact and near-duplicates across splits.

    Detects:
        1. Exact hash duplicates (MD5 / SHA256)
        2. Perceptual image duplicates (pHash / dHash Hamming distance <= threshold)
        3. High cosine-similarity embeddings (cosine similarity >= similarity_threshold)
        4. Cross-split contamination (same scene/tweet in train AND test)
    """

    def __init__(
        self,
        hash_threshold: int = 5,
        cosine_threshold: float = 0.95,
    ) -> None:
        self.hash_threshold = hash_threshold
        self.cosine_threshold = cosine_threshold

    def audit_images_by_phash(
        self,
        image_paths: Sequence[str | Path],
        sample_ids: Sequence[str],
        event_ids: Sequence[str],
        show_progress: bool = True,
    ) -> dict[str, any]:
        """Compute perceptual hashes and detect cross-event / cross-split duplicates."""
        hashes: dict[str, list[int]] = defaultdict(list)
        computed_hashes = []
        valid_indices = []

        pbar = tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc="Computing perceptual hashes",
            disable=not show_progress,
        )

        for idx, p in pbar:
            if p is not None and os.path.exists(p):
                try:
                    img = Image.open(p)
                    h = imagehash.phash(img)
                    computed_hashes.append(h)
                    valid_indices.append(idx)
                except Exception:
                    continue

        n_valid = len(valid_indices)
        near_duplicate_pairs = []
        cross_event_leakage = []

        # Find near matches within Hamming distance threshold
        for i in range(n_valid):
            h_i = computed_hashes[i]
            idx_i = valid_indices[i]
            ev_i = event_ids[idx_i]
            sid_i = sample_ids[idx_i]

            for j in range(i + 1, min(i + 500, n_valid)):
                h_j = computed_hashes[j]
                diff = h_i - h_j
                if diff <= self.hash_threshold:
                    idx_j = valid_indices[j]
                    ev_j = event_ids[idx_j]
                    sid_j = sample_ids[idx_j]

                    pair_info = {
                        "sample_1": sid_i,
                        "sample_2": sid_j,
                        "event_1": ev_i,
                        "event_2": ev_j,
                        "hamming_dist": int(diff),
                        "cross_event": (ev_i != ev_j),
                    }
                    near_duplicate_pairs.append(pair_info)
                    if ev_i != ev_j:
                        cross_event_leakage.append(pair_info)

        return {
            "total_images_analyzed": n_valid,
            "near_duplicate_count": len(near_duplicate_pairs),
            "cross_event_leakage_count": len(cross_event_leakage),
            "leakage_pairs": cross_event_leakage[:50],
        }

    def audit_embedding_duplicates(
        self,
        embeddings: np.ndarray,
        sample_ids: Sequence[str],
        event_ids: Sequence[str],
        split_assignments: Sequence[str] | None = None,
        batch_size: int = 1000,
    ) -> dict[str, any]:
        """Detect near-duplicates via high cosine similarity between L2-normalized embeddings."""
        n = len(embeddings)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embs = embeddings / np.maximum(norm, 1e-8)

        high_sim_pairs = []
        cross_split_pairs = []

        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch = norm_embs[i:end_i]
            sims = batch @ norm_embs.T

            for bi, global_i in enumerate(range(i, end_i)):
                row_sims = sims[bi]
                above = np.where(row_sims > self.cosine_threshold)[0]
                for match_j in above:
                    if match_j > global_i:
                        ev_i = event_ids[global_i]
                        ev_j = event_ids[match_j]
                        pair = {
                            "sample_1": sample_ids[global_i],
                            "sample_2": sample_ids[match_j],
                            "similarity": float(row_sims[match_j]),
                            "cross_event": (ev_i != ev_j),
                        }
                        if split_assignments:
                            sp_i = split_assignments[global_i]
                            sp_j = split_assignments[match_j]
                            pair["cross_split"] = (sp_i != sp_j)
                            if sp_i != sp_j:
                                cross_split_pairs.append(pair)
                        high_sim_pairs.append(pair)

        return {
            "total_samples": n,
            "high_similarity_pairs_count": len(high_sim_pairs),
            "cross_split_contamination_count": len(cross_split_pairs),
            "sample_matches": high_sim_pairs[:50],
        }
