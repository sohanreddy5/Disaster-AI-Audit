"""Data loading utilities for disaster AI audit benchmark.

Provides standardized dataset loaders for five disaster-related datasets:
CrisisMMD, HumAID, MEDIC, xBD, and BRIGHT. Each loader returns samples in a
common dict format so that downstream split generators, feature extractors,
and evaluation code can operate dataset-agnostically.

Standardized sample format::

    {
        "sample_id":      str,   # unique identifier
        "event_id":       str,   # disaster event slug
        "disaster_type":  str,   # e.g. "hurricane", "earthquake", "wildfire"
        "region":         str,   # geographic region
        "year":           int,   # year of the disaster
        "modality":       str,   # "image" | "text" | "both"
        "image_path":     str | None,
        "text":           str | None,
        "label":          str,   # class label
        "label_task":     str,   # task that the label belongs to
    }
"""

from __future__ import annotations

from src.data.crisismmd import CrisisMMDDataset
from src.data.humaid import HumAIDDataset
from src.data.medic import MEDICDataset
from src.data.xbd import xBDDataset
from src.data.bright import BRIGHTDataset

__all__ = [
    "CrisisMMDDataset",
    "HumAIDDataset",
    "MEDICDataset",
    "xBDDataset",
    "BRIGHTDataset",
]
