"""CrisisMMD v2.0 dataset loader.

CrisisMMD v2.0 contains ~18 k image–tweet pairs across **7 disaster events**,
each annotated with two label tasks:

* **informativeness**: ``informative`` / ``not_informative``
* **humanitarian**: multi-class humanitarian category

Reference
---------
Alam, F., Ofli, F., & Imran, M.  (2018).  CrisisMMD: Multimodal Crisis
Dataset.  *ICWSM*.
"""

from __future__ import annotations

import csv
import os
import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

__all__ = ["CrisisMMDDataset"]


# ── Event catalogue ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _EventMeta:
    """Metadata for a single disaster event in CrisisMMD."""

    event_id: str
    disaster_type: str
    region: str
    year: int


_EVENTS: List[_EventMeta] = [
    _EventMeta("california_wildfires", "wildfire", "north_america", 2017),
    _EventMeta("hurricane_harvey", "hurricane", "north_america", 2017),
    _EventMeta("hurricane_irma", "hurricane", "caribbean", 2017),
    _EventMeta("hurricane_maria", "hurricane", "caribbean", 2017),
    _EventMeta("iraq_iran_earthquake", "earthquake", "middle_east", 2017),
    _EventMeta("mexico_earthquake", "earthquake", "central_america", 2017),
    _EventMeta("srilanka_floods", "flood", "south_asia", 2017),
]

_INFORMATIVENESS_LABELS: List[str] = ["informative", "not_informative"]
_HUMANITARIAN_LABELS: List[str] = [
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "affected_individuals",
    "vehicle_damage",
    "missing_or_found_people",
    "injured_or_dead_people",
    "other_relevant_information",
    "not_humanitarian",
]


# ── Dataset class ────────────────────────────────────────────────────────────

@dataclass
class CrisisMMDDataset:
    """Load CrisisMMD v2.0 from a local root directory.

    The expected directory layout mirrors the official release::

        root/
        ├── <event_id>/
        │   ├── <event_id>_informativeness_<split>.tsv
        │   ├── <event_id>_humanitarian_<split>.tsv
        │   └── images/
        │       └── *.jpg
        └── ...

    Parameters
    ----------
    root : str | Path
        Path to the CrisisMMD v2.0 root directory.
    label_task : str
        Which label task to load — ``"informativeness"`` or ``"humanitarian"``.
    splits : Sequence[str] | None
        Which splits to load (e.g. ``["train", "dev", "test"]``).
        ``None`` means load all available splits.

    Attributes
    ----------
    samples : list[dict[str, Any]]
        Standardized sample dicts.
    """

    root: Path
    label_task: str = "informativeness"
    splits: Optional[Sequence[str]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    # Class-level constants
    EVENTS: ClassVar[List[_EventMeta]] = _EVENTS
    EVENT_LOOKUP: ClassVar[Dict[str, _EventMeta]] = {e.event_id: e for e in _EVENTS}
    INFORMATIVENESS_LABELS: ClassVar[List[str]] = _INFORMATIVENESS_LABELS
    HUMANITARIAN_LABELS: ClassVar[List[str]] = _HUMANITARIAN_LABELS
    NAME: ClassVar[str] = "crisismmd"

    def __post_init__(self) -> None:
        import json
        self._json = json
        self.root = Path(self.root)
        if self.label_task in ("informative", "informativeness"):
            self.label_task = "informative"
        elif self.label_task == "humanitarian":
            pass
        else:
            raise ValueError(
                f"label_task must be 'informative' / 'informativeness' or 'humanitarian', "
                f"got '{self.label_task}'"
            )
        self.samples = self._load()


    # ── loading ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        """Parse JSON or TSV annotation files and build standardized sample list."""
        samples: List[Dict[str, Any]] = []

        # 1. Check for JSON format (HuggingFace / QCRI JSON structure)
        json_dir = self.root / self.label_task
        if not json_dir.is_dir():
            json_dir = self.root

        json_splits = ["train", "dev", "test"] if self.splits is None else self.splits
        loaded_json = False
        for s in json_splits:
            jf = json_dir / f"{s}.json"
            if jf.exists():
                loaded_json = True
                with open(jf, "r", encoding="utf-8") as fp:
                    items = self._json.load(fp)
                for item in items:
                    ev_name = item.get("event_name", "unknown")
                    meta = self.EVENT_LOOKUP.get(ev_name)
                    dtype = meta.disaster_type if meta else "disaster"
                    reg = meta.region if meta else "unknown"
                    yr = meta.year if meta else 2017

                    img_rel = item.get("image_path")
                    abs_img = str(self.root / img_rel) if img_rel else None

                    samples.append(
                        {
                            "sample_id": str(item.get("image_id") or item.get("tweet_id")),
                            "event_id": ev_name,
                            "disaster_type": dtype,
                            "region": reg,
                            "year": yr,
                            "modality": "both",
                            "image_path": abs_img,
                            "text": item.get("tweet_text"),
                            "label": str(item.get("label", "")).strip().lower(),
                            "label_task": self.label_task,
                        }
                    )

        if loaded_json and len(samples) > 0:
            return samples

        # 2. Fall back to event directory TSV format
        for event in _EVENTS:
            event_dir = self.root / event.event_id
            if not event_dir.is_dir():
                continue

            tsv_files = self._find_tsv_files(event_dir, event.event_id)
            for tsv_path in tsv_files:
                samples.extend(self._parse_tsv(tsv_path, event))

        return samples


    def _find_tsv_files(
        self, event_dir: Path, event_id: str
    ) -> List[Path]:
        """Locate annotation TSVs for the configured task and splits."""
        found: List[Path] = []
        for child in sorted(event_dir.iterdir()):
            if not child.suffix == ".tsv":
                continue
            name_lower = child.stem.lower()
            if self.label_task not in name_lower:
                continue
            if self.splits is not None:
                if not any(s in name_lower for s in self.splits):
                    continue
            found.append(child)
        return found

    def _parse_tsv(
        self, tsv_path: Path, event: _EventMeta
    ) -> List[Dict[str, Any]]:
        """Parse a single TSV file into standardized dicts."""
        samples: List[Dict[str, Any]] = []
        with open(tsv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                image_path = row.get("image_path") or row.get("image", "")
                text = row.get("tweet_text") or row.get("text", "")
                label = row.get("label") or row.get("label_text", "")
                if not label:
                    continue

                # Resolve image path relative to root
                abs_image_path: Optional[str] = None
                if image_path:
                    candidate = self.root / image_path
                    if not candidate.exists():
                        candidate = self.root / event.event_id / image_path
                    abs_image_path = str(candidate)

                sample_id = (
                    row.get("image_id")
                    or row.get("tweet_id")
                    or f"{event.event_id}_{len(samples)}"
                )

                samples.append(
                    {
                        "sample_id": str(sample_id),
                        "event_id": event.event_id,
                        "disaster_type": event.disaster_type,
                        "region": event.region,
                        "year": event.year,
                        "modality": "both",
                        "image_path": abs_image_path,
                        "text": text if text else None,
                        "label": label.strip().lower(),
                        "label_task": self.label_task,
                    }
                )
        return samples

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def event_ids(self) -> List[str]:
        """Unique event IDs present in the loaded samples."""
        return sorted({s["event_id"] for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    # ── dummy data generation ────────────────────────────────────────────

    @classmethod
    def generate_dummy(
        cls,
        n_per_event: int = 50,
        output_dir: Optional[str | Path] = None,
        label_task: str = "informativeness",
        seed: int = 42,
    ) -> "CrisisMMDDataset":
        """Create a fake CrisisMMD dataset for testing.

        Generates synthetic TSV annotation files and tiny placeholder images
        under *output_dir*, then returns a ``CrisisMMDDataset`` loaded from
        that directory.

        Parameters
        ----------
        n_per_event : int
            Number of samples to generate per event.
        output_dir : str | Path | None
            Where to write the fake data.  Defaults to a temp directory.
        label_task : str
            ``"informativeness"`` or ``"humanitarian"``.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        CrisisMMDDataset
            A fully-loaded dataset instance backed by dummy data.
        """
        import tempfile

        rng = random.Random(seed)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="crisismmd_dummy_"))
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labels = (
            _INFORMATIVENESS_LABELS
            if label_task == "informativeness"
            else _HUMANITARIAN_LABELS
        )

        for event in _EVENTS:
            event_dir = output_dir / event.event_id
            image_dir = event_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            rows: List[Dict[str, str]] = []
            for i in range(n_per_event):
                img_name = f"{event.event_id}_{i:04d}.jpg"
                img_path = image_dir / img_name
                # Write a 1×1 JPEG-like placeholder (valid enough for path checks)
                img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

                tweet_text = " ".join(
                    rng.choices(string.ascii_lowercase.split(), k=12)
                    if hasattr(string.ascii_lowercase, "split")
                    else [
                        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
                        for _ in range(12)
                    ]
                )
                label = rng.choice(labels)
                rows.append(
                    {
                        "image_id": f"{event.event_id}_{i:04d}",
                        "image_path": f"{event.event_id}/images/{img_name}",
                        "tweet_text": tweet_text,
                        "label": label,
                    }
                )

            tsv_path = event_dir / f"{event.event_id}_{label_task}_train.tsv"
            with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["image_id", "image_path", "tweet_text", "label"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)

        return cls(root=output_dir, label_task=label_task)
