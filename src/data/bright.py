"""BRIGHT dataset loader.

BRIGHT (Building Recognition using Image Geospatial data for Humanitarian
Technology) provides high-resolution **optical** and **SAR** satellite imagery
for building-damage assessment.  Each image pair (optical + SAR) is labelled
with a building-damage level.

Reference
---------
BRIGHT Challenge — xView/DIU, 2024-2025.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

__all__ = ["BRIGHTDataset"]


# ── Event catalogue ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _EventMeta:
    event_id: str
    disaster_type: str
    region: str
    year: int


_EVENTS: List[_EventMeta] = [
    _EventMeta("turkey_earthquake_2023", "earthquake", "middle_east", 2023),
    _EventMeta("morocco_earthquake_2023", "earthquake", "africa", 2023),
    _EventMeta("hawaii_wildfire_2023", "wildfire", "north_america", 2023),
    _EventMeta("libya_flood_2023", "flood", "africa", 2023),
    _EventMeta("hurricane_ian_2022", "hurricane", "north_america", 2022),
    _EventMeta("pakistan_flood_2022", "flood", "south_asia", 2022),
    _EventMeta("tonga_volcano_2022", "volcanic_eruption", "oceania", 2022),
    _EventMeta("ukraine_conflict_2022", "conflict", "europe", 2022),
]

_DAMAGE_LABELS: List[str] = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


# ── Dataset class ────────────────────────────────────────────────────────────

@dataclass
class BRIGHTDataset:
    """Load the BRIGHT dataset from a local root directory.

    Expected layout::

        root/
        ├── metadata.csv          # or metadata.json
        ├── optical/
        │   └── <event_id>/
        │       └── *.tif
        └── sar/
            └── <event_id>/
                └── *.tif

    ``metadata.csv`` columns: ``image_id, event_id, label, optical_path,
    sar_path``.

    Alternatively, if a JSON metadata file is present::

        [
            {
                "image_id": "...",
                "event_id": "...",
                "label": "...",
                "optical_path": "...",
                "sar_path": "..."
            },
            ...
        ]

    Parameters
    ----------
    root : str | Path
        Root directory of the BRIGHT dataset.

    Attributes
    ----------
    samples : list[dict[str, Any]]
    """

    root: Path
    samples: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    EVENTS: ClassVar[List[_EventMeta]] = _EVENTS
    DAMAGE_LABELS: ClassVar[List[str]] = _DAMAGE_LABELS
    NAME: ClassVar[str] = "bright"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.samples = self._load()

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        """Attempt CSV then JSON metadata loading."""
        csv_path = self.root / "metadata.csv"
        json_path = self.root / "metadata.json"

        if csv_path.exists():
            return self._load_csv(csv_path)
        elif json_path.exists():
            return self._load_json(json_path)
        else:
            # Fallback: scan optical directory structure
            return self._load_from_directory()

    def _load_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        event_map = {e.event_id: e for e in _EVENTS}
        samples: List[Dict[str, Any]] = []

        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                samples.append(self._row_to_sample(row, event_map))
        return samples

    def _load_json(self, json_path: Path) -> List[Dict[str, Any]]:
        event_map = {e.event_id: e for e in _EVENTS}
        with open(json_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        return [self._row_to_sample(r, event_map) for r in records]

    def _row_to_sample(
        self,
        row: Dict[str, str],
        event_map: Dict[str, _EventMeta],
    ) -> Dict[str, Any]:
        """Convert a metadata row to a standardized sample dict."""
        image_id = row.get("image_id", "")
        event_id_raw = row.get("event_id", "")
        label = row.get("label", "no-damage")
        optical_path = row.get("optical_path", "")
        sar_path = row.get("sar_path", "")

        event = event_map.get(event_id_raw)
        if event is None:
            event = _EventMeta(
                event_id=event_id_raw or "unknown",
                disaster_type="unknown",
                region="unknown",
                year=0,
            )

        # Use optical image as the primary image_path
        primary_image = optical_path or sar_path
        if primary_image and not Path(primary_image).is_absolute():
            primary_image = str(self.root / primary_image)

        return {
            "sample_id": str(image_id),
            "event_id": event.event_id,
            "disaster_type": event.disaster_type,
            "region": event.region,
            "year": event.year,
            "modality": "image",
            "image_path": primary_image if primary_image else None,
            "text": None,
            "label": label.strip().lower(),
            "label_task": "damage_level",
        }

    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """Scan the optical/ directory tree as a fallback."""
        event_map = {e.event_id: e for e in _EVENTS}
        samples: List[Dict[str, Any]] = []
        optical_dir = self.root / "optical"
        if not optical_dir.is_dir():
            return samples

        for event_dir in sorted(optical_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            event_id = event_dir.name
            event = event_map.get(event_id, _EventMeta(
                event_id=event_id,
                disaster_type="unknown",
                region="unknown",
                year=0,
            ))
            for img_file in sorted(event_dir.glob("*.tif")):
                samples.append(
                    {
                        "sample_id": img_file.stem,
                        "event_id": event.event_id,
                        "disaster_type": event.disaster_type,
                        "region": event.region,
                        "year": event.year,
                        "modality": "image",
                        "image_path": str(img_file),
                        "text": None,
                        "label": "unknown",
                        "label_task": "damage_level",
                    }
                )
        return samples

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def event_ids(self) -> List[str]:
        return sorted({s["event_id"] for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    # ── dummy data ───────────────────────────────────────────────────────

    @classmethod
    def generate_dummy(
        cls,
        n_per_event: int = 50,
        output_dir: Optional[str | Path] = None,
        seed: int = 42,
    ) -> "BRIGHTDataset":
        """Create a fake BRIGHT dataset for testing.

        Parameters
        ----------
        n_per_event : int
            Samples per event.
        output_dir : str | Path | None
            Directory for synthetic data.
        seed : int
            Random seed.

        Returns
        -------
        BRIGHTDataset
        """
        import tempfile

        rng = random.Random(seed)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="bright_dummy_"))
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, str]] = []
        for event in _EVENTS:
            optical_event_dir = output_dir / "optical" / event.event_id
            sar_event_dir = output_dir / "sar" / event.event_id
            optical_event_dir.mkdir(parents=True, exist_ok=True)
            sar_event_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_per_event):
                img_id = f"{event.event_id}_{i:04d}"
                opt_name = f"{img_id}.tif"
                sar_name = f"{img_id}.tif"

                # Write tiny placeholder files
                (optical_event_dir / opt_name).write_bytes(b"\x00" * 32)
                (sar_event_dir / sar_name).write_bytes(b"\x00" * 32)

                label = rng.choice(_DAMAGE_LABELS)
                rows.append(
                    {
                        "image_id": img_id,
                        "event_id": event.event_id,
                        "label": label,
                        "optical_path": f"optical/{event.event_id}/{opt_name}",
                        "sar_path": f"sar/{event.event_id}/{sar_name}",
                    }
                )

        csv_path = output_dir / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image_id", "event_id", "label", "optical_path", "sar_path"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return cls(root=output_dir)
