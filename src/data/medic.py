"""MEDIC dataset loader.

MEDIC (MEdieval DIsaster Corpus — later renamed to Multimodal Emergency
Disaster Image Classification) contains ~71 k images across multiple disaster
types, each labelled with a **disaster type** category.  This is an
**image-only** dataset.

Reference
---------
Alam, F., Alam, T., Hasan, M. A., Hasnat, A., Imran, M., & Ofli, F. (2023).
MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification.
*Neural Computing and Applications*.
"""

from __future__ import annotations

import csv
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

__all__ = ["MEDICDataset"]


# ── Event / class catalogue ─────────────────────────────────────────────────

@dataclass(frozen=True)
class _EventMeta:
    event_id: str
    disaster_type: str
    region: str
    year: int


# MEDIC spans many disaster events; we group them by disaster type.
# The actual event IDs come from the metadata in the dataset itself.
_DISASTER_TYPE_LABELS: List[str] = [
    "earthquake",
    "fire",
    "flood",
    "hurricane",
    "landslide",
    "not_disaster",
    "other_disaster",
]

# Representative events from the MEDIC collection
_EVENTS: List[_EventMeta] = [
    _EventMeta("nepal_earthquake_2015", "earthquake", "south_asia", 2015),
    _EventMeta("ecuador_earthquake_2016", "earthquake", "south_america", 2016),
    _EventMeta("italy_earthquake_2016", "earthquake", "europe", 2016),
    _EventMeta("mexico_earthquake_2017", "earthquake", "central_america", 2017),
    _EventMeta("california_wildfires_2017", "fire", "north_america", 2017),
    _EventMeta("california_wildfires_2018", "fire", "north_america", 2018),
    _EventMeta("houston_floods_2016", "flood", "north_america", 2016),
    _EventMeta("kerala_floods_2018", "flood", "south_asia", 2018),
    _EventMeta("hurricane_harvey_2017", "hurricane", "north_america", 2017),
    _EventMeta("hurricane_irma_2017", "hurricane", "caribbean", 2017),
    _EventMeta("hurricane_maria_2017", "hurricane", "caribbean", 2017),
    _EventMeta("hurricane_matthew_2016", "hurricane", "caribbean", 2016),
    _EventMeta("landslide_worldwide", "landslide", "global", 2018),
    _EventMeta("general_not_disaster", "not_disaster", "global", 2018),
]


# ── Dataset class ────────────────────────────────────────────────────────────

@dataclass
class MEDICDataset:
    """Load the MEDIC dataset from a local root directory.

    Expected layout::

        root/
        ├── train.tsv          # or train.csv
        ├── dev.tsv
        ├── test.tsv
        └── images/
            └── *.jpg

    Each TSV/CSV row contains at minimum ``image``, ``label`` (disaster type),
    and optionally ``event_name`` for per-event resolution.

    Parameters
    ----------
    root : str | Path
        Root directory.
    splits : Sequence[str] | None
        Splits to load.  ``None`` loads all.

    Attributes
    ----------
    samples : list[dict[str, Any]]
    """

    root: Path
    splits: Optional[Sequence[str]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    EVENTS: ClassVar[List[_EventMeta]] = _EVENTS
    DISASTER_TYPE_LABELS: ClassVar[List[str]] = _DISASTER_TYPE_LABELS
    NAME: ClassVar[str] = "medic"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.samples = self._load()

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        event_map = {e.event_id: e for e in _EVENTS}

        for annotation_file in sorted(self.root.glob("*.*")):
            if annotation_file.suffix.lower() not in (".tsv", ".csv"):
                continue
            if self.splits is not None and not any(
                s in annotation_file.stem.lower() for s in self.splits
            ):
                continue

            delimiter = "\t" if annotation_file.suffix.lower() == ".tsv" else ","
            with open(annotation_file, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                for row in reader:
                    image_rel = row.get("image") or row.get("image_path", "")
                    label = row.get("label") or row.get("label_image", "")
                    if not label or not image_rel:
                        continue

                    abs_image_path = str(self.root / image_rel)

                    # Resolve event
                    event_id_raw = row.get("event_name") or row.get("event", "")
                    event = event_map.get(event_id_raw)
                    if event is None:
                        # Match by disaster type label
                        label_lower = label.strip().lower()
                        for em in _EVENTS:
                            if em.disaster_type == label_lower:
                                event = em
                                break
                    if event is None:
                        event = _EventMeta(
                            event_id=event_id_raw or "unknown",
                            disaster_type=label.strip().lower(),
                            region="unknown",
                            year=0,
                        )

                    sample_id = row.get("image_id") or Path(image_rel).stem

                    samples.append(
                        {
                            "sample_id": str(sample_id),
                            "event_id": event.event_id,
                            "disaster_type": event.disaster_type,
                            "region": event.region,
                            "year": event.year,
                            "modality": "image",
                            "image_path": abs_image_path,
                            "text": None,
                            "label": label.strip().lower(),
                            "label_task": "disaster_type",
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
    ) -> "MEDICDataset":
        """Create a fake MEDIC dataset for testing.

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
        MEDICDataset
        """
        import tempfile

        rng = random.Random(seed)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="medic_dummy_"))
        else:
            output_dir = Path(output_dir)

        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict[str, str]] = []
        for event in _EVENTS:
            for i in range(n_per_event):
                img_name = f"{event.event_id}_{i:04d}.jpg"
                img_path = image_dir / img_name
                img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

                all_rows.append(
                    {
                        "image_id": f"{event.event_id}_{i:04d}",
                        "image": f"images/{img_name}",
                        "label": event.disaster_type,
                        "event_name": event.event_id,
                    }
                )

        rng.shuffle(all_rows)
        tsv_path = output_dir / "train.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image_id", "image", "label", "event_name"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(all_rows)

        return cls(root=output_dir)
