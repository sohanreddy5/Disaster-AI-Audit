"""xBD (xView2) dataset loader.

xBD is the largest public building-damage assessment dataset, created for the
xView2 challenge.  It contains high-resolution satellite imagery captured
**before and after** natural disasters with per-building damage annotations on
a 0–3 ordinal scale:

* 0 — no damage
* 1 — minor damage
* 2 — major damage
* 3 — destroyed

Reference
---------
Gupta, R., Hosfelt, R., Saez, S., et al.  (2019).  xBD: A Dataset for
Assessing Building Damage from Satellite Imagery.  *CVPR Workshops*.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

__all__ = ["xBDDataset"]


# ── Event catalogue ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _EventMeta:
    event_id: str
    disaster_type: str
    region: str
    year: int


_EVENTS: List[_EventMeta] = [
    # Floods
    _EventMeta("midwest_flooding", "flood", "north_america", 2019),
    _EventMeta("nepal_flooding", "flood", "south_asia", 2017),
    _EventMeta("florence_flood", "flood", "north_america", 2018),
    # Hurricanes / typhoons / cyclones
    _EventMeta("hurricane_harvey", "hurricane", "north_america", 2017),
    _EventMeta("hurricane_matthew", "hurricane", "caribbean", 2016),
    _EventMeta("hurricane_michael", "hurricane", "north_america", 2018),
    _EventMeta("hurricane_florence", "hurricane", "north_america", 2018),
    _EventMeta("typhoon_mangkhut", "typhoon", "east_asia", 2018),
    _EventMeta("cyclone_idai", "cyclone", "africa", 2019),
    # Earthquakes
    _EventMeta("mexico_earthquake", "earthquake", "central_america", 2017),
    _EventMeta("nepal_earthquake", "earthquake", "south_asia", 2015),
    # Fires
    _EventMeta("santa_rosa_wildfire", "wildfire", "north_america", 2017),
    _EventMeta("woolsey_fire", "wildfire", "north_america", 2018),
    _EventMeta("socal_fire", "wildfire", "north_america", 2017),
    # Volcanic eruption
    _EventMeta("guatemala_volcano", "volcanic_eruption", "central_america", 2018),
    # Tsunami
    _EventMeta("palu_tsunami", "tsunami", "southeast_asia", 2018),
    # Tornado
    _EventMeta("joplin_tornado", "tornado", "north_america", 2011),
    _EventMeta("moore_tornado", "tornado", "north_america", 2013),
    _EventMeta("tuscaloosa_tornado", "tornado", "north_america", 2011),
]

_DAMAGE_LABELS: List[str] = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

_DAMAGE_LEVEL_MAP: Dict[str, int] = {lbl: i for i, lbl in enumerate(_DAMAGE_LABELS)}


# ── Dataset class ────────────────────────────────────────────────────────────

@dataclass
class xBDDataset:
    """Load the xBD (xView2) dataset from a local root directory.

    Expected layout (matches the official xView2 release)::

        root/
        ├── train/
        │   ├── images/
        │   │   ├── <disaster>_<id>_pre_disaster.png
        │   │   └── <disaster>_<id>_post_disaster.png
        │   └── labels/
        │       └── <disaster>_<id>_post_disaster.json
        ├── tier3/
        │   └── ...
        └── test/
            └── ...

    Each JSON label file contains per-building polygons with a damage class.
    This loader emits one sample **per image pair** (pre + post), with the
    overall damage label being the **maximum damage level** across buildings
    in that pair.

    Parameters
    ----------
    root : str | Path
        Root directory.
    splits : Sequence[str] | None
        Splits to load (e.g. ``["train", "tier3", "test"]``).
    """

    root: Path
    splits: Optional[Sequence[str]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    EVENTS: ClassVar[List[_EventMeta]] = _EVENTS
    DAMAGE_LABELS: ClassVar[List[str]] = _DAMAGE_LABELS
    NAME: ClassVar[str] = "xbd"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.samples = self._load()

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        event_map = {e.event_id: e for e in _EVENTS}

        split_dirs = (
            [self.root / s for s in self.splits]
            if self.splits
            else [d for d in self.root.iterdir() if d.is_dir()]
        )

        for split_dir in sorted(split_dirs):
            if not split_dir.is_dir():
                continue
            labels_dir = split_dir / "labels"
            images_dir = split_dir / "images"
            if not labels_dir.is_dir():
                continue

            for label_file in sorted(labels_dir.glob("*_post_disaster.json")):
                parsed = self._parse_label_file(
                    label_file, images_dir, event_map
                )
                if parsed is not None:
                    samples.append(parsed)

        return samples

    def _parse_label_file(
        self,
        label_file: Path,
        images_dir: Path,
        event_map: Dict[str, _EventMeta],
    ) -> Optional[Dict[str, Any]]:
        """Parse a single xBD label JSON and return a standardized sample."""
        try:
            with open(label_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        metadata = data.get("metadata", {})
        disaster_name = metadata.get("disaster", label_file.stem.split("_")[0])
        disaster_type = metadata.get("disaster_type", "unknown")

        # Find matching event
        event: Optional[_EventMeta] = event_map.get(disaster_name)
        if event is None:
            for eid, emeta in event_map.items():
                if eid in disaster_name or disaster_name in eid:
                    event = emeta
                    break
        if event is None:
            event = _EventMeta(
                event_id=disaster_name,
                disaster_type=disaster_type,
                region="unknown",
                year=0,
            )

        # Determine max damage across all buildings
        features = (
            data.get("features", {}).get("xy", [])
            if isinstance(data.get("features"), dict)
            else []
        )
        max_damage: int = 0
        for feat in features:
            props = feat.get("properties", {})
            subtype = props.get("subtype", "no-damage")
            level = _DAMAGE_LEVEL_MAP.get(subtype, 0)
            max_damage = max(max_damage, level)

        # Build image path — use post-disaster image as the primary
        stem = label_file.stem  # e.g. disaster_00001_post_disaster
        post_img = images_dir / f"{stem}.png"
        pre_stem = stem.replace("_post_disaster", "_pre_disaster")
        pre_img = images_dir / f"{pre_stem}.png"

        # Sample id from file stem
        sample_id = stem.replace("_post_disaster", "")

        return {
            "sample_id": sample_id,
            "event_id": event.event_id,
            "disaster_type": event.disaster_type,
            "region": event.region,
            "year": event.year,
            "modality": "image",
            "image_path": str(post_img),
            "text": None,
            "label": _DAMAGE_LABELS[max_damage],
            "label_task": "damage_level",
        }

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
    ) -> "xBDDataset":
        """Create a fake xBD dataset for testing.

        Parameters
        ----------
        n_per_event : int
            Image pairs per event.
        output_dir : str | Path | None
            Directory for synthetic data.
        seed : int
            Random seed.

        Returns
        -------
        xBDDataset
        """
        import tempfile

        rng = random.Random(seed)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="xbd_dummy_"))
        else:
            output_dir = Path(output_dir)

        split_name = "train"
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for event in _EVENTS:
            for i in range(n_per_event):
                stem = f"{event.event_id}_{i:05d}"

                # Write placeholder images
                for suffix in ("_pre_disaster.png", "_post_disaster.png"):
                    (images_dir / f"{stem}{suffix}").write_bytes(
                        b"\x89PNG\r\n\x1a\n" + b"\x00" * 30
                    )

                # Pick random damage per-building
                n_buildings = rng.randint(1, 8)
                features = []
                for _ in range(n_buildings):
                    dmg = rng.choice(_DAMAGE_LABELS)
                    features.append(
                        {
                            "properties": {"subtype": dmg},
                            "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                        }
                    )

                label_data = {
                    "metadata": {
                        "disaster": event.event_id,
                        "disaster_type": event.disaster_type,
                    },
                    "features": {"xy": features},
                }
                label_path = labels_dir / f"{stem}_post_disaster.json"
                with open(label_path, "w", encoding="utf-8") as fh:
                    json.dump(label_data, fh)

        return cls(root=output_dir, splits=[split_name])
