"""HumAID dataset loader.

HumAID contains ~77 k annotated tweets across **19 disaster events**, each
labelled with a humanitarian category (e.g. *rescue*, *infrastructure damage*,
*sympathy and support*).  This is a **text-only** dataset.

Reference
---------
Alam, F., Sajjad, H., Imran, M., & Ofli, F.  (2021).  HumAID: Human
Annotated Disaster Tweets.  *ICWSM*.
"""

from __future__ import annotations

import csv
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

__all__ = ["HumAIDDataset"]


# ── Event catalogue ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _EventMeta:
    event_id: str
    disaster_type: str
    region: str
    year: int


_EVENTS: List[_EventMeta] = [
    _EventMeta("california_wildfires_2018", "wildfire", "north_america", 2018),
    _EventMeta("canada_wildfires_2016", "wildfire", "north_america", 2016),
    _EventMeta("cyclone_idai_2019", "cyclone", "africa", 2019),
    _EventMeta("ecuador_earthquake_2016", "earthquake", "south_america", 2016),
    _EventMeta("greece_wildfires_2018", "wildfire", "europe", 2018),
    _EventMeta("hurricane_dorian_2019", "hurricane", "caribbean", 2019),
    _EventMeta("hurricane_florence_2018", "hurricane", "north_america", 2018),
    _EventMeta("hurricane_harvey_2017", "hurricane", "north_america", 2017),
    _EventMeta("hurricane_irma_2017", "hurricane", "caribbean", 2017),
    _EventMeta("hurricane_maria_2017", "hurricane", "caribbean", 2017),
    _EventMeta("hurricane_matthew_2016", "hurricane", "caribbean", 2016),
    _EventMeta("italy_earthquake_aug_2016", "earthquake", "europe", 2016),
    _EventMeta("kaikoura_earthquake_2016", "earthquake", "oceania", 2016),
    _EventMeta("kerala_floods_2018", "flood", "south_asia", 2018),
    _EventMeta("maryland_floods_2018", "flood", "north_america", 2018),
    _EventMeta("midwestern_us_floods_2019", "flood", "north_america", 2019),
    _EventMeta("pakistan_earthquake_2019", "earthquake", "south_asia", 2019),
    _EventMeta("puebla_mexico_earthquake_2017", "earthquake", "central_america", 2017),
    _EventMeta("srilanka_floods_2017", "flood", "south_asia", 2017),
]


_HUMANITARIAN_LABELS: List[str] = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "dont_know_or_cant_judge",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
]


# ── Dataset class ────────────────────────────────────────────────────────────

@dataclass
class HumAIDDataset:
    """Load the HumAID dataset from a local root directory.

    Expected layout::

        root/
        ├── <event_id>/
        │   └── <event_id>_<split>.tsv   # tab-separated: tweet_id \\t text \\t label
        └── ...

    Or a flat layout with one combined TSV per split::

        root/
        ├── train.tsv
        ├── dev.tsv
        └── test.tsv

    Parameters
    ----------
    root : str | Path
        Root directory of the HumAID dataset.
    splits : Sequence[str] | None
        Splits to load (e.g. ``["train", "dev", "test"]``).  ``None`` loads all.

    Attributes
    ----------
    samples : list[dict[str, Any]]
        Standardized sample dicts.
    """

    root: Path
    splits: Optional[Sequence[str]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    EVENTS: ClassVar[List[_EventMeta]] = _EVENTS
    HUMANITARIAN_LABELS: ClassVar[List[str]] = _HUMANITARIAN_LABELS
    NAME: ClassVar[str] = "humaid"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.samples = self._load()

    # ── internal loading ─────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        # Try per-event sub-directories first
        event_map = {e.event_id: e for e in _EVENTS}
        loaded_any = False

        for event in _EVENTS:
            event_dir = self.root / event.event_id
            if not event_dir.is_dir():
                continue

            # Check JSON files (Hugging Face HumAID format)
            for json_path in sorted(event_dir.glob("*.json")):
                if self.splits is not None and not any(
                    s in json_path.stem.lower() for s in self.splits
                ):
                    continue
                import json
                with open(json_path, "r", encoding="utf-8") as fh:
                    items = json.load(fh)
                for item in items:
                    lbl = item.get("class_label") or item.get("label", "")
                    if not lbl:
                        continue
                    samples.append(
                        {
                            "sample_id": str(item.get("tweet_id", f"{event.event_id}_{len(samples)}")),
                            "event_id": event.event_id,
                            "disaster_type": event.disaster_type,
                            "region": event.region,
                            "year": event.year,
                            "modality": "text",
                            "image_path": None,
                            "text": item.get("tweet_text", ""),
                            "label": lbl.strip().lower(),
                            "label_task": "humanitarian",
                        }
                    )
                loaded_any = True

            # Check TSV files
            for tsv_path in sorted(event_dir.glob("*.tsv")):
                if self.splits is not None and not any(
                    s in tsv_path.stem.lower() for s in self.splits
                ):
                    continue
                samples.extend(self._parse_tsv(tsv_path, fixed_event=event))
                loaded_any = True


        # Fallback: flat combined TSV files
        if not loaded_any:
            for tsv_path in sorted(self.root.glob("*.tsv")):
                if self.splits is not None and not any(
                    s in tsv_path.stem.lower() for s in self.splits
                ):
                    continue
                samples.extend(self._parse_tsv(tsv_path, fixed_event=None))

        return samples

    def _parse_tsv(
        self,
        tsv_path: Path,
        fixed_event: Optional[_EventMeta],
    ) -> List[Dict[str, Any]]:
        """Parse a single TSV annotation file."""
        rows: List[Dict[str, Any]] = []
        event_map = {e.event_id: e for e in _EVENTS}

        with open(tsv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                tweet_id = row.get("tweet_id", "")
                text = row.get("tweet_text") or row.get("text", "")
                label = row.get("label") or row.get("class_label", "")
                if not label:
                    continue

                # Resolve event
                if fixed_event is not None:
                    event = fixed_event
                else:
                    event_id_raw = row.get("event_type") or row.get("event", "")
                    event = event_map.get(event_id_raw)
                    if event is None:
                        # Try fuzzy matching by substring
                        for eid, emeta in event_map.items():
                            if eid in event_id_raw or event_id_raw in eid:
                                event = emeta
                                break
                    if event is None:
                        # Create an unknown event placeholder
                        event = _EventMeta(
                            event_id=event_id_raw or "unknown",
                            disaster_type="unknown",
                            region="unknown",
                            year=0,
                        )

                rows.append(
                    {
                        "sample_id": str(tweet_id) if tweet_id else f"{event.event_id}_{len(rows)}",
                        "event_id": event.event_id,
                        "disaster_type": event.disaster_type,
                        "region": event.region,
                        "year": event.year,
                        "modality": "text",
                        "image_path": None,
                        "text": text.strip() if text else None,
                        "label": label.strip().lower(),
                        "label_task": "humanitarian",
                    }
                )
        return rows

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
        seed: int = 42,
    ) -> "HumAIDDataset":
        """Create a fake HumAID dataset for testing.

        Parameters
        ----------
        n_per_event : int
            Samples per event.
        output_dir : str | Path | None
            Directory for synthetic data.  Defaults to a temp directory.
        seed : int
            Random seed.

        Returns
        -------
        HumAIDDataset
        """
        import tempfile

        rng = random.Random(seed)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="humaid_dummy_"))
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for event in _EVENTS:
            event_dir = output_dir / event.event_id
            event_dir.mkdir(parents=True, exist_ok=True)

            rows: List[Dict[str, str]] = []
            for i in range(n_per_event):
                words = [
                    "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
                    for _ in range(rng.randint(8, 20))
                ]
                rows.append(
                    {
                        "tweet_id": f"{event.event_id}_{i:04d}",
                        "tweet_text": " ".join(words),
                        "label": rng.choice(_HUMANITARIAN_LABELS),
                    }
                )

            tsv_path = event_dir / f"{event.event_id}_train.tsv"
            with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["tweet_id", "tweet_text", "label"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)

        return cls(root=output_dir)
