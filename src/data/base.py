"""Base dataset class for disaster AI benchmark."""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np


@dataclass
class Sample:
    """A single sample in the standardized benchmark format.

    Every dataset loader converts its native format into this structure.
    """

    sample_id: str
    event_id: str
    disaster_type: str
    region: str
    year: int
    modality: Literal["image", "text", "both"]
    image_path: str | None
    text: str | None
    label: int | str
    label_task: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "sample_id": self.sample_id,
            "event_id": self.event_id,
            "disaster_type": self.disaster_type,
            "region": self.region,
            "year": self.year,
            "modality": self.modality,
            "image_path": self.image_path,
            "text": self.text,
            "label": self.label,
            "label_task": self.label_task,
        }


class DisasterDataset(abc.ABC):
    """Abstract base class for all disaster benchmark datasets.

    Subclasses must implement :meth:`_load_samples` which returns the raw
    list of :class:`Sample` objects, and must set the class-level metadata
    attributes.

    Parameters
    ----------
    root : str | Path
        Root directory where the dataset files reside.
    label_task : str | None
        Which labelling task to use (datasets may have multiple).
        ``None`` uses the dataset's default.
    """

    # -- Class-level metadata (override in subclasses) --
    name: str = ""
    default_label_task: str = ""
    available_label_tasks: tuple[str, ...] = ()
    modality: Literal["image", "text", "both"] = "both"

    def __init__(self, root: str | Path, label_task: str | None = None) -> None:
        self.root = Path(root)
        self.label_task = label_task or self.default_label_task
        self._samples: list[Sample] | None = None

    # -- Abstract ----------------------------------------------------------

    @abc.abstractmethod
    def _load_samples(self) -> list[Sample]:
        """Load all samples from disk. Must be implemented by subclasses."""
        ...

    # -- Public API --------------------------------------------------------

    @property
    def samples(self) -> list[Sample]:
        """Lazily-loaded list of :class:`Sample`."""
        if self._samples is None:
            self._samples = self._load_samples()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all samples as a list of standardized dicts."""
        return [s.to_dict() for s in self.samples]

    @property
    def event_ids(self) -> list[str]:
        """Unique, sorted event identifiers in this dataset."""
        return sorted({s.event_id for s in self.samples})

    @property
    def labels(self) -> np.ndarray:
        """Label array (integer-encoded)."""
        return np.array([s.label for s in self.samples])

    @property
    def sample_event_ids(self) -> list[str]:
        """Per-sample event identifiers (same order as samples)."""
        return [s.event_id for s in self.samples]

    def get_label_map(self) -> dict[int | str, int]:
        """Map raw labels to contiguous integers."""
        unique = sorted(set(s.label for s in self.samples))
        return {lbl: i for i, lbl in enumerate(unique)}

    def encode_labels(self) -> np.ndarray:
        """Return labels as contiguous integer array."""
        lmap = self.get_label_map()
        return np.array([lmap[s.label] for s in self.samples])

    # -- Dummy data generation ---------------------------------------------

    @classmethod
    def generate_dummy(
        cls,
        n_per_event: int = 50,
        output_dir: str | Path = "/tmp/dummy_disaster_data",
    ) -> "DisasterDataset":
        """Generate a dummy dataset for testing.

        Subclasses should override this to produce realistic dummy data.
        The default implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{cls.__name__}.generate_dummy() not implemented. "
            "Override in subclass."
        )

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _make_sample_id(dataset_name: str, *components: str) -> str:
        """Deterministic sample ID from dataset name and components."""
        raw = f"{dataset_name}__{'__'.join(str(c) for c in components)}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        n = len(self) if self._samples is not None else "?"
        return (
            f"{self.__class__.__name__}(root='{self.root}', "
            f"label_task='{self.label_task}', n_samples={n})"
        )
