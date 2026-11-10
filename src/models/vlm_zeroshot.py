"""Zero-shot classification with open Vision-Language Models via open_clip.

Loads a CLIP / SigLIP / other open_clip-compatible model, encodes images and
text prompts, and classifies by cosine similarity between image embeddings and
class-level text embeddings.

Custom prompt templates (e.g. ``"a satellite image of {class_name}"``) are
supported and averaged when multiple templates are provided per class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ImageListDataset(Dataset):
    """Thin dataset that applies an open_clip transform to a list of images."""

    def __init__(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
        transform: object,
    ) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        return self.transform(img)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ZeroShotVLM:
    """Zero-shot image classification via open_clip cosine similarity.

    Parameters
    ----------
    model_name : str
        Name of the model in the ``open_clip`` registry
        (e.g. ``"ViT-L-14"``).
    pretrained : str
        Pretrained weights tag (e.g. ``"openai"``, ``"laion2b_s32b_b82k"``).
    device : str
        PyTorch device string (default ``"cpu"``).
    batch_size : int
        Batch size for image encoding (default ``64``).
    prompt_templates : list[str] | None
        List of prompt template strings, each containing ``{class_name}``
        as a placeholder.  When ``None`` the default template
        ``"a photo of a {class_name}"`` is used.  Multiple templates are
        ensembled by averaging the resulting text embeddings per class.
    """

    DEFAULT_TEMPLATES: list[str] = [
        "a photo of a {class_name}",
    ]

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
        batch_size: int = 64,
        prompt_templates: Optional[list[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.prompt_templates = prompt_templates or self.DEFAULT_TEMPLATES

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

        # Cached text embeddings — populated by build_class_embeddings()
        self._class_names: list[str] = []
        self._text_embeddings: Optional[torch.Tensor] = None  # (n_classes, D)

    # ------------------------------------------------------------------
    # Text embedding construction
    # ------------------------------------------------------------------

    def build_class_embeddings(self, class_names: list[str]) -> None:
        """Pre-compute normalised text embeddings for each class.

        Parameters
        ----------
        class_names : list[str]
            Human-readable class names.  The order determines the class index
            used in ``predict`` / ``predict_proba`` outputs.
        """
        self._class_names = class_names
        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for name in class_names:
                prompts = [t.format(class_name=name) for t in self.prompt_templates]
                tokens = self._tokenizer(prompts).to(self.device)
                text_feats = self._model.encode_text(tokens)  # (T, D)
                text_feats = F.normalize(text_feats, dim=-1)
                # Average across templates, then re-normalise
                mean_feat = text_feats.mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=-1)
                all_embeddings.append(mean_feat)

        self._text_embeddings = torch.stack(all_embeddings, dim=0)  # (C, D)

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_images(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
    ) -> torch.Tensor:
        """Encode images into normalised CLIP embedding space.

        Parameters
        ----------
        images : sequence of paths or PIL Images
            Images to encode.

        Returns
        -------
        torch.Tensor of shape ``(n_images, D)``
            L2-normalised image embeddings.
        """
        dataset = _ImageListDataset(images, self._preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_feats: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                feats = self._model.encode_image(batch)
                feats = F.normalize(feats, dim=-1)
                all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
    ) -> NDArray[np.integer]:
        """Classify images by argmax cosine similarity.

        Parameters
        ----------
        images : sequence of paths or PIL Images

        Returns
        -------
        ndarray of shape ``(n_images,)``
            Integer class indices corresponding to the class_names order
            passed to :meth:`build_class_embeddings`.

        Raises
        ------
        RuntimeError
            If :meth:`build_class_embeddings` has not been called.
        """
        sims = self._similarities(images)
        return sims.argmax(axis=1)

    def predict_proba(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
        temperature: float = 1.0,
    ) -> NDArray[np.floating]:
        """Return softmax-calibrated class probabilities.

        Parameters
        ----------
        images : sequence of paths or PIL Images
        temperature : float
            Softmax temperature; lower values sharpen the distribution.

        Returns
        -------
        ndarray of shape ``(n_images, n_classes)``
        """
        sims = self._similarities(images)
        # Softmax over similarity scores
        exp = np.exp((sims - sims.max(axis=1, keepdims=True)) / temperature)
        return exp / exp.sum(axis=1, keepdims=True)

    def _similarities(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
    ) -> NDArray[np.floating]:
        """Compute cosine similarities between images and class embeddings."""
        if self._text_embeddings is None:
            raise RuntimeError(
                "Text embeddings not built. Call build_class_embeddings() first."
            )
        image_feats = self.encode_images(images)  # (N, D)
        # Cosine similarity = dot product of normalised vectors
        sims = (image_feats @ self._text_embeddings.T).numpy()  # (N, C)
        return sims
