"""Models package for disaster AI audit benchmark.

Provides probe classifiers, event-prior baselines, domain generalization
trainers, and zero-shot VLM inference for evaluating multimodal disaster AI.
"""

from src.models.probes import LogisticProbe, MLPProbe
from src.models.event_prior import EventPriorBaseline
from src.models.domain_generalization import GroupDROTrainer, IRMTrainer, CORALTrainer
from src.models.vlm_zeroshot import ZeroShotVLM

__all__ = [
    "LogisticProbe",
    "MLPProbe",
    "EventPriorBaseline",
    "GroupDROTrainer",
    "IRMTrainer",
    "CORALTrainer",
    "ZeroShotVLM",
]
