"""label-quality-scorer: score annotation quality for classification datasets.

Measures inter-annotator agreement (Cohen's / Fleiss' kappa), label-entropy
disagreement hotspots, annotator reliability profiles, class-imbalance
diagnostics, and cross-validation-style self-consistency flags — then rolls
everything into a ranked review queue so annotators can triage the riskiest
items first.

See https://anushamukka.com for the author.
"""

from .agreement import (
    cohen_kappa,
    fleiss_kappa,
    pairwise_kappa_matrix,
    percent_agreement,
)
from .entropy import item_entropy, disagreement_hotspots
from .reliability import annotator_profiles
from .imbalance import class_distribution
from .self_consistency import self_consistency_flags
from .review_queue import build_review_queue
from .report import score_annotations, LabelQualityConfig

__version__ = "1.0.0"

__all__ = [
    "LabelQualityConfig",
    "annotator_profiles",
    "build_review_queue",
    "class_distribution",
    "cohen_kappa",
    "disagreement_hotspots",
    "fleiss_kappa",
    "item_entropy",
    "pairwise_kappa_matrix",
    "percent_agreement",
    "score_annotations",
    "self_consistency_flags",
]
