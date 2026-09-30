"""label-quality-scorer: score annotation quality for classification datasets.

Measures inter-annotator agreement (Cohen's / Fleiss' kappa, Krippendorff's
alpha), label-entropy disagreement hotspots, annotator reliability profiles, class-imbalance
diagnostics, and cross-validation-style self-consistency flags - then rolls
everything into a ranked review queue so annotators can triage the riskiest
items first.

See https://anushamukka.com for the author.
"""

from .agreement import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    pairwise_kappa_matrix,
    percent_agreement,
)
from .entropy import disagreement_report, item_entropy, disagreement_hotspots
from .reliability import annotator_profiles, rank_annotators
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
    "disagreement_report",
    "fleiss_kappa",
    "item_entropy",
    "krippendorff_alpha",
    "pairwise_kappa_matrix",
    "percent_agreement",
    "rank_annotators",
    "score_annotations",
    "self_consistency_flags",
]
