"""Top-level scoring: one call that produces the full quality report."""

from dataclasses import dataclass, field

from .agreement import (
    fleiss_kappa,
    group_by_item,
    majority_label,
    pairwise_kappa_matrix,
    percent_agreement,
)
from .entropy import disagreement_hotspots
from .imbalance import class_distribution
from .reliability import annotator_profiles
from .review_queue import build_review_queue
from .self_consistency import self_consistency_flags


@dataclass
class LabelQualityConfig:
    """Tunable knobs for :func:`score_annotations`."""

    hotspot_top_n: int = 20
    review_queue_top_n: int = 50
    minority_threshold: float = 0.05
    cv_folds: int = 5
    review_weights: dict = field(
        default_factory=lambda: {
            "entropy": 0.4,
            "unreliable_annotators": 0.3,
            "inconsistency": 0.3,
        }
    )


def score_annotations(records, features=None, config=None):
    """Score a whole annotation collection; return a JSON-serializable report.

    ``records``: list of dicts with keys ``item_id``, ``annotator``, ``label``.
    ``features``: optional dict item_id -> numeric vector, enabling the
    self-consistency check against consensus (majority) labels.
    """
    config = config or LabelQualityConfig()
    records = list(records)
    if not records:
        raise ValueError("score_annotations needs at least one annotation record")
    for r in records:
        for key in ("item_id", "annotator", "label"):
            if key not in r:
                raise ValueError(f"record missing required key: {key!r}")

    groups = group_by_item(records)
    multi_rater = [
        [lab for _, lab in pairs] for pairs in groups.values() if len(pairs) >= 2
    ]

    agreement = {
        "n_items": len(groups),
        "n_annotators": len({r["annotator"] for r in records}),
        "n_annotations": len(records),
        "items_with_2plus_raters": len(multi_rater),
        "percent_agreement": percent_agreement(records),
        "fleiss_kappa": fleiss_kappa(multi_rater) if multi_rater else None,
        "pairwise_kappa": {
            f"{a}__{b}": v for (a, b), v in pairwise_kappa_matrix(records).items()
        },
    }

    profiles = annotator_profiles(records)
    hotspots = disagreement_hotspots(
        records, top_n=config.hotspot_top_n, min_raters=2
    )

    consensus = [
        (iid, majority_label([lab for _, lab in pairs]))
        for iid, pairs in groups.items()
    ]
    imbalance = class_distribution(
        [lab for _, lab in consensus],
        minority_threshold=config.minority_threshold,
    )

    consistency = None
    if features:
        consistency = self_consistency_flags(
            consensus, features, n_folds=config.cv_folds
        )

    reliability_map = {
        a: p["reliability_score"] for a, p in profiles.items()
    }
    queue = build_review_queue(
        records,
        annotator_reliability=reliability_map,
        consistency=consistency,
        top_n=config.review_queue_top_n,
        weights=config.review_weights,
    )

    n_suspicious = (
        sum(1 for v in consistency.values() if v["suspicious"])
        if consistency
        else 0
    )
    report = {
        "summary": {
            "n_items": agreement["n_items"],
            "n_annotators": agreement["n_annotators"],
            "n_annotations": agreement["n_annotations"],
            "percent_agreement": agreement["percent_agreement"],
            "fleiss_kappa": agreement["fleiss_kappa"],
            "imbalance_ratio": imbalance["imbalance_ratio"],
            "minority_classes": imbalance["minority_classes"],
            "suspicious_labels": n_suspicious,
            "review_queue_size": len(queue),
        },
        "agreement": agreement,
        "annotator_profiles": profiles,
        "disagreement_hotspots": hotspots,
        "class_distribution": imbalance,
        "self_consistency": consistency,
        "review_queue": queue,
    }
    return report
