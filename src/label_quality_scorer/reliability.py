"""Annotator reliability profiles: how much each rater can be trusted."""

from collections import defaultdict

import numpy as np

from .agreement import cohen_kappa, group_by_item, majority_label
from .entropy import item_entropy


def _annotator_pairwise_kappas(records, annotator):
    by_annotator = defaultdict(dict)
    for r in records:
        by_annotator[r["annotator"]][r["item_id"]] = r["label"]
    mine = by_annotator[annotator]
    kappas = []
    for other, theirs in by_annotator.items():
        if other == annotator:
            continue
        shared = sorted(set(mine) & set(theirs))
        if len(shared) < 2:
            continue
        la = [mine[i] for i in shared]
        lb = [theirs[i] for i in shared]
        if len(set(la)) < 2 or len(set(lb)) < 2:
            continue
        kappas.append(cohen_kappa(la, lb))
    return kappas


def annotator_profiles(records):
    """Build a reliability profile per annotator.

    Metrics per annotator:
      - n_items: items they rated
      - agreement_with_majority: fraction of their ratings matching the item
        majority label (ties resolved deterministically)
      - mean_pairwise_kappa: mean Cohen's kappa vs. every other annotator
        over shared items (None when not measurable)
      - label_entropy: entropy of their own label distribution — near 0.0
        means they stamp one label on everything
      - top_class_bias: their most-used class's share minus the global share
        (positive = they overuse that class relative to the pool)
      - reliability_score: 0..1 blend of majority agreement (50%) and mean
        pairwise kappa rescaled to 0..1 (50%); kappa of 1.0 maps to 1.0,
        kappa of 0.0 maps to 0.5, negatives clamp toward 0.
    """
    groups = group_by_item(records)
    global_counts = defaultdict(int)
    total = 0
    for pairs in groups.values():
        for _, lab in pairs:
            global_counts[lab] += 1
            total += 1

    by_annotator = defaultdict(list)
    for r in records:
        by_annotator[r["annotator"]].append(r)

    profiles = {}
    for annotator, recs in sorted(by_annotator.items()):
        labels = [r["label"] for r in recs]
        own_counts = defaultdict(int)
        for lab in labels:
            own_counts[lab] += 1

        matches = 0
        counted = 0
        for r in recs:
            item_labels = [lab for _, lab in groups[r["item_id"]]]
            if len(item_labels) >= 2:
                counted += 1
                if r["label"] == majority_label(item_labels):
                    matches += 1
        maj_agree = matches / counted if counted else float("nan")

        kappas = _annotator_pairwise_kappas(records, annotator)
        mean_kappa = float(np.mean(kappas)) if kappas else None

        entropy = item_entropy(labels)
        top_class, top_n = max(own_counts.items(), key=lambda kv: (kv[1], kv[0]))
        bias = (top_n / len(labels)) - (global_counts[top_class] / total)

        kappa_part = (
            max(0.0, min(1.0, (mean_kappa + 1.0) / 2.0))
            if mean_kappa is not None
            else 0.5
        )
        maj_part = maj_agree if not np.isnan(maj_agree) else 0.5
        reliability = round(0.5 * maj_part + 0.5 * kappa_part, 4)

        profiles[annotator] = {
            "n_items": len(recs),
            "agreement_with_majority": round(maj_agree, 4)
            if not np.isnan(maj_agree)
            else None,
            "mean_pairwise_kappa": round(mean_kappa, 4)
            if mean_kappa is not None
            else None,
            "label_entropy": round(entropy, 4),
            "top_class": top_class,
            "top_class_bias": round(bias, 4),
            "reliability_score": reliability,
        }
    return profiles
