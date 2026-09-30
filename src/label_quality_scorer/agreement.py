"""Inter-annotator agreement metrics: Cohen's kappa, Fleiss' kappa, and helpers.

Records are dicts with keys ``item_id``, ``annotator``, and ``label``.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np


def group_by_item(records):
    """Map item_id -> list of (annotator, label) for that item."""
    groups = defaultdict(list)
    for r in records:
        groups[r["item_id"]].append((r["annotator"], r["label"]))
    return dict(groups)


def majority_label(labels):
    """Deterministic majority vote; ties broken by sorted label order."""
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    best = max(counts.values())
    return sorted(c for c, n in counts.items() if n == best)[0]


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two aligned label lists.

    Returns 1.0 for perfect agreement, 0.0 for chance-level agreement.
    Returns 0.0 (not NaN) when a rater shows no variation - kappa is
    undefined there, and 0.0 is the honest "no measurable agreement" value.
    """
    a = list(labels_a)
    b = list(labels_b)
    if len(a) != len(b) or not a:
        raise ValueError("cohen_kappa needs two non-empty, equal-length lists")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    classes = sorted(set(a) | set(b))
    expected = sum(
        (sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n)
        for c in classes
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(item_labels):
    """Fleiss' kappa for 3+ raters.

    ``item_labels`` is a list of per-item label lists (one entry per rating).
    Handles items with differing rater counts via the per-item formulation.
    Items with fewer than 2 ratings are ignored.
    """
    groups = [list(g) for g in item_labels if len(g) >= 2]
    if not groups:
        raise ValueError("fleiss_kappa needs at least one item with 2+ ratings")
    classes = sorted({lab for g in groups for lab in g})
    n_items = len(groups)
    p_item = []
    class_totals = {c: 0 for c in classes}
    total_ratings = 0
    for g in groups:
        n = len(g)
        counts = {c: 0 for c in classes}
        for lab in g:
            counts[lab] += 1
            class_totals[lab] += 1
        total_ratings += n
        p_item.append((sum(v * v for v in counts.values()) - n) / (n * (n - 1)))
    p_bar = float(np.mean(p_item))
    p_expected = sum((class_totals[c] / total_ratings) ** 2 for c in classes)
    if p_expected >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_expected) / (1.0 - p_expected)


def krippendorff_alpha(item_labels):
    """Krippendorff's alpha (nominal metric) for 2+ raters.

    ``item_labels`` is a list of per-item label lists (one entry per rating).
    Unlike Fleiss' kappa this handles missing ratings naturally: items may
    have differing rater counts, and items with fewer than 2 ratings are
    ignored. Good for the common case where not every annotator rated every
    item.

    Returns 1.0 for perfect agreement, 0.0 for chance-level agreement, and
    0.0 (not NaN) when the labels show no variation to measure.
    """
    units = [list(g) for g in item_labels if len(g) >= 2]
    if not units:
        raise ValueError(
            "krippendorff_alpha needs at least one item with 2+ ratings"
        )
    values = sorted({lab for g in units for lab in g}, key=str)
    if len(values) < 2:
        return 0.0  # no variation: nothing to measure

    # Coincidence matrix: for each unit with m ratings, every ordered pair
    # of distinct positions contributes 1/(m - 1) to its (value, value) cell.
    coinc = {c: {k: 0.0 for k in values} for c in values}
    for g in units:
        m = len(g)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[g[i]][g[j]] += 1.0 / (m - 1)

    marginal = {c: sum(coinc[c].values()) for c in values}
    total = sum(marginal.values())
    off_diag = sum(
        coinc[c][k] for c in values for k in values if c != k
    )
    # Nominal distance: 0 on the diagonal, 1 off it.
    do = off_diag / total
    de = (
        sum(marginal[c] * marginal[k] for c in values for k in values if c != k)
        / (total * (total - 1))
    )
    if de <= 0:
        return 0.0
    return 1.0 - do / de


def pairwise_kappa_matrix(records):
    """Mean Cohen's kappa for every annotator pair over their shared items.

    Returns a dict {(annotator_a, annotator_b): kappa or None}.
    ``None`` means the pair shares fewer than 2 items or has no label
    variation (kappa undefined) - treated as "cannot measure", not as zero.
    """
    by_annotator = defaultdict(dict)
    for r in records:
        by_annotator[r["annotator"]][r["item_id"]] = r["label"]
    annotators = sorted(by_annotator)
    matrix = {}
    for a1, a2 in combinations(annotators, 2):
        shared = sorted(set(by_annotator[a1]) & set(by_annotator[a2]))
        if len(shared) < 2:
            matrix[(a1, a2)] = None
            continue
        la = [by_annotator[a1][i] for i in shared]
        lb = [by_annotator[a2][i] for i in shared]
        if len(set(la)) < 2 or len(set(lb)) < 2:
            matrix[(a1, a2)] = None
            continue
        matrix[(a1, a2)] = cohen_kappa(la, lb)
    return matrix


def percent_agreement(records):
    """Mean per-item percent agreement across all items with 2+ ratings."""
    groups = group_by_item(records)
    scores = []
    for pairs in groups.values():
        labels = [lab for _, lab in pairs]
        if len(labels) < 2:
            continue
        counts = defaultdict(int)
        for lab in labels:
            counts[lab] += 1
        n = len(labels)
        scores.append(
            sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))
        )
    return float(np.mean(scores)) if scores else float("nan")
