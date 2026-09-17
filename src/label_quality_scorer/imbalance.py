"""Class-imbalance diagnostics for a labeled collection."""

from collections import Counter

import numpy as np


def class_distribution(labels, minority_threshold=0.05):
    """Summarize the class balance of a label list.

    Returns counts, proportions, imbalance_ratio (largest / smallest class),
    a Gini coefficient (0 = perfectly balanced, ->1 = one class dominates),
    effective number of classes (exp of Shannon entropy), and the list of
    minority classes below ``minority_threshold`` share.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("class_distribution needs at least one label")
    counts = Counter(labels)
    total = len(labels)
    classes = sorted(counts)
    proportions = {c: counts[c] / total for c in classes}

    smallest = min(counts.values())
    imbalance_ratio = max(counts.values()) / smallest if smallest else float("inf")

    # Gini over class shares: 0 = perfectly even, approaches 1 = skewed.
    shares = np.array([proportions[c] for c in classes])
    gini = float((np.abs(np.subtract.outer(shares, shares)).sum()) / (2 * len(shares)))

    entropy = float(-np.sum(shares * np.log(shares)))
    effective_classes = float(np.exp(entropy))

    minority = [c for c in classes if proportions[c] < minority_threshold]

    return {
        "n_items": total,
        "n_classes": len(classes),
        "counts": dict(counts),
        "proportions": {c: round(p, 4) for c, p in proportions.items()},
        "imbalance_ratio": round(imbalance_ratio, 3),
        "gini": round(gini, 4),
        "effective_n_classes": round(effective_classes, 3),
        "minority_classes": minority,
    }
