"""Label-entropy disagreement hotspots: find the items raters fight over most."""

from collections import defaultdict

import numpy as np

from .agreement import group_by_item


def item_entropy(labels, normalize=True):
    """Shannon entropy (bits) of the label distribution for one item.

    0.0 = unanimous; with ``normalize=True`` the maximum is 1.0 (a uniform
    spread over the observed classes).
    """
    labels = list(labels)
    if not labels:
        return 0.0
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    n = len(labels)
    probs = np.array([v / n for v in counts.values()])
    entropy = float(-np.sum(probs * np.log2(probs)))
    if normalize and len(counts) > 1:
        entropy /= np.log2(len(counts))
    return entropy


def disagreement_hotspots(records, top_n=20, min_raters=2):
    """Rank items by normalized label entropy (highest first).

    Returns a list of dicts: item_id, entropy, n_raters, label_counts,
    majority_label.
    """
    from .agreement import majority_label

    groups = group_by_item(records)
    rows = []
    for item_id, pairs in groups.items():
        labels = [lab for _, lab in pairs]
        if len(labels) < min_raters:
            continue
        counts = defaultdict(int)
        for lab in labels:
            counts[lab] += 1
        rows.append(
            {
                "item_id": item_id,
                "entropy": item_entropy(labels),
                "n_raters": len(labels),
                "label_counts": dict(counts),
                "majority_label": majority_label(labels),
            }
        )
    rows.sort(key=lambda r: (-r["entropy"], -r["n_raters"], str(r["item_id"])))
    return rows[:top_n] if top_n else rows
