"""Ranked review queue: turn every signal into one triage list."""

from .agreement import group_by_item, majority_label
from .entropy import item_entropy


def build_review_queue(
    records,
    annotator_reliability=None,
    consistency=None,
    top_n=50,
    weights=None,
):
    """Rank items by a composite risk score (highest risk first).

    Risk = w_entropy * normalized_entropy
         + w_unreliable * (1 - mean annotator reliability on the item)
         + w_inconsistent * (1 - self-consistency plausibility)

    ``annotator_reliability``: dict annotator -> 0..1 score (from
    :func:`annotator_profiles`); missing annotators count as 0.5.
    ``consistency``: dict item_id -> {"plausibility": ...} (from
    :func:`self_consistency_flags`); missing items count as 1.0.

    Returns a list of dicts with item_id, risk_score, and a reasons list.
    """
    weights = weights or {}
    w_ent = weights.get("entropy", 0.4)
    w_rel = weights.get("unreliable_annotators", 0.3)
    w_con = weights.get("inconsistency", 0.3)

    groups = group_by_item(records)
    queue = []
    for item_id, pairs in groups.items():
        labels = [lab for _, lab in pairs]
        entropy = item_entropy(labels)

        if annotator_reliability:
            rels = [
                annotator_reliability.get(a, 0.5) for a, _ in pairs
            ]
            mean_rel = sum(rels) / len(rels)
        else:
            mean_rel = 0.5

        plaus = 1.0
        if consistency and item_id in consistency:
            plaus = consistency[item_id].get("plausibility", 1.0)

        risk = w_ent * entropy + w_rel * (1 - mean_rel) + w_con * (1 - plaus)

        reasons = []
        if entropy >= 0.7:
            reasons.append(f"high rater disagreement (entropy {entropy:.2f})")
        elif entropy >= 0.4:
            reasons.append(f"moderate rater disagreement (entropy {entropy:.2f})")
        if mean_rel < 0.5:
            reasons.append(
                f"rated by low-reliability annotators (mean {mean_rel:.2f})"
            )
        if plaus < 0.5:
            reasons.append(
                f"model disagrees with consensus label (plausibility {plaus:.2f})"
            )
        if not reasons:
            reasons.append("no strong risk signals")

        queue.append(
            {
                "item_id": item_id,
                "risk_score": round(risk, 4),
                "n_raters": len(pairs),
                "majority_label": majority_label(labels),
                "entropy": round(entropy, 4),
                "mean_annotator_reliability": round(mean_rel, 4),
                "label_plausibility": round(plaus, 4),
                "reasons": reasons,
            }
        )
    queue.sort(key=lambda r: (-r["risk_score"], str(r["item_id"])))
    return queue[:top_n] if top_n else queue
