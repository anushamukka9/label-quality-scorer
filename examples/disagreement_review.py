"""Disagreement review: agreement metrics, annotator ranking, and a rater-level
disagreement report on the bundled example annotations.

Runs from the repo root:
    python examples/disagreement_review.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from label_quality_scorer import (  # noqa: E402
    disagreement_report,
    fleiss_kappa,
    krippendorff_alpha,
    percent_agreement,
    rank_annotators,
)
from label_quality_scorer.agreement import group_by_item  # noqa: E402

HERE = os.path.dirname(__file__)


def load_records(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            {"item_id": r["item_id"], "annotator": r["annotator"], "label": r["label"]}
            for r in csv.DictReader(fh)
        ]


def main():
    records = load_records(os.path.join(HERE, "annotations.csv"))
    groups = group_by_item(records)
    multi = [[lab for _, lab in pairs] for pairs in groups.values() if len(pairs) >= 2]

    print("=== agreement metrics ===")
    print(f"percent agreement    : {percent_agreement(records):.3f}")
    print(f"Fleiss' kappa        : {fleiss_kappa(multi):.3f}")
    print(f"Krippendorff's alpha : {krippendorff_alpha(multi):.3f}")

    print("\n=== annotator reliability ranking ===")
    for rank, (name, score, prof) in enumerate(rank_annotators(records), 1):
        print(
            f"  {rank}. {name:6s} score={score:.2f} "
            f"items={prof['n_items']} "
            f"majority-agree={prof['agreement_with_majority']} "
            f"pairwise-kappa={prof['mean_pairwise_kappa']}"
        )

    print("\n=== top disagreement items (who said what) ===")
    for row in disagreement_report(records, top_n=5):
        ratings = ", ".join(f"{a}={lab}" for a, lab in sorted(row["ratings"].items()))
        print(
            f"  {row['item_id']}: entropy={row['entropy']:.2f} "
            f"pairwise-agree={row['pairwise_agreement']:.2f} "
            f"majority={row['majority_label']}  [{ratings}]"
        )


if __name__ == "__main__":
    main()
