"""Quickstart: score the bundled example annotations end to end.

Runs from the repo root:
    python examples/quickstart.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from label_quality_scorer import score_annotations  # noqa: E402

HERE = os.path.dirname(__file__)


def load_records(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            {"item_id": r["item_id"], "annotator": r["annotator"], "label": r["label"]}
            for r in csv.DictReader(fh)
        ]


def load_features(path):
    features = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        feat_cols = [c for c in reader.fieldnames if c != "item_id"]
        for row in reader:
            features[row["item_id"]] = [float(row[c]) for c in feat_cols]
    return features


def main():
    records = load_records(os.path.join(HERE, "annotations.csv"))
    features = load_features(os.path.join(HERE, "item_features.csv"))

    report = score_annotations(records, features=features)
    s = report["summary"]
    print("=== label-quality-scorer quickstart ===")
    print(f"items={s['n_items']} annotators={s['n_annotators']} "
          f"annotations={s['n_annotations']}")
    print(f"percent agreement : {s['percent_agreement']:.3f}")
    print(f"Fleiss' kappa     : {s['fleiss_kappa']:.3f}")
    print(f"imbalance ratio   : {s['imbalance_ratio']}")
    print(f"suspicious labels : {s['suspicious_labels']}")

    print("\nannotator reliability:")
    for name, prof in report["annotator_profiles"].items():
        print(f"  {name:6s} reliability={prof['reliability_score']:.2f} "
              f"majority-agree={prof['agreement_with_majority']} "
              f"kappa={prof['mean_pairwise_kappa']}")

    print("\ntop review queue:")
    for i, item in enumerate(report["review_queue"][:5], 1):
        print(f"  {i}. {item['item_id']} risk={item['risk_score']:.2f} "
              f"-> {'; '.join(item['reasons'])}")

    out = os.path.join(HERE, "quickstart_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
