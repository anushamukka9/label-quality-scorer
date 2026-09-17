"""CLI: score a labeled CSV/JSONL file and emit a JSON quality report."""

import argparse
import csv
import json
import sys

from .report import LabelQualityConfig, score_annotations


def _load_annotations(path, item_col, annotator_col, label_col):
    records = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(
                    {
                        "item_id": obj[item_col],
                        "annotator": obj[annotator_col],
                        "label": obj[label_col],
                    }
                )
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = {item_col, annotator_col, label_col} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"columns missing from {path}: {sorted(missing)} "
                    f"(found: {reader.fieldnames})"
                )
            for row in reader:
                records.append(
                    {
                        "item_id": row[item_col],
                        "annotator": row[annotator_col],
                        "label": row[label_col],
                    }
                )
    if not records:
        raise ValueError(f"no annotation records found in {path}")
    return records


def _load_features(path, item_col="item_id"):
    """Load item_id -> feature vector from a CSV (all non-id columns numeric)."""
    features = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if item_col not in (reader.fieldnames or []):
            raise ValueError(
                f"item column {item_col!r} not found in {path} "
                f"(found: {reader.fieldnames})"
            )
        feature_cols = [c for c in reader.fieldnames if c != item_col]
        if not feature_cols:
            raise ValueError(f"no feature columns found in {path}")
        for row in reader:
            try:
                vec = [float(row[c]) for c in feature_cols]
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric feature value in {path}: {exc}"
                ) from exc
            features[row[item_col]] = vec
    return features


def _print_summary(report):
    s = report["summary"]
    print("=== label-quality-scorer ===")
    print(f"items: {s['n_items']} | annotators: {s['n_annotators']} "
          f"| annotations: {s['n_annotations']}")
    pa = s["percent_agreement"]
    fk = s["fleiss_kappa"]
    print(f"percent agreement: {pa:.3f}" if pa == pa else "percent agreement: n/a")
    print(f"Fleiss' kappa: {fk:.3f}" if fk is not None else "Fleiss' kappa: n/a "
          "(need 2+ raters per item)")
    print(f"class imbalance ratio: {s['imbalance_ratio']}")
    if s["minority_classes"]:
        print(f"minority classes: {', '.join(map(str, s['minority_classes']))}")
    print(f"suspicious labels (self-consistency): {s['suspicious_labels']}")
    print("\nTop review queue items:")
    for i, item in enumerate(report["review_queue"][:10], 1):
        print(f"  {i:2d}. {item['item_id']}  risk={item['risk_score']:.2f}  "
              f"majority={item['majority_label']}  "
              f"{'; '.join(item['reasons'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="label-quality-scorer",
        description="Score annotation quality: agreement, reliability, "
                    "imbalance, suspicious labels, and a ranked review queue.",
    )
    parser.add_argument("input", help="annotations file (CSV or JSONL)")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--annotator-col", default="annotator")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--features",
        default=None,
        help="CSV of item features (item_id + numeric columns) to enable "
             "the cross-validated self-consistency check",
    )
    parser.add_argument(
        "-o", "--output", default="label_quality_report.json",
        help="where to write the JSON report",
    )
    parser.add_argument("--top-n", type=int, default=50,
                        help="review queue length")
    args = parser.parse_args(argv)

    try:
        records = _load_annotations(
            args.input, args.item_col, args.annotator_col, args.label_col
        )
        features = (
            _load_features(args.features) if args.features else None
        )
        config = LabelQualityConfig(review_queue_top_n=args.top_n)
        report = score_annotations(records, features=features, config=config)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    _print_summary(report)
    print(f"\nfull report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
