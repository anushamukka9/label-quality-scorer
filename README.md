# label-quality-scorer

Score **annotation quality** for classification datasets: inter-annotator
agreement, annotator reliability, disagreement hotspots, class-imbalance
diagnostics, suspicious-label detection — and a **ranked review queue** so
adjudicators triage the riskiest items first.

Bad labels silently cap model quality. `label-quality-scorer` answers the
questions annotation leads actually ask: *do my raters agree?* (Cohen's /
Fleiss' kappa), *which items are they fighting over?* (label entropy),
*which raters can I trust?* (reliability profiles), *is the class balance
sane?* (imbalance diagnostics), and *which labels would a model disagree
with?* (cross-validated self-consistency flags).

## Install

```bash
pip install label-quality-scorer        # (once published)
# or from source:
pip install .
```

Requires Python ≥ 3.9 and `numpy` only.

## Quickstart

```bash
# Score a labeled CSV (item_id, annotator, label columns):
label-quality-scorer examples/annotations.csv -o report.json

# Add item features to enable the self-consistency check:
label-quality-scorer examples/annotations.csv \
  --features examples/item_features.csv -o report.json
```

```python
from label_quality_scorer import score_annotations

records = [
    {"item_id": "t1", "annotator": "ana", "label": "pos"},
    {"item_id": "t1", "annotator": "ben", "label": "pos"},
    {"item_id": "t2", "annotator": "ana", "label": "neg"},
    {"item_id": "t2", "annotator": "ben", "label": "pos"},  # disagreement!
]

report = score_annotations(records)
print(report["summary"]["fleiss_kappa"])
print(report["review_queue"][0])   # riskiest item, with reasons
```

Or run the bundled example:

```bash
python examples/quickstart.py
```

## API

| Function | What it does |
|---|---|
| `score_annotations(records, features=None, config=None)` | Full pipeline → JSON-serializable report dict |
| `cohen_kappa(a, b)` | Pairwise rater agreement, chance-corrected |
| `fleiss_kappa(item_labels)` | Multi-rater agreement (3+ raters) |
| `pairwise_kappa_matrix(records)` | Cohen's kappa for every annotator pair |
| `percent_agreement(records)` | Mean per-item raw agreement |
| `item_entropy(labels)` | Normalized label entropy for one item |
| `disagreement_hotspots(records, top_n=20)` | Items ranked by rater disagreement |
| `annotator_profiles(records)` | Per-rater reliability scores + bias diagnostics |
| `class_distribution(labels)` | Counts, imbalance ratio, Gini, minority classes |
| `self_consistency_flags(items, features, n_folds=5)` | k-fold CV suspicious-label flags |
| `build_review_queue(records, ...)` | Composite-risk ranked triage list |

See [`docs/usage.md`](docs/usage.md) for the full guide: input formats,
report fields, tuning, and how to read kappa values.

## Architecture

```
src/label_quality_scorer/
├── agreement.py        # Cohen's kappa, Fleiss' kappa, pairwise matrix
├── entropy.py          # per-item label entropy, disagreement hotspots
├── reliability.py      # annotator reliability profiles
├── imbalance.py        # class distribution diagnostics
├── self_consistency.py # k-fold CV suspicious-label detection (numpy only)
├── review_queue.py     # composite-risk ranked triage list
├── report.py           # score_annotations() + LabelQualityConfig
└── cli.py              # `label-quality-scorer` console script
```

Design notes:

- **Long-format input** — one row per (item, annotator) rating, so rater
  counts can vary per item.
- **Deterministic everywhere** — majority-vote ties break by sort order,
  CV folds are stratified round-robin; no RNG, no flaky outputs.
- **Honest missing values** — unmeasurable kappas are `None`, not zero;
  single-rater items skip agreement math but still feed imbalance stats.
- **Light dependency footprint** — `numpy` only. The self-consistency
  classifier is a nearest-centroid model implemented in ~30 lines, so the
  whole tool stays auditable.

## Example report (CLI summary)

```
=== label-quality-scorer ===
items: 15 | annotators: 3 | annotations: 42
percent agreement: 0.733
Fleiss' kappa: 0.564
class imbalance ratio: 2.333
suspicious labels (self-consistency): 1

Top review queue items:
   1. t011  risk=0.73  majority=negative  high rater disagreement (entropy 0.92); model disagrees with consensus label (plausibility 0.00)
```

## License

MIT — Copyright (c) 2026 Anusha Mukka. See [LICENSE](LICENSE).

Author: [Anusha Mukka](https://anushamukka.com)
