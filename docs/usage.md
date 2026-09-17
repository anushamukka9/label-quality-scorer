# Usage guide — label-quality-scorer

This guide covers the full workflow: preparing data, running the CLI,
interpreting the report, and using the Python API directly.

## 1. Preparing your annotations

The core input is a table of individual ratings — one row per
(annotator, item) pair:

| column     | meaning                              |
|------------|--------------------------------------|
| `item_id`  | unique id of the item being labeled  |
| `annotator`| id of the person (or model) labeling |
| `label`    | the label they assigned              |

CSV and JSONL are both accepted. For JSONL, each line is a JSON object
with the same three keys.

```jsonl
{"item_id": "t001", "annotator": "ana", "label": "positive"}
{"item_id": "t001", "annotator": "ben", "label": "positive"}
```

Items may have different numbers of raters; single-rater items still count
toward class-imbalance stats but are excluded from agreement metrics.

### Optional: item features

To enable the cross-validated self-consistency check, supply a second CSV
mapping each `item_id` to numeric features (embeddings, text stats,
sensor readings — anything fixed-length and numeric):

```csv
item_id,len_chars,exclaim_count,sentiment_lexicon
t001,120,2,0.82
```

Features are z-scored internally, so mixed scales are fine.

## 2. Running the CLI

```bash
# Basic scoring
label-quality-scorer annotations.csv -o report.json

# With self-consistency check
label-quality-scorer annotations.csv --features item_features.csv -o report.json

# JSONL input, custom column names, longer review queue
label-quality-scorer ratings.jsonl \
  --item-col doc_id --annotator-col rater --label-col verdict \
  --top-n 100 -o report.json
```

The CLI prints a summary and the top-10 review items, and writes the full
JSON report to `-o`.

## 3. Reading the report

- **`summary`** — headline numbers: counts, percent agreement, Fleiss'
  kappa, imbalance ratio, minority classes, suspicious-label count.
- **`agreement`** — overall percent agreement, Fleiss' kappa across all
  items with 2+ raters, and the pairwise Cohen's kappa matrix
  (`null` = pair shares too few items to measure).
- **`annotator_profiles`** — per-rater reliability: agreement with the
  majority label, mean pairwise kappa, own-label entropy (near 0 means
  they stamp one label on everything), class bias, and a 0–1
  `reliability_score`.
- **`disagreement_hotspots`** — items ranked by normalized label entropy:
  the items your raters fought over most. Re-annotate or adjudicate these
  first.
- **`class_distribution`** — counts, proportions, imbalance ratio
  (largest/smallest class), Gini coefficient, effective number of classes,
  and classes below the minority threshold.
- **`self_consistency`** — per item: the cross-validated predicted label,
  `plausibility` (fraction of folds agreeing with the consensus label),
  and a `suspicious` flag when plausibility < 0.5.
- **`review_queue`** — every item ranked by composite risk:
  `0.4 * entropy + 0.3 * (1 - mean annotator reliability) +
  0.3 * (1 - plausibility)`, each with human-readable reasons. Hand the
  top of this list to your adjudicators.

## 4. Python API

```python
from label_quality_scorer import (
    score_annotations, cohen_kappa, fleiss_kappa,
    annotator_profiles, build_review_queue,
)

records = [
    {"item_id": "t1", "annotator": "ana", "label": "pos"},
    {"item_id": "t1", "annotator": "ben", "label": "pos"},
    ...
]

report = score_annotations(records, features={"t1": [0.1, 0.9]})

# Or use the pieces individually:
kappa = cohen_kappa(["a", "a", "b"], ["a", "b", "b"])
profiles = annotator_profiles(records)
queue = build_review_queue(records)
```

## 5. Tuning

Pass a `LabelQualityConfig` to `score_annotations`:

- `hotspot_top_n` / `review_queue_top_n` — list lengths
- `minority_threshold` — class-share cutoff for minority flags (default 0.05)
- `cv_folds` — folds for the self-consistency check (default 5; reduced
  automatically on tiny datasets)
- `review_weights` — rebalance the review-queue risk blend

## 6. Interpreting kappa

| kappa range | reading              |
|-------------|----------------------|
| 0.81 – 1.00 | near-perfect agreement |
| 0.61 – 0.80 | substantial          |
| 0.41 – 0.60 | moderate             |
| 0.21 – 0.40 | fair                 |
| ≤ 0.20      | slight / chance-level |

Low kappa with high percent agreement usually means a skewed class
distribution (raters agree on the dominant class by default) — check the
class-distribution section before rewriting your guidelines.
