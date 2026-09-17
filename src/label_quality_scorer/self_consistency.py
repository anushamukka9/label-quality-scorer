"""Cross-validation-style self-consistency flags for suspicious labels.

Given consensus labels plus numeric feature vectors per item, train a simple
nearest-centroid classifier on stratified folds and check whether each item's
label survives cross-validation. Items whose consensus label is repeatedly
*not* predicted are flagged as suspicious — a cheap, model-based second
opinion on the annotations.

Pure numpy, fully deterministic (stratified round-robin folds, no RNG).
"""

from collections import defaultdict

import numpy as np


def _stratified_folds(item_ids, labels, n_folds):
    """Deterministic stratified fold assignment via round-robin per class."""
    by_class = defaultdict(list)
    for iid, lab in zip(item_ids, labels):
        by_class[lab].append(iid)
    folds = [[] for _ in range(n_folds)]
    for lab in sorted(by_class, key=str):
        members = sorted(by_class[lab], key=str)
        for k, iid in enumerate(members):
            folds[k % n_folds].append(iid)
    return folds


def _nearest_centroid_predict(X_train, y_train, X_test):
    classes = sorted(set(y_train), key=str)
    centroids = {}
    for c in classes:
        pts = X_train[np.array(y_train) == np.array(c)]
        centroids[c] = pts.mean(axis=0)
    preds = []
    for x in X_test:
        best, best_d = None, None
        for c in classes:
            d = float(np.sum((x - centroids[c]) ** 2))
            if best_d is None or d < best_d:
                best, best_d = c, d
        preds.append(best)
    return preds


def self_consistency_flags(items, features, n_folds=5):
    """Flag labels that fail k-fold cross-validated self-consistency.

    ``items``: list of (item_id, consensus_label).
    ``features``: dict item_id -> 1-D numeric array-like.

    Returns dict item_id -> {"predicted_label", "plausibility",
    "suspicious"} where plausibility is the fraction of folds predicting the
    consensus label. An item is suspicious when plausibility < 0.5.

    Items lacking features are skipped. When there are fewer than 2 classes
    or too few items for meaningful folds, every covered item keeps
    plausibility 1.0 (nothing measurable — not flagged).
    """
    covered = [(iid, lab) for iid, lab in items if iid in features]
    labels = sorted({lab for _, lab in covered}, key=str)
    if len(covered) < 2 or len(labels) < 2:
        return {
            iid: {
                "predicted_label": lab,
                "plausibility": 1.0,
                "suspicious": False,
            }
            for iid, lab in covered
        }

    item_ids = [iid for iid, _ in covered]
    y = [lab for _, lab in covered]
    X = np.array([np.asarray(features[iid], dtype=float) for iid in item_ids])
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Standardize so no single feature dominates the distance.
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / std

    folds = min(n_folds, len(item_ids))
    fold_of = {}
    for f, members in enumerate(_stratified_folds(item_ids, y, folds)):
        for iid in members:
            fold_of[iid] = f

    id_to_idx = {iid: k for k, iid in enumerate(item_ids)}
    votes = {iid: [] for iid in item_ids}
    for f in range(folds):
        test_ids = [iid for iid in item_ids if fold_of[iid] == f]
        train_ids = [iid for iid in item_ids if fold_of[iid] != f]
        if not test_ids or not train_ids:
            continue
        if len({y[id_to_idx[i]] for i in train_ids}) < 2:
            continue  # degenerate fold: nothing to learn
        X_train = Xs[[id_to_idx[i] for i in train_ids]]
        y_train = [y[id_to_idx[i]] for i in train_ids]
        X_test = Xs[[id_to_idx[i] for i in test_ids]]
        for iid, pred in zip(
            test_ids, _nearest_centroid_predict(X_train, y_train, X_test)
        ):
            votes[iid].append(pred)

    results = {}
    for iid, lab in covered:
        vs = votes[iid]
        if not vs:
            plaus, pred = 1.0, lab
        else:
            plaus = sum(1 for v in vs if v == lab) / len(vs)
            pred = max(set(vs), key=lambda v: (vs.count(v), str(v)))
        results[iid] = {
            "predicted_label": pred,
            "plausibility": round(plaus, 4),
            "suspicious": bool(plaus < 0.5),
        }
    return results
