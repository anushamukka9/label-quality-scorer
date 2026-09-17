"""Tests for label-quality-scorer."""

import json
import math
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from label_quality_scorer import (  # noqa: E402
    annotator_profiles,
    build_review_queue,
    class_distribution,
    cohen_kappa,
    disagreement_hotspots,
    fleiss_kappa,
    item_entropy,
    pairwise_kappa_matrix,
    percent_agreement,
    score_annotations,
    self_consistency_flags,
)
from label_quality_scorer.cli import main as cli_main  # noqa: E402


def rec(item, annotator, label):
    return {"item_id": item, "annotator": annotator, "label": label}


# ---------------------------------------------------------------- agreement

def test_cohen_kappa_known_value():
    # 5/6 observed agreement, 1/2 expected -> (5/6 - 1/2) / (1/2) = 2/3
    a = [1, 1, 1, 0, 0, 0]
    b = [1, 1, 0, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(2 / 3)


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    # Independent labelings: kappa near 0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 0.2


def test_fleiss_kappa_known_value():
    # Hand-computed: P_bar = 2/3, P_e = 1/2 -> kappa = 1/3
    groups = [
        ["A", "A", "A"],
        ["A", "A", "B"],
        ["B", "B", "B"],
        ["A", "B", "B"],
    ]
    assert fleiss_kappa(groups) == pytest.approx(1 / 3)


def test_fleiss_kappa_perfect_agreement():
    groups = [["x", "x", "x"], ["y", "y", "y"], ["x", "x"]]
    assert fleiss_kappa(groups) == pytest.approx(1.0)


def test_pairwise_kappa_matrix_shapes():
    records = [
        rec("i1", "a", "pos"), rec("i1", "b", "pos"),
        rec("i2", "a", "neg"), rec("i2", "b", "neg"),
        rec("i3", "a", "pos"), rec("i3", "b", "pos"),
    ]
    m = pairwise_kappa_matrix(records)
    assert set(m) == {("a", "b")}
    assert m[("a", "b")] == pytest.approx(1.0)


def test_percent_agreement():
    records = [
        rec("i1", "a", "pos"), rec("i1", "b", "pos"),   # agree
        rec("i2", "a", "pos"), rec("i2", "b", "neg"),   # disagree
    ]
    assert percent_agreement(records) == pytest.approx(0.5)


# ------------------------------------------------------------------ entropy

def test_item_entropy_unanimous_is_zero():
    assert item_entropy(["pos", "pos", "pos"]) == 0.0


def test_item_entropy_max_normalized():
    assert item_entropy(["a", "b"], normalize=True) == pytest.approx(1.0)


def test_disagreement_hotspots_ranking():
    records = [
        rec("split", "a", "pos"), rec("split", "b", "neg"),   # max entropy
        rec("calm", "a", "pos"), rec("calm", "b", "pos"),     # zero entropy
    ]
    hot = disagreement_hotspots(records, top_n=2)
    assert hot[0]["item_id"] == "split"
    assert hot[0]["entropy"] == pytest.approx(1.0)
    assert hot[1]["item_id"] == "calm"


# -------------------------------------------------------------- reliability

def test_annotator_profiles_detects_outlier():
    records = []
    for i in range(6):
        records += [rec(f"i{i}", "good1", "pos"), rec(f"i{i}", "good2", "pos")]
    # outlier always disagrees
    for i in range(6):
        records.append(rec(f"i{i}", "outlier", "neg"))
    profiles = annotator_profiles(records)
    assert profiles["good1"]["reliability_score"] > profiles["outlier"]["reliability_score"]
    assert profiles["outlier"]["agreement_with_majority"] == pytest.approx(0.0)
    assert profiles["outlier"]["label_entropy"] == pytest.approx(0.0)


# --------------------------------------------------------------- imbalance

def test_class_distribution_metrics():
    d = class_distribution(["a"] * 90 + ["b"] * 10)
    assert d["counts"] == {"a": 90, "b": 10}
    assert d["imbalance_ratio"] == pytest.approx(9.0)
    assert d["minority_classes"] == []
    d2 = class_distribution(["a"] * 97 + ["b"] * 3)
    assert d2["minority_classes"] == ["b"]
    assert 0 < d["gini"] < 1


def test_class_distribution_rejects_empty():
    with pytest.raises(ValueError):
        class_distribution([])


# --------------------------------------------------------- self-consistency

def test_self_consistency_flags_mislabeled_item():
    # Two tight clusters; one item labeled as the wrong cluster.
    items, features = [], {}
    for i in range(10):
        iid = f"neg{i}"
        items.append((iid, "neg"))
        features[iid] = [0.0 + i * 0.01, 0.0]
    for i in range(10):
        iid = f"pos{i}"
        items.append((iid, "pos"))
        features[iid] = [10.0 + i * 0.01, 10.0]
    # The suspicious one: sits in the negative cluster, labeled positive.
    items.append(("odd", "pos"))
    features["odd"] = [0.05, 0.02]

    flags = self_consistency_flags(items, features, n_folds=5)
    assert flags["odd"]["suspicious"] is True
    assert flags["odd"]["predicted_label"] == "neg"
    assert flags["neg0"]["suspicious"] is False
    assert flags["neg0"]["plausibility"] == pytest.approx(1.0)


def test_self_consistency_single_class_not_flagged():
    items = [(f"i{i}", "only") for i in range(4)]
    features = {f"i{i}": [float(i)] for i in range(4)}
    flags = self_consistency_flags(items, features)
    assert all(not v["suspicious"] for v in flags.values())


# ------------------------------------------------------------- review queue

def test_review_queue_ranks_riskiest_first():
    records = [
        rec("risky", "a", "pos"), rec("risky", "b", "neg"),
        rec("safe", "a", "pos"), rec("safe", "b", "pos"),
    ]
    queue = build_review_queue(records)
    assert queue[0]["item_id"] == "risky"
    assert queue[0]["risk_score"] > queue[1]["risk_score"]
    assert any("disagreement" in r for r in queue[0]["reasons"])


# ------------------------------------------------------------------- report

def test_score_annotations_full_report(tmp_path):
    records = [
        rec("i1", "a", "pos"), rec("i1", "b", "pos"),
        rec("i2", "a", "neg"), rec("i2", "b", "neg"),
        rec("i3", "a", "pos"), rec("i3", "b", "neg"),
    ]
    features = {"i1": [1.0], "i2": [-1.0], "i3": [0.9]}
    report = score_annotations(records, features=features)
    assert set(report) == {
        "summary", "agreement", "annotator_profiles",
        "disagreement_hotspots", "class_distribution",
        "self_consistency", "review_queue",
    }
    assert report["summary"]["n_items"] == 3
    assert report["summary"]["fleiss_kappa"] is not None
    # JSON-serializable:
    json.dumps(report)
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report))
    assert json.loads(out.read_text())["summary"]["n_items"] == 3


def test_score_annotations_rejects_bad_input():
    with pytest.raises(ValueError):
        score_annotations([])
    with pytest.raises(ValueError):
        score_annotations([{"item_id": "x"}])


# ---------------------------------------------------------------------- cli

def test_cli_end_to_end(tmp_path):
    src = os.path.join(os.path.dirname(__file__), "..", "examples", "annotations.csv")
    out = str(tmp_path / "report.json")
    rc = cli_main([src, "-o", out, "--top-n", "5"])
    assert rc == 0
    report = json.loads(open(out).read())
    assert report["summary"]["n_items"] == 15
    assert len(report["review_queue"]) == 5


def test_cli_with_features(tmp_path):
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    out = str(tmp_path / "report.json")
    rc = cli_main([
        os.path.join(base, "annotations.csv"),
        "--features", os.path.join(base, "item_features.csv"),
        "-o", out,
    ])
    assert rc == 0
    report = json.loads(open(out).read())
    assert report["self_consistency"] is not None
    assert report["summary"]["suspicious_labels"] >= 0


def test_cli_missing_file_returns_error(capsys):
    rc = cli_main(["/nonexistent/annotations.csv", "-o", "x.json"])
    assert rc == 1
    assert "error" in capsys.readouterr().err
