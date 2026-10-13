import math

import numpy as np
import pytest

from mayaz.eval.metrics import Decision, accuracy, brier, bucket_of, ece, kl_from_gold, summarize


def d(pred, gold, gold_label=None, kind="choice"):
    keys = [str(i) for i in range(len(pred))]
    gold = np.asarray(gold, float)
    label = gold_label if gold_label is not None else keys[int(np.argmax(gold))]
    return Decision(kind=kind, keys=keys, pred=np.asarray(pred, float), gold=gold, gold_label=label)


def test_accuracy_hard_and_tie_split():
    ds = [d([0.7, 0.2, 0.1], [1, 0, 0]), d([0.1, 0.8, 0.1], [1, 0, 0]), d([0.5, 0.5], [1, 0])]
    assert accuracy(ds) == pytest.approx((1 + 0 + 0.5) / 3)


def test_kl_from_gold_hand_value_and_zero_gold_terms():
    g, p = [0.5, 0.5, 0.0], [0.25, 0.75, 0.0]
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl_from_gold([d(p, g)]) == pytest.approx(expected)
    assert kl_from_gold([d(g, g)]) == pytest.approx(0.0)


def test_kl_is_finite_when_pred_has_exact_zero():
    assert math.isfinite(kl_from_gold([d([1.0, 0.0], [0.5, 0.5])]))


def test_brier_is_squared_distance_to_soft_gold():
    assert brier([d([0.6, 0.4], [0.5, 0.5])]) == pytest.approx(0.01 + 0.01)


def test_ece_perfectly_calibrated_is_zero_and_overconfident_is_positive():
    calibrated = [d([0.75, 0.25], [1, 0])] * 3 + [d([0.75, 0.25], [0, 1])]
    assert ece(calibrated) == pytest.approx(0.0)
    overconfident = [d([0.95, 0.05], [0, 1])] * 4
    assert ece(overconfident) == pytest.approx(0.95)


BOUNDS = [(2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-20"), (20, "6-20"), (21, "21+"), (64, "21+")]


@pytest.mark.parametrize("k,bucket", BOUNDS)
def test_bucket_boundaries(k, bucket):
    assert bucket_of(k) == bucket


def test_summarize_reports_overall_and_per_bucket():
    ds = [d([0.7, 0.3], [1, 0]), d([0.2, 0.3, 0.5], [0, 0, 1])]
    s = summarize(ds)
    assert s["overall"]["n"] == 2 and s["overall"]["accuracy"] == pytest.approx(1.0)
    assert set(s["by_bucket"]) == {"2", "3-5"}
    assert set(s["by_type"]) == {"choice"}
