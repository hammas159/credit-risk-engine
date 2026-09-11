"""Credit risk engine tests.

Scorecard behaviour is almost entirely exact — WoE is a logarithm, points are a linear
transform, the four-fifths rule is a ratio — so nearly everything here is asserted
rather than compared against a tolerance.
"""

from __future__ import annotations

import math
import random

import pytest

from creditrisk.binning import bin_categorical, bin_numeric, quantile_edges
from creditrisk.fairness import audit, group_metrics, threshold_for_parity
from creditrisk.scorecard import (
    Scorecard,
    brier_score,
    calibration_table,
    fit_logistic,
    gini,
)


def synthetic(n: int = 1500, seed: int = 11):
    """Higher income means lower risk. Nothing else carries signal."""
    rng = random.Random(seed)
    income = [rng.uniform(20, 200) for _ in range(n)]
    age = [rng.uniform(21, 70) for _ in range(n)]
    target = [1 if rng.random() < max(0.02, 0.45 - v / 400) else 0 for v in income]
    return income, age, target


def built_scorecard():
    income, age, target = synthetic()
    f_income = bin_numeric("income", income, target, n_bins=5)
    f_age = bin_numeric("age", age, target, n_bins=5)
    rows = [[f_income.transform(i), f_age.transform(a)] for i, a in zip(income, age)]
    weights, bias = fit_logistic(rows, target)
    card = Scorecard(
        {"income": f_income, "age": f_age},
        {"income": weights[0], "age": weights[1]},
        bias,
    ).build()
    return card, (income, age, target)


class TestBinning:
    def test_quantile_edges_are_equal_frequency(self):
        """Equal-width bins on a skewed feature put 95% of the population in one bin."""
        edges = quantile_edges(list(range(100)), 4)
        assert len(edges) == 3
        assert edges == sorted(edges)

    def test_woe_is_negative_where_risk_is_high(self):
        """WoE is ln(P(good)/P(bad)), so a high-risk bin must be negative. Getting
        this sign wrong inverts the entire scorecard."""
        income, _, target = synthetic()
        feature = bin_numeric("income", income, target, n_bins=5)
        worst = max(feature.bins, key=lambda b: b.bad_rate)
        assert feature.woe[worst.label] < 0

    def test_woe_is_monotonic_for_a_monotonic_feature(self):
        income, _, target = synthetic()
        assert bin_numeric("income", income, target, n_bins=5).is_monotonic()

    def test_a_noise_feature_scores_useless(self):
        _, age, target = synthetic()
        assert bin_numeric("age", age, target, n_bins=5).iv < 0.02

    def test_a_predictive_feature_scores_high(self):
        income, _, target = synthetic()
        assert bin_numeric("income", income, target, n_bins=5).iv > 0.3

    def test_leakage_is_flagged_as_suspicious(self):
        """An IV above 0.5 almost always means the feature encodes the outcome."""
        target = [i % 2 for i in range(400)]
        leak = [float(y) for y in target]  # the label itself
        assert "suspicious" in bin_numeric("leak", leak, target, n_bins=2).strength()

    def test_a_bin_with_no_bads_does_not_produce_infinity(self):
        """Continuity correction, not an arbitrary epsilon."""
        values = [1.0] * 50 + [100.0] * 50
        target = [0] * 50 + [1] * 50
        feature = bin_numeric("x", values, target, n_bins=2)
        assert all(math.isfinite(w) for w in feature.woe.values())

    def test_missing_values_get_their_own_bin(self):
        """Rather than an imputed lie. Missingness is frequently predictive in
        lending — a blank employer field is information."""
        values = [1.0, 2.0, 3.0, None, None, 4.0]
        feature = bin_numeric("x", values, [0, 0, 1, 1, 1, 0], n_bins=2)
        assert any(b.is_missing and b.total == 2 for b in feature.bins)

    def test_undersized_bins_are_merged(self):
        """A bin of nine accounts produces a WoE that will not survive next quarter."""
        values = [float(i) for i in range(100)]
        target = [i % 2 for i in range(100)]
        feature = bin_numeric("x", values, target, n_bins=20, min_bin_fraction=0.15)
        numeric = [b for b in feature.bins if not b.is_missing]
        assert all(b.total >= 15 for b in numeric)

    def test_categorical_binning(self):
        feature = bin_categorical(
            "grade", ["A", "A", "B", "B", "C", None], [0, 0, 1, 1, 1, 0]
        )
        assert {b.label for b in feature.bins} == {"A", "B", "C", "missing"}

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            bin_numeric("x", [1.0, 2.0], [0])

    def test_a_single_class_yields_no_evidence(self):
        feature = bin_numeric("x", [1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], n_bins=2)
        assert feature.iv == 0.0


class TestScorecard:
    def test_higher_quality_applicants_score_higher(self):
        """The bug this test exists for: the model predicts P(bad) while a score is
        log-odds of *good*, so the sign flips on the way into points. Get it wrong and
        the scorecard is statistically right and commercially backwards."""
        card, _ = built_scorecard()
        assert card.score({"income": 180, "age": 45}) > card.score({"income": 25, "age": 22})

    def test_score_and_probability_never_disagree(self):
        """They are the same number in two units, so this is exact."""
        card, _ = built_scorecard()
        app = {"income": 120, "age": 40}
        odds = math.exp((card.score(app) - card.offset) / card.factor)
        assert card.probability(app) == pytest.approx(1 / (1 + odds), abs=1e-6)

    def test_points_sum_to_the_score(self):
        """No hidden constant, or the decision cannot be explained line by line."""
        card, _ = built_scorecard()
        app = {"income": 120, "age": 40}
        assert sum(c.points for c in card.contributions(app)) == card.score(app)

    def test_pdo_doubles_the_odds(self):
        """The definition of the points scale: +PDO points halves the risk."""
        card, _ = built_scorecard()

        def odds(score):
            return math.exp((score - card.offset) / card.factor)

        assert odds(600 + card.pdo) == pytest.approx(2 * odds(600), rel=1e-9)

    def test_base_score_sits_at_base_odds(self):
        card, _ = built_scorecard()
        expected = card.offset + card.factor * math.log(card.base_odds)
        assert expected == pytest.approx(card.base_score, abs=1e-6)

    def test_a_missing_field_still_scores(self):
        """An incomplete application must produce a decision, not an exception."""
        card, _ = built_scorecard()
        assert isinstance(card.score({"income": 120}), int)

    def test_reason_codes_name_the_worst_factor_first(self):
        card, _ = built_scorecard()
        reasons = card.reason_codes({"income": 25, "age": 22})
        assert reasons and reasons[0]["feature"] == "income"
        assert reasons[0]["points_lost"] > 0

    def test_a_factor_that_cost_nothing_is_not_a_reason(self):
        """Adverse-action notices must list reasons for the decline, not a feature
        list."""
        card, _ = built_scorecard()
        best = {"income": 195, "age": 45}
        assert all(r["points_lost"] > 0 for r in card.reason_codes(best))

    def test_reason_codes_are_measured_against_the_best_attainable(self):
        card, _ = built_scorecard()
        reasons = card.reason_codes({"income": 25, "age": 22})
        income_best = max(card.points["income"].values())
        income_reason = next(r for r in reasons if r["feature"] == "income")
        assert income_reason["points_lost"] == income_best - income_reason["points"]


class TestMetrics:
    def test_gini_is_one_for_perfect_separation(self):
        assert gini([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_gini_is_zero_for_a_coin_flip(self):
        assert gini([0.5] * 4, [0, 1, 0, 1]) == 0.0

    def test_gini_is_negative_when_the_model_is_inverted(self):
        """Which is what an inverted scorecard would look like in production."""
        assert gini([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == -1.0

    def test_gini_needs_both_classes(self):
        assert gini([0.1, 0.9], [0, 0]) == 0.0

    def test_the_model_discriminates(self):
        card, (income, age, target) = built_scorecard()
        probs = [card.probability({"income": i, "age": a}) for i, a in zip(income, age)]
        assert gini(probs, target) > 0.3

    def test_brier_rewards_calibration_not_just_ranking(self):
        """A model can rank perfectly and still be wrong about the level — and a
        lender prices from the level."""
        confident = brier_score([0.01, 0.99], [0, 1])
        timid = brier_score([0.45, 0.55], [0, 1])
        assert confident < timid

    def test_calibration_table_compares_predicted_with_observed(self):
        card, (income, age, target) = built_scorecard()
        probs = [card.probability({"income": i, "age": a}) for i, a in zip(income, age)]
        table = calibration_table(probs, target, n_bins=5)
        assert len(table) >= 5
        assert table[0]["predicted"] < table[-1]["predicted"]
        assert table[0]["observed"] < table[-1]["observed"]

    def test_empty_inputs_do_not_raise(self):
        assert brier_score([], []) == 0.0
        assert calibration_table([], []) == []


class TestFairness:
    def test_group_metrics(self):
        metrics = group_metrics(
            ["a", "a", "b", "b"], [1, 0, 1, 1], [0, 1, 0, 1]
        )
        assert metrics["a"].n == 2
        assert metrics["b"].approval_rate == 1.0

    def test_disparate_impact_flags_the_four_fifths_rule(self):
        groups = ["majority"] * 100 + ["minority"] * 100
        approved = [1] * 90 + [0] * 10 + [1] * 50 + [0] * 50
        outcomes = [0] * 200
        report = audit(groups, approved, outcomes)
        assert report["worst_disparate_impact"] == pytest.approx(50 / 90, abs=1e-4)
        assert report["four_fifths_rule_flag"] is True

    def test_an_even_handed_model_is_not_flagged(self):
        groups = ["majority"] * 100 + ["minority"] * 100
        approved = [1] * 85 + [0] * 15 + [1] * 80 + [0] * 20
        report = audit(groups, approved, [0] * 200)
        assert report["four_fifths_rule_flag"] is False

    def test_the_reference_group_is_the_largest(self):
        """Picking it alphabetically is an accident waiting for a label to change."""
        groups = ["zzz"] * 100 + ["aaa"] * 10
        report = audit(groups, [1] * 110, [0] * 110)
        assert report["reference_group"] == "zzz"

    def test_equal_opportunity_gap_looks_only_at_those_who_would_repay(self):
        groups = ["a"] * 4 + ["b"] * 4
        outcomes = [0, 0, 1, 1, 0, 0, 1, 1]
        approved = [1, 1, 0, 0, 1, 0, 0, 0]  # group b rejects a good applicant
        report = audit(groups, approved, outcomes, reference="a")
        assert report["comparisons"]["b"]["equal_opportunity_gap"] == pytest.approx(-0.5)

    def test_equalised_odds_takes_the_worse_of_the_two_gaps(self):
        groups = ["a"] * 4 + ["b"] * 4
        outcomes = [0, 0, 1, 1, 0, 0, 1, 1]
        approved = [1, 1, 0, 0, 1, 1, 1, 1]  # b approves everyone: TPR equal, FPR worse
        report = audit(groups, approved, outcomes, reference="a")
        comparison = report["comparisons"]["b"]
        assert comparison["equal_opportunity_gap"] == 0.0
        assert comparison["equalised_odds_gap"] == pytest.approx(1.0)

    def test_base_rate_difference_is_reported_alongside(self):
        """It is the usual explanation offered for a gap, and the reader is entitled
        to judge it rather than be told it."""
        groups = ["a"] * 4 + ["b"] * 4
        report = audit(groups, [1] * 8, [0, 0, 0, 0, 1, 1, 1, 1], reference="a")
        assert report["comparisons"]["b"]["bad_rate_difference"] == pytest.approx(1.0)

    def test_the_incompatibility_is_stated_not_hidden(self):
        """Demographic parity and equalised odds cannot both hold when base rates
        differ. A report quoting one measure is concealing that."""
        report = audit(["a", "a", "b", "b"], [1, 0, 1, 0], [0, 1, 0, 1])
        assert "equalised odds" in report["note"]

    def test_a_single_group_has_nothing_to_compare(self):
        assert "nothing to compare" in audit(["a", "a"], [1, 0], [0, 1])["note"]

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            group_metrics(["a"], [1, 0], [0])

    def test_parity_thresholds_are_group_specific(self):
        """Computed so the trade-off is concrete: this achieves parity exactly, and
        using it is unlawful in most jurisdictions because the protected attribute
        enters the decision."""
        scores = [float(i) for i in range(100)]
        groups = ["a"] * 50 + ["b"] * 50
        cutoffs = threshold_for_parity(scores, groups, target_rate=0.5)
        assert cutoffs["a"] != cutoffs["b"]
