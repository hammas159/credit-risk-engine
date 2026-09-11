"""Scorecard construction, scoring, and reason codes.

A scorecard turns a logistic model into integer points, which is what actually gets
deployed in lending. The conversion is fixed by two numbers a business chooses, not by
anything the model knows:

    PDO           Points to Double the Odds. "Every 20 points halves the risk."
    base_score    the score at base_odds

    factor = PDO / ln(2)
    offset = base_score − factor × ln(base_odds)
    score  = offset + factor × ln(odds)

The reason this survives in regulated lending is that the score decomposes exactly:
each feature contributes a whole number of points, and those points sum to the score.
So "declined because: 6 months at address (−42), 3 recent enquiries (−31)" is not an
approximation of the decision — it *is* the decision, restated.

Adverse-action notices are a legal requirement in most jurisdictions. A model that
cannot produce them cannot be deployed, whatever its AUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .binning import BinnedFeature


@dataclass
class ScorecardPoints:
    feature: str
    bin_label: str
    points: int


@dataclass
class Scorecard:
    """Integer points per feature bin, plus the intercept."""

    features: dict[str, BinnedFeature]
    coefficients: dict[str, float]
    intercept: float
    pdo: float = 20.0
    base_score: int = 600
    base_odds: float = 50.0
    points: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def build(self) -> Scorecard:
        """Allocate points to every bin.

        **Sign.** The fitted model predicts the probability of *bad*, while a credit
        score is by universal convention the log-odds of *good* — higher is safer. So
        the logit is negated on the way into points:

            ln(odds_good) = −(intercept + Σ βᵢ·WoEᵢ)

        Getting this backwards produces a scorecard that runs the right way round
        statistically and the wrong way round commercially: the best applicants score
        lowest, and nobody notices until the portfolio does.

        The intercept is spread evenly across features so the points of any complete
        application sum to the score with no leftover term. A scorecard with a hidden
        constant cannot be explained line by line, which defeats the purpose.
        """
        n = len(self.features) or 1
        shared = (self.offset - self.factor * self.intercept) / n

        self.points = {}
        for name, feature in self.features.items():
            beta = self.coefficients.get(name, 0.0)
            self.points[name] = {
                label: int(round(shared - self.factor * beta * woe))
                for label, woe in feature.woe.items()
            }
        return self

    # ---- scoring -------------------------------------------------------------

    def contributions(self, application: dict) -> list[ScorecardPoints]:
        out: list[ScorecardPoints] = []
        for name, feature in self.features.items():
            value = application.get(name)
            bin_label = next(
                (b.label for b in feature.bins if b.contains(value)),
                next((b.label for b in feature.bins if b.is_missing), ""),
            )
            if not bin_label:
                continue
            out.append(ScorecardPoints(name, bin_label, self.points[name][bin_label]))
        return out

    def score(self, application: dict) -> int:
        return sum(c.points for c in self.contributions(application))

    def probability(self, application: dict) -> float:
        """Probability of *bad*, recovered from the score.

        The inverse of the points transformation, so score and probability can never
        disagree — they are the same number in two units.
        """
        odds = math.exp((self.score(application) - self.offset) / self.factor)
        return round(1.0 / (1.0 + odds), 6)

    # ---- explanation ---------------------------------------------------------

    def reason_codes(self, application: dict, *, limit: int = 4) -> list[dict]:
        """Why this application scored what it did, worst first.

        Measured against each feature's **best attainable** points, not against zero
        or against the population mean. "You lost 42 points relative to the best
        possible answer on this question" is actionable; "this feature contributed
        180 points" is not.
        """
        reasons = []
        for c in self.contributions(application):
            best = max(self.points[c.feature].values())
            reasons.append(
                {
                    "feature": c.feature,
                    "bin": c.bin_label,
                    "points": c.points,
                    "points_lost": best - c.points,
                }
            )
        reasons.sort(key=lambda r: -r["points_lost"])
        # A reason that cost nothing is not a reason for the decline.
        return [r for r in reasons if r["points_lost"] > 0][:limit]


def logistic(x: float) -> float:
    # Guarded against overflow at the tails, which a scorecard reaches routinely.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_logistic(
    rows: list[list[float]],
    target: list[int],
    *,
    epochs: int = 400,
    lr: float = 0.1,
    l2: float = 0.001,
) -> tuple[list[float], float]:
    """Plain gradient-descent logistic regression.

    Written out rather than imported, because on WoE-transformed features the problem
    is small, convex and well conditioned - the fit is a handful of lines, and having
    it here means the whole scorecard can be read end to end without leaving the repo.
    """
    if not rows:
        return [], 0.0
    n_features = len(rows[0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(rows)

    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for row, y in zip(rows, target, strict=False):
            prediction = logistic(sum(w * x for w, x in zip(weights, row, strict=False)) + bias)
            error = prediction - y
            for j, x in enumerate(row):
                grad_w[j] += error * x
            grad_b += error
        for j in range(n_features):
            # L2 keeps a bin that separates perfectly from taking an unbounded
            # coefficient, which is the usual failure on small samples.
            weights[j] -= lr * (grad_w[j] / n + l2 * weights[j])
        bias -= lr * grad_b / n

    return weights, bias


# --- calibration ----------------------------------------------------------------


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Mean squared error of a probability forecast. Lower is better.

    Reported alongside discrimination because they answer different questions. A model
    can rank perfectly and still be wrong about the level - and a lender prices from
    the level, not the ranking.
    """
    if not outcomes:
        return 0.0
    return round(
        sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=False)) / len(outcomes), 6
    )


def calibration_table(
    probabilities: list[float], outcomes: list[int], *, n_bins: int = 10
) -> list[dict]:
    """Predicted versus observed bad rate, by decile of predicted risk."""
    if not probabilities:
        return []
    paired = sorted(zip(probabilities, outcomes, strict=False))
    size = max(1, len(paired) // n_bins)
    table = []
    for i in range(0, len(paired), size):
        chunk = paired[i : i + size]
        if not chunk:
            continue
        table.append(
            {
                "n": len(chunk),
                "predicted": round(sum(p for p, _ in chunk) / len(chunk), 6),
                "observed": round(sum(y for _, y in chunk) / len(chunk), 6),
            }
        )
    return table


def gini(probabilities: list[float], outcomes: list[int]) -> float:
    """Gini coefficient, = 2·AUC − 1. The discrimination measure lenders quote.

    Computed by counting concordant pairs directly. On the sample sizes a scorecard is
    built from, the exact calculation is fast enough and avoids the threshold-sweeping
    approximations that make AUC implementations disagree at the third decimal.
    """
    bads = [p for p, y in zip(probabilities, outcomes, strict=False) if y == 1]
    goods = [p for p, y in zip(probabilities, outcomes, strict=False) if y == 0]
    if not bads or not goods:
        return 0.0
    concordant = sum(1.0 if b > g else 0.5 if b == g else 0.0 for b in bads for g in goods)
    auc = concordant / (len(bads) * len(goods))
    return round(2 * auc - 1, 6)
