"""One applicant in, a score and its adverse-action reasons out.

    python demo.py

Fits a scorecard on synthetic data where the true relationship is known
(higher income means lower risk, age carries no signal), then scores one
applicant and explains the decline in points. No network, no dependencies.
"""
import random
import sys

sys.path.insert(0, "src")

from creditrisk.binning import bin_numeric
from creditrisk.scorecard import Scorecard, fit_logistic, gini

CUTOFF = 600


def synthetic(n: int = 1500, seed: int = 11):
    """Higher income means lower risk. Age is noise, and should earn ~no points."""
    rng = random.Random(seed)
    income = [rng.uniform(20, 200) for _ in range(n)]
    age = [rng.uniform(21, 70) for _ in range(n)]
    target = [1 if rng.random() < max(0.02, 0.45 - v / 400) else 0 for v in income]
    return income, age, target


income, age, target = synthetic()
f_income = bin_numeric("income", income, target, n_bins=5)
f_age = bin_numeric("age", age, target, n_bins=5)
rows = [[f_income.transform(i), f_age.transform(a)] for i, a in zip(income, age, strict=False)]
weights, bias = fit_logistic(rows, target)
card = Scorecard(
    {"income": f_income, "age": f_age},
    {"income": weights[0], "age": weights[1]},
    bias,
).build()

applicant = {"income": 34.0, "age": 29.0}

print("INPUT")
print(f"   applicant          {applicant}")
print(f"   cutoff             {CUTOFF}")
print(f"   scorecard          fitted on {len(income)} accounts, "
      f"PDO={card.pdo:.0f} base={card.base_score}")
print()

score = card.score(applicant)
prob = card.probability(applicant)

print("OUTPUT")
print(f"   score              {score}")
print(f"   P(default)         {prob:.1%}")
print(f"   decision           {'APPROVE' if score >= CUTOFF else 'DECLINE'}"
      f"   (cutoff {CUTOFF})")
print()
print("   points breakdown")
for c in card.contributions(applicant):
    print(f"      {c.feature:10} {c.bin_label:>18}   {c.points:+4d} pts")
print()
print("   adverse-action reason codes")
for r in card.reason_codes(applicant):
    print(f"      {r}")
print()
fitted = [card.probability({"income": i, "age": a})
          for i, a in zip(income, age, strict=False)]
print(f"   model gini         {gini(fitted, target):.3f}")
