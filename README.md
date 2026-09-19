<h1 align="center">credit-risk-engine (Python · WoE/IV binning · logistic scorecard)</h1>
<p align="center"><i>A credit scorecard that can explain every decline, in points, to a regulator</i></p>

<p align="center">
  <a href="#why-scorecards-when-gradient-boosting-scores-better">Why scorecards</a> &middot;
  <a href="#weight-of-evidence">Weight of Evidence</a> &middot;
  <a href="#the-points-scale">The points scale</a> &middot;
  <a href="#calibration-not-just-discrimination">Calibration</a> &middot;
  <a href="#fairness-four-measures-because-they-are-incompatible">Fairness</a> &middot;
  <a href="#problems-hit-while-building-this">Problems hit</a>
</p>

<p align="center">
  <a href="https://github.com/hammasbuilds/credit-risk-engine/actions/workflows/ci.yml"><img src="https://github.com/hammasbuilds/credit-risk-engine/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/core%20deps-zero-success" alt="deps">
  <img src="https://img.shields.io/badge/stack-pandas%20%C2%B7%20Streamlit-orange" alt="stack">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="license"></a>
</p>

---

## Why scorecards, when gradient boosting scores better

```mermaid
flowchart LR
    D["applicant data"] --> W["WoE / IV binning"]
    W --> L["logistic fit"]
    L --> P["integer points scale"]
    P --> S["score"]
    S --> R["reason codes<br/>why THIS decline"]
    S --> C["calibration check"]
    S --> F["fairness audit<br/>four measures"]

    style R fill:#2563eb,color:#fff
    style F fill:#f59e0b,color:#fff
```

Gradient boosting scores better. **A scorecard can be read by a person** - and when a
regulator asks why an application was declined, "points" is an answer and "feature
importance" is not.


Because in regulated lending **an unexplainable decision is not a decision**. Adverse-
action notices are a legal requirement in most jurisdictions; a model that cannot
produce one cannot be deployed whatever its AUC.

A scorecard decomposes **exactly**. Each feature contributes a whole number of points
and those points sum to the score, so:

> *Declined — 6 months at address (−42), 3 recent enquiries (−31)*

is not an approximation of the decision. It **is** the decision, restated.

```python
card.reason_codes({"income": 25, "age": 22})
# [{"feature": "income", "bin": "[-inf, 56.8)", "points": 242, "points_lost": 76}]
```

Reasons are measured against each feature's **best attainable** points — *"you lost 76
points relative to the best possible answer"* is actionable. *"This feature contributed
242 points"* is not. A factor that cost nothing is not listed, because an adverse-action
notice lists reasons for the decline, not a feature inventory.

## Weight of Evidence

```
WoE = ln( P(good | bin) / P(bad | bin) )
```

Three properties fall out, and together they are why the technique has lasted forty
years: after binning, higher WoE always means lower risk (**monotonic**); WoE is already
log-odds, the scale logistic regression works on (**linear**); outliers land in the edge
bin and missing values get their own bin rather than an imputed lie (**robust**).

`is_monotonic()` is checked, not assumed — a feature that is not monotonic cannot be
described in one sentence to a credit committee, and a scorecard that cannot be
described will not be approved.

### Information Value, and the band that matters

```
< 0.02    useless
0.02–0.1  weak
0.1–0.3   medium
0.3–0.5   strong
> 0.5     suspicious — usually leakage, not a great feature
```

The top band is the useful one. An IV above 0.5 almost always means the feature encodes
the outcome — a collections flag that is only ever set *after* default. That is a test:
feeding the label back in as a feature must be flagged, not celebrated.

Bins smaller than 5% of the population are merged. A bin of nine accounts produces a WoE
that will not survive contact with next quarter's data.

## The points scale

Two numbers the business chooses, not the model:

```
factor = PDO / ln 2
offset = base_score − factor × ln(base_odds)
score  = offset + factor × ln(odds_good)
```

`PDO` is *Points to Double the Odds* — "every 20 points halves the risk". That is
asserted directly:

```python
def test_pdo_doubles_the_odds():
    assert odds(600 + card.pdo) == pytest.approx(2 * odds(600))
```

### The bug this project taught me

The model predicts `P(bad)`. A credit score is, by universal convention, **log-odds of
good** — higher is safer. The sign has to flip on the way into points.

I got it wrong first, and the failure is instructive: the scorecard was *statistically
correct and commercially inverted*. Every metric looked fine; the best applicants simply
scored lowest. Nothing catches that except checking the direction — so now that check is
the first test in the file, and `gini` returning **−1.0** for an inverted model is
another.

## Calibration, not just discrimination

```python
gini(probs, target)          # ranking  — who is riskier than whom
brier_score(probs, target)   # level    — is 5% actually 5%
calibration_table(probs, target)
```

Both, because they answer different questions. A model can rank perfectly and still be
badly wrong about the level — and a lender **prices from the level**.

## Fairness: four measures, because they are incompatible

Except in degenerate cases, a model **cannot** satisfy demographic parity and equalised
odds at once when base rates differ between groups. That is a theorem, not a tuning
problem. A report quoting one measure is concealing it, so `audit()` returns all four
and says so in the output.

| | |
|---|---|
| **Demographic parity** | equal approval rates — ignores whether groups differ in actual risk |
| **Equal opportunity** | equal TPR — among people who *would repay*, equal chance of approval |
| **Equalised odds** | equal TPR *and* FPR — strictest, rarely achievable |
| **Disparate impact** | approval-rate ratio; the four-fifths rule flags below 0.8 |

The reference group is the **largest**, not the alphabetically first — the comparison
people care about is "relative to the majority", and picking it by name is an accident
waiting for a label to change.

`bad_rate_difference` is reported alongside every gap, because it is the usual
explanation offered for one and the reader is entitled to judge it rather than be told
it.

`threshold_for_parity()` exists to make the trade-off concrete: group-specific cut-offs
achieve demographic parity *exactly*, and in most jurisdictions using them is itself
unlawful, because the protected attribute enters the decision. A tool that computes this
should say so — so it does, in its own docstring.

**On Pakistan:** the PDPA does not yet define algorithmic fairness thresholds. The
four-fifths rule is applied here as the strictest widely recognised standard rather than
as a local legal requirement. Erring stricter than the law demands is the defensible
direction.

## Usage

```python
income_f = bin_numeric("income", income, target, n_bins=5)
rows = [[income_f.transform(i), age_f.transform(a)] for i, a in zip(income, age)]
weights, bias = fit_logistic(rows, target)

card = Scorecard({"income": income_f, "age": age_f},
                 dict(zip(["income", "age"], weights)), bias).build()

card.score({"income": 180, "age": 45})        # 584
card.probability({"income": 180, "age": 45})  # 0.0336
card.reason_codes({"income": 25, "age": 22})

audit(gender, approved, outcomes)
```

---

## Input

![input](docs/images/input.png)

## Output

`python demo.py`

![output](docs/images/output.png)

*The reason code names `income`, not `age` — even though `age` contributed more points.
What a declined applicant is owed is the factor that cost them the most relative to the
best available bin, and `age` carries no signal to lose points on.*

---

## Tests

**40 tests. No dependencies, no data download.**

Scorecard behaviour is almost entirely exact — WoE is a logarithm, points are a linear
transform, four-fifths is a ratio — so nearly everything is asserted rather than
compared against a tolerance.

| Covered | |
|---|---|
| Binning | WoE sign, monotonicity, IV bands, **leakage flagged**, zero-bad bins finite, missing bin, small-bin merge, categorical |
| Scorecard | **direction**, points sum to score, PDO doubles odds, base score at base odds, score/probability agree, missing field |
| Reason codes | worst first, measured against best attainable, zero-cost factors excluded |
| Metrics | Gini at ±1 and 0, Brier rewards calibration, calibration table ordering |
| Fairness | four-fifths flag, even-handed model not flagged, largest reference group, TPR-only gap, worse-of-two odds gap, base rates, incompatibility stated |

## Limits

- `fit_logistic` is plain gradient descent with L2. Fine on WoE features, where the
  problem is small and convex; it is not a substitute for a solver on wide raw data.
- No reject inference. Scoring only the accepted population biases the model, and the
  standard fixes (parcelling, augmentation) all rest on assumptions worth stating
  explicitly rather than burying in a function.
- Binning is quantile-based with a size floor, not an optimal monotonic search. The
  monotonicity is *checked*, not enforced.
- The fairness audit measures outcomes. It cannot tell you whether a feature is a proxy
  for a protected attribute — that needs domain knowledge, not statistics.

## Keywords

credit scoring &middot; scorecard &middot; weight of evidence &middot; WoE &middot; information value &middot; IV binning &middot; logistic regression &middot; reason codes &middot; adverse action &middot; model calibration &middot; Brier score &middot; fairness &middot; disparate impact &middot; equalized odds &middot; explainable AI &middot; regulated ML &middot; credit risk &middot; Basel

## License

MIT

---

## Run it yourself

```bash
git clone https://github.com/hammasbuilds/credit-risk-engine
cd credit-risk-engine

pip install -e .         # zero dependencies to resolve
pytest -q                # 40 tests, under a second
```

```python
from creditrisk import bin_numeric, fit_logistic, Scorecard, gini, brier_score, audit

income_f = bin_numeric("income", income, default_flag, n_bins=5)
age_f    = bin_numeric("age",    age,    default_flag, n_bins=5)
print(income_f.iv, income_f.strength(), income_f.is_monotonic())

rows = [[income_f.transform(i), age_f.transform(a)] for i, a in zip(income, age)]
weights, bias = fit_logistic(rows, default_flag)

card = Scorecard({"income": income_f, "age": age_f},
                 dict(zip(["income", "age"], weights)), bias).build()

card.score({"income": 180, "age": 45})        # 584
card.probability({"income": 180, "age": 45})  # 0.0336
card.reason_codes({"income": 25, "age": 22})  # adverse-action notice

audit(gender, approved, outcomes)             # four fairness measures, all four
```

## Problems hit while building this

**The first scorecard was statistically correct and commercially inverted.** The model
predicts `P(bad)`, but a credit score is by universal convention the log-odds of *good*
— higher is safer. Without the sign flip, the best applicants scored **lowest**: income
180k returned a score of 397 and a 96% probability of default.

What makes this the worst bug in the repo is that **every metric looked fine**. Gini was
healthy, calibration was healthy, the reason codes were well-formed. Nothing catches it
except deliberately checking the direction. That check is now the first test in the file,
and `gini` returning **−1.0** on an inverted model is a second line of defence.

**Information Value above 0.5 was originally labelled "excellent".** It is almost always
**leakage** — a collections flag that is only ever set after default, a field populated
by the decision itself. *Fixed* by labelling that band `suspicious (check for leakage)`,
with a test that feeds the label back in as a feature and asserts it is flagged rather
than celebrated.

**Reporting one fairness measure was hiding a theorem.** Demographic parity and
equalised odds cannot both hold when base rates differ between groups — that is proven,
not a tuning problem. *Fixed* by returning all four measures plus the incompatibility
note, so a reader has to choose a standard rather than be handed one.
