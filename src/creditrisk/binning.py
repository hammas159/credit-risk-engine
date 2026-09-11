"""Weight of Evidence binning and Information Value.

The transformation credit scoring has used for forty years, and the reason scorecards
survive in regulated lending while more accurate models do not: **every step is
explainable to a regulator, a customer and a court.**

Weight of Evidence replaces a raw feature value with the log-odds contributed by its
bin:

    WoE = ln( P(good | bin) / P(bad | bin) )

Three properties fall out of that, and together they are why the technique persists:

  monotonic    after binning, a higher WoE always means a lower risk, so the direction
               of every feature can be stated in one sentence
  linear       WoE is already log-odds, which is exactly the scale logistic regression
               works on, so the model stays a sum of terms
  robust       outliers land in the edge bin and stop mattering; missing values get
               their own bin rather than an imputed lie

Information Value scores how much a feature separates good from bad:

    IV = Σ (P(good|bin) − P(bad|bin)) × WoE

    < 0.02   useless
    0.02-0.1 weak
    0.1-0.3  medium
    0.3-0.5  strong
    > 0.5    suspicious — usually leakage, not a great feature

That last band matters more than the others. An IV above 0.5 almost always means the
feature encodes the outcome: a "collections flag" that is only ever set after default.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

# Prevents a bin with zero goods or zero bads from sending the logarithm to infinity.
# 0.5 is the conventional continuity correction in credit scoring, not an arbitrary
# epsilon - it is the same adjustment used for zero cells in a contingency table.
_SMOOTHING = 0.5


@dataclass
class Bin:
    label: str
    lower: float = -math.inf
    upper: float = math.inf
    goods: int = 0
    bads: int = 0
    is_missing: bool = False

    @property
    def total(self) -> int:
        return self.goods + self.bads

    @property
    def bad_rate(self) -> float:
        return self.bads / self.total if self.total else 0.0

    def contains(self, value: float | None) -> bool:
        if value is None:
            return self.is_missing
        if self.is_missing:
            return False
        return self.lower <= value < self.upper


@dataclass
class BinnedFeature:
    name: str
    bins: list[Bin] = field(default_factory=list)
    woe: dict[str, float] = field(default_factory=dict)
    iv: float = 0.0

    def transform(self, value: float | None) -> float:
        for b in self.bins:
            if b.contains(value):
                return self.woe[b.label]
        # Outside every bin: treat as missing rather than guessing a neighbour.
        missing = next((b for b in self.bins if b.is_missing), None)
        return self.woe[missing.label] if missing else 0.0

    def is_monotonic(self) -> bool:
        """Does WoE move in one direction across the numeric bins?

        Checked rather than assumed. A non-monotonic feature cannot be described in
        one sentence to a credit committee, and a scorecard that cannot be described
        will not be approved.
        """
        values = [self.woe[b.label] for b in self.bins if not b.is_missing]
        if len(values) < 2:
            return True
        increasing = all(b >= a for a, b in zip(values, values[1:]))
        decreasing = all(b <= a for a, b in zip(values, values[1:]))
        return increasing or decreasing

    def strength(self) -> str:
        for limit, label in ((0.02, "useless"), (0.1, "weak"), (0.3, "medium"),
                             (0.5, "strong")):
            if self.iv < limit:
                return label
        return "suspicious (check for leakage)"


def _woe_and_iv(bins: list[Bin]) -> tuple[dict[str, float], float]:
    total_goods = sum(b.goods for b in bins)
    total_bads = sum(b.bads for b in bins)
    if total_goods == 0 or total_bads == 0:
        # One class absent entirely: there is no evidence to weigh.
        return {b.label: 0.0 for b in bins}, 0.0

    woe: dict[str, float] = {}
    iv = 0.0
    for b in bins:
        good_share = (b.goods + _SMOOTHING) / (total_goods + _SMOOTHING * len(bins))
        bad_share = (b.bads + _SMOOTHING) / (total_bads + _SMOOTHING * len(bins))
        value = math.log(good_share / bad_share)
        woe[b.label] = round(value, 6)
        iv += (good_share - bad_share) * value
    return woe, round(iv, 6)


def quantile_edges(values: Sequence[float], n_bins: int) -> list[float]:
    """Cut points at equal-frequency quantiles.

    Equal frequency, not equal width: equal-width bins on a skewed feature - income,
    balance, almost everything in lending - put 95% of the population in one bin and
    learn nothing.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean or n_bins < 2:
        return []
    edges = [clean[min(len(clean) - 1, int(len(clean) * i / n_bins))]
             for i in range(1, n_bins)]
    return sorted(set(edges))


def bin_numeric(
    name: str,
    values: Sequence[float | None],
    target: Sequence[int],
    *,
    n_bins: int = 5,
    edges: Sequence[float] | None = None,
    min_bin_fraction: float = 0.05,
) -> BinnedFeature:
    """Bin a numeric feature. `target` is 1 for bad (default), 0 for good.

    Bins smaller than `min_bin_fraction` of the population are merged into their
    neighbour: a bin of nine accounts produces a WoE that will not survive contact
    with next quarter's data.
    """
    if len(values) != len(target):
        raise ValueError("values and target must be the same length")

    cuts = list(edges) if edges is not None else quantile_edges(
        [v for v in values if v is not None], n_bins
    )

    bins: list[Bin] = []
    bounds = [-math.inf, *cuts, math.inf]
    for lower, upper in zip(bounds, bounds[1:]):
        bins.append(Bin(label=f"[{lower:g}, {upper:g})", lower=lower, upper=upper))
    missing_bin = Bin(label="missing", is_missing=True)

    for value, y in zip(values, target):
        target_bin = missing_bin if value is None else next(
            (b for b in bins if b.contains(value)), missing_bin
        )
        if y:
            target_bin.bads += 1
        else:
            target_bin.goods += 1

    # Merge undersized numeric bins into the next one along.
    population = len(values)
    merged: list[Bin] = []
    for b in bins:
        if merged and b.total < population * min_bin_fraction:
            previous = merged[-1]
            previous.upper = b.upper
            previous.goods += b.goods
            previous.bads += b.bads
            previous.label = f"[{previous.lower:g}, {previous.upper:g})"
        else:
            merged.append(b)

    if missing_bin.total:
        merged.append(missing_bin)

    woe, iv = _woe_and_iv(merged)
    return BinnedFeature(name=name, bins=merged, woe=woe, iv=iv)


def bin_categorical(
    name: str, values: Sequence[str | None], target: Sequence[int]
) -> BinnedFeature:
    """One bin per category, plus a bin for missing."""
    if len(values) != len(target):
        raise ValueError("values and target must be the same length")

    by_label: dict[str, Bin] = {}
    for value, y in zip(values, target):
        label = "missing" if value is None else str(value)
        b = by_label.setdefault(label, Bin(label=label, is_missing=value is None))
        if y:
            b.bads += 1
        else:
            b.goods += 1

    bins = sorted(by_label.values(), key=lambda b: (b.is_missing, b.label))
    woe, iv = _woe_and_iv(bins)
    return BinnedFeature(name=name, bins=bins, woe=woe, iv=iv)
