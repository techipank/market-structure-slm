"""Support and resistance derived from confirmed swings.

A level is not one swing; it is a *cluster* of swings that occurred at
approximately the same price. Price revisiting 452.10, 452.60 and 451.90 over
three months is one level touched three times, and reporting it as three
levels is both wrong and expensive in prompt tokens.

Clustering is greedy in volatility units: swings within ``level_cluster_atr``
ATR of a running cluster mean join it. Greedy rather than k-means because it
is deterministic, order-stable, needs no k, and runs left-to-right, which
keeps truncation invariance trivially true.

Support vs resistance is assigned relative to the *current* price, not to how
the level formed. A ceiling that price closes above becomes a floor; that is
the whole idea of polarity, and hard-coding a level as "resistance" because it
started as a swing high would misdescribe half of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_engine.swings import SwingPoint


@dataclass(frozen=True)
class PriceLevel:
    price: float
    #: How many confirmed swings sit inside the cluster.
    touches: int
    #: Bar index of the first and most recent swing in the cluster.
    first_index: int
    last_index: int
    #: "SUPPORT" when below the reference price, "RESISTANCE" when above.
    side: str
    #: Distance from the reference price, in ATR multiples. Signed: negative
    #: means the level is below current price.
    distance_atr: float


def build_levels(
    swings: list[SwingPoint],
    reference_price: float,
    atr_value: float,
    cluster_atr: float,
    min_touches: int,
    max_per_side: int,
) -> list[PriceLevel]:
    """Cluster confirmed swing prices into levels around `reference_price`.

    `swings` must already be filtered to those confirmed at or before the bar
    being described - this function has no notion of time beyond the indices
    it is handed.
    """
    if not swings or not np.isfinite(atr_value) or atr_value <= 0:
        return []

    tolerance = atr_value * cluster_atr
    ordered = sorted(swings, key=lambda s: s.price)

    clusters: list[list[SwingPoint]] = []
    for swing in ordered:
        if clusters and abs(swing.price - _mean_price(clusters[-1])) <= tolerance:
            clusters[-1].append(swing)
        else:
            clusters.append([swing])

    levels: list[PriceLevel] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        price = _mean_price(cluster)
        distance = (price - reference_price) / atr_value
        levels.append(
            PriceLevel(
                price=price,
                touches=len(cluster),
                first_index=min(s.index for s in cluster),
                last_index=max(s.index for s in cluster),
                side="RESISTANCE" if price > reference_price else "SUPPORT",
                distance_atr=distance,
            )
        )

    # Keep the nearest levels on each side. Nearest, not strongest: a
    # 40-touch level 15 ATR away is history, while the two levels bracketing
    # current price are what any description of "where we are" needs.
    supports = sorted(
        (lv for lv in levels if lv.side == "SUPPORT"), key=lambda lv: -lv.price
    )[:max_per_side]
    resistances = sorted(
        (lv for lv in levels if lv.side == "RESISTANCE"), key=lambda lv: lv.price
    )[:max_per_side]
    return sorted([*supports, *resistances], key=lambda lv: lv.price)


def _mean_price(cluster: list[SwingPoint]) -> float:
    return float(sum(s.price for s in cluster) / len(cluster))
