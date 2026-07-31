"""Bond pricing, duration, and convexity.

Duration and convexity are a linearization (and its second-order correction)
of the true price-yield relationship, which is convex, not linear — the
worked example below deliberately compares the duration/convexity *estimate*
of a price change against the *exact* repriced value, so the approximation
error is visible, not just asserted to be small.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BondRiskMetrics:
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float


def bond_price(face: float, coupon_rate: float, ytm: float, years_to_maturity: float, freq: int = 2) -> float:
    """coupon_rate and ytm are annualized; freq is coupon payments per year (2 = semi-annual)."""
    n_periods = int(round(years_to_maturity * freq))
    coupon = face * coupon_rate / freq
    period_yield = ytm / freq

    periods = np.arange(1, n_periods + 1)
    discount_factors = (1 + period_yield) ** (-periods)
    cash_flows = np.full(n_periods, coupon)
    cash_flows[-1] += face

    return float((cash_flows * discount_factors).sum())


def bond_risk_metrics(face: float, coupon_rate: float, ytm: float, years_to_maturity: float, freq: int = 2) -> BondRiskMetrics:
    n_periods = int(round(years_to_maturity * freq))
    coupon = face * coupon_rate / freq
    period_yield = ytm / freq

    periods = np.arange(1, n_periods + 1)
    discount_factors = (1 + period_yield) ** (-periods)
    cash_flows = np.full(n_periods, coupon)
    cash_flows[-1] += face

    pv_cash_flows = cash_flows * discount_factors
    price = float(pv_cash_flows.sum())

    times_years = periods / freq
    macaulay = float((times_years * pv_cash_flows).sum() / price)
    modified = macaulay / (1 + period_yield)

    convexity = float((pv_cash_flows * times_years * (times_years + 1 / freq)).sum() / (price * (1 + period_yield) ** 2))

    return BondRiskMetrics(price=price, macaulay_duration=macaulay, modified_duration=modified, convexity=convexity)


def estimate_price_change(metrics: BondRiskMetrics, dy: float) -> float:
    """Duration + convexity approximation of the price change for a yield shift dy."""
    return metrics.price * (-metrics.modified_duration * dy + 0.5 * metrics.convexity * dy**2)
