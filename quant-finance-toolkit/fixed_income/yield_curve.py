"""Bootstrapping a zero-coupon yield curve from a small set of par-bond market quotes.

Each input instrument is assumed to be a par bond (priced at 100, coupon rate
equal to its quoted market yield) — the standard simplifying assumption for a
bootstrapping teaching example. Given zero rates for all earlier maturities,
each new instrument's coupon-bearing cash flows before its own maturity can
be discounted using already-known zero rates, leaving exactly one unknown
(the zero rate at that instrument's own maturity) to solve for algebraically.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ParInstrument:
    maturity_years: float
    par_yield: float  # also used as the annual coupon rate for coupon-bearing instruments
    freq: int = 2  # 1 = the instrument pays no coupon before maturity (money-market style)


def bootstrap_zero_curve(instruments: list[ParInstrument]) -> list[tuple[float, float]]:
    """Returns [(maturity, zero_rate), ...] sorted by maturity. Zero rates are continuously
    compounded for simplicity of discounting in later modules (short_rate.py uses the same convention).
    """
    instruments = sorted(instruments, key=lambda i: i.maturity_years)
    zero_rates: list[tuple[float, float]] = []  # (maturity, continuously-compounded zero rate)

    for inst in instruments:
        if inst.freq == 1 and inst.maturity_years <= 1.0:
            # Money-market style instrument: single payment at maturity, no interim coupons.
            # par_yield here is a simple annual rate: 100 -> 100*(1+par_yield*T) at maturity.
            zero_rate = np.log(1 + inst.par_yield * inst.maturity_years) / inst.maturity_years
            zero_rates.append((inst.maturity_years, float(zero_rate)))
            continue

        n_periods = int(round(inst.maturity_years * inst.freq))
        coupon = 100 * inst.par_yield / inst.freq
        period_times = np.array([(k + 1) / inst.freq for k in range(n_periods)])

        pv_known_coupons = 0.0
        for t in period_times[:-1]:
            z = _interpolate_zero_rate(zero_rates, t)
            pv_known_coupons += coupon * np.exp(-z * t)

        # Par condition: 100 = pv_known_coupons + (coupon + 100) * exp(-z_T * T)
        final_t = period_times[-1]
        remaining_pv = 100 - pv_known_coupons
        discount_factor = remaining_pv / (coupon + 100)
        if discount_factor <= 0:
            raise ValueError(f"Bootstrapping failed at maturity {final_t}: implied discount factor is non-positive")
        z_final = -np.log(discount_factor) / final_t
        zero_rates.append((final_t, float(z_final)))

    return zero_rates


def _interpolate_zero_rate(known_rates: list[tuple[float, float]], t: float) -> float:
    """Linear interpolation on already-bootstrapped zero rates for a coupon date that
    falls between two already-known maturities (or before the first one)."""
    if not known_rates:
        raise ValueError(f"No zero rates available yet to interpolate at t={t}")
    maturities = np.array([m for m, _ in known_rates])
    rates = np.array([z for _, z in known_rates])

    if t <= maturities[0]:
        return float(rates[0])
    if t >= maturities[-1]:
        return float(rates[-1])
    return float(np.interp(t, maturities, rates))


def discount_factor(zero_curve: list[tuple[float, float]], t: float) -> float:
    z = _interpolate_zero_rate(zero_curve, t)
    return float(np.exp(-z * t))
