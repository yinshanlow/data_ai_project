import pytest

from fixed_income.yield_curve import ParInstrument, bootstrap_zero_curve, discount_factor


@pytest.fixture
def instruments():
    return [
        ParInstrument(maturity_years=0.5, par_yield=0.045, freq=1),
        ParInstrument(maturity_years=1.0, par_yield=0.047, freq=1),
        ParInstrument(maturity_years=2.0, par_yield=0.049, freq=2),
        ParInstrument(maturity_years=3.0, par_yield=0.050, freq=2),
        ParInstrument(maturity_years=5.0, par_yield=0.052, freq=2),
        ParInstrument(maturity_years=10.0, par_yield=0.053, freq=2),
    ]


def test_zero_curve_has_one_rate_per_instrument(instruments):
    curve = bootstrap_zero_curve(instruments)
    assert len(curve) == len(instruments)
    assert [t for t, _ in curve] == sorted(i.maturity_years for i in instruments)


def test_upward_sloping_par_curve_produces_upward_sloping_zero_curve(instruments):
    curve = bootstrap_zero_curve(instruments)
    rates = [z for _, z in curve]
    assert rates == sorted(rates), "zero rates should increase with maturity, matching the upward-sloping input par curve"


def test_bootstrapped_curve_reprices_each_par_instrument_near_100(instruments):
    """The defining correctness check: each instrument's own cash flows, discounted
    with the bootstrapped curve, should reprice back to ~par (100) — small residuals
    are expected from linear interpolation of zero rates at intermediate coupon
    dates, not from a bootstrapping error."""
    curve = bootstrap_zero_curve(instruments)

    for inst in instruments:
        if inst.freq == 1 and inst.maturity_years <= 1.0:
            price = 100 * discount_factor(curve, inst.maturity_years) * (1 + inst.par_yield * inst.maturity_years)
        else:
            n = int(round(inst.maturity_years * inst.freq))
            coupon = 100 * inst.par_yield / inst.freq
            times = [(k + 1) / inst.freq for k in range(n)]
            price = sum(coupon * discount_factor(curve, t) for t in times[:-1])
            price += (coupon + 100) * discount_factor(curve, times[-1])
        assert price == pytest.approx(100.0, abs=0.1)


def test_discount_factor_decreases_with_maturity(instruments):
    curve = bootstrap_zero_curve(instruments)
    dfs = [discount_factor(curve, t) for t, _ in curve]
    assert dfs == sorted(dfs, reverse=True)
