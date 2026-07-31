"""Interactive Streamlit dashboard for the Quant Finance Toolkit.

Wraps the already-tested library code in pricing/, risk/, ai_augmented/,
advanced_models/, and fixed_income/ — every number shown here comes from the
same functions covered by tests/, not reimplemented for the UI.

Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_models.delta_hedging import hedging_frequency_study
from advanced_models.heston import HestonParams, heston_smile
from advanced_models.merton_jump_diffusion import merton_smile
from ai_augmented.features import FEATURE_COLUMNS, build_factor_panel
from ai_augmented.interpretability import explain_model
from ai_augmented.risk_model import RISK_FEATURE_COLUMNS, build_volatility_panel, train_volatility_model
from ai_augmented.signal_model import chronological_split, compare_baseline_vs_ml
from fixed_income.bonds import bond_price, bond_risk_metrics, estimate_price_change
from fixed_income.short_rate import VasicekParams, simulate_vasicek_paths, vasicek_yield
from fixed_income.yield_curve import ParInstrument, bootstrap_zero_curve, discount_factor
from pricing.binomial_tree import binomial_price, convergence_to_bs
from pricing.black_scholes import bs_greeks, bs_price
from pricing.exotics import asian_arithmetic_call, barrier_down_and_out_call
from pricing.monte_carlo import antithetic_mc_price, control_variate_mc_price, naive_mc_price
from pricing.pde_solver import crank_nicolson_price
from risk.cvar import apply_historical_scenario, crisis_period_total_return, historical_cvar, parametric_cvar
from risk.data import fetch_prices, prices_to_returns
from risk.portfolio import efficient_frontier, max_sharpe_portfolio, min_variance_portfolio
from risk.var import historical_var, kupiec_test, monte_carlo_var, parametric_var, rolling_var_backtest

st.set_page_config(page_title="Quant Finance Toolkit", page_icon="📈", layout="wide")

st.title("📈 Quant Finance Toolkit")
st.caption(
    "Derivatives pricing, portfolio risk, and AI-augmented quant research — interactive demo. "
    "Every chart here calls the same tested functions as `tests/`, not a separate UI-only implementation. "
    "[Full source & READMEs on GitHub](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit)."
)

tabs = st.tabs([
    "Overview", "A · Pricing", "B · Beyond Black-Scholes", "D · Risk",
    "E · AI-Augmented", "F · C++ Performance", "C · Fixed Income",
])


# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.header("Why this toolkit is built the way it is")
    st.markdown(
        """
Built to demonstrate two things together: **core quantitative finance fundamentals** (options
pricing, PDEs, stochastic calculus, C++) and **AI/ML engineering applied to quant research** —
increasingly the more differentiated combination, not just classical derivatives math on its own.

Every module cross-validates against another and states where the simpler model breaks down —
not just in the READMEs, but with a real number or plot proving it. A few highlights:
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Tests passing", "81 / 81")
    col2.metric("Modules", "6")
    col3.metric("MC variance cut (control variate)", "66%")

    st.markdown(
        """
- **Part A**: the binomial tree is shown *converging* to Black-Scholes as steps increase — not just implemented.
- **Part D**: on real market data, every VaR method's breach rate roughly **triples** during a genuine
  high-volatility window — a structural procyclicality finding, not a one-method flaw.
- **Part E** (the differentiator): the ML signal model is reported **not** beating a hand-crafted
  baseline out-of-sample — an honest, common real-world result, kept in rather than hidden.
- **Part F**: a naive single-threaded C++ rewrite is honestly reported **slower** than vectorized
  NumPy; the multi-threaded version is what delivers a genuine speedup.

Full write-ups: [pricing](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/pricing) ·
[risk](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/risk) ·
[ai_augmented](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/ai_augmented) ·
[advanced_models](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/advanced_models) ·
[perf](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/perf) ·
[fixed_income](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/fixed_income)
        """
    )


# ---------------------------------------------------------------- Part A
with tabs[1]:
    st.header("Part A — Derivatives Pricing")
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot (S)", value=100.0, min_value=1.0, key="a_S")
    K = c2.number_input("Strike (K)", value=100.0, min_value=1.0, key="a_K")
    T = c3.number_input("Maturity (years)", value=1.0, min_value=0.05, key="a_T")
    r = c4.number_input("Risk-free rate", value=0.05, format="%.3f", key="a_r")
    sigma = c5.number_input("Volatility", value=0.20, format="%.3f", key="a_sigma")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True, key="a_type")

    st.subheader("Black-Scholes closed form")
    greeks = bs_greeks(S, K, T, r, sigma, option_type)
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Price", f"{greeks.price:.4f}")
    g2.metric("Delta", f"{greeks.delta:.4f}")
    g3.metric("Gamma", f"{greeks.gamma:.4f}")
    g4.metric("Vega", f"{greeks.vega:.4f}")
    g5.metric("Theta/yr", f"{greeks.theta:.4f}")
    g6.metric("Rho", f"{greeks.rho:.4f}")

    st.subheader("Binomial tree — convergence to Black-Scholes")
    with st.spinner("Computing tree at increasing step counts..."):
        conv = convergence_to_bs(S, K, T, r, sigma, option_type, step_counts=(5, 10, 25, 50, 100, 200, 400, 800))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    steps, prices = zip(*conv)
    ax.plot(steps, prices, marker="o", label="Binomial tree")
    ax.axhline(greeks.price, color="crimson", linestyle="--", label=f"Black-Scholes = {greeks.price:.4f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    amer_tree = binomial_price(S, K, T, r, sigma, 300, option_type, "american")
    st.caption(f"300-step American {option_type} (early exercise): **{amer_tree:.4f}**")

    st.subheader("Monte Carlo — variance reduction")
    n_paths = st.slider("Number of paths", 10_000, 500_000, 100_000, step=10_000, key="a_npaths")
    if st.button("Run Monte Carlo comparison", key="a_mc_button"):
        with st.spinner("Simulating..."):
            naive = naive_mc_price(S, K, T, r, sigma, n_paths, option_type, seed=1)
            anti = antithetic_mc_price(S, K, T, r, sigma, n_paths, option_type, seed=1)
            cv = control_variate_mc_price(S, K, T, r, sigma, n_paths, option_type, seed=1)
        mc_df = pd.DataFrame([
            {"Estimator": "Naive MC", "Price": naive.price, "Std error": naive.std_error},
            {"Estimator": "Antithetic variates", "Price": anti.price, "Std error": anti.std_error},
            {"Estimator": "Control variate", "Price": cv.price, "Std error": cv.std_error},
        ])
        st.dataframe(mc_df.style.format({"Price": "{:.4f}", "Std error": "{:.5f}"}), use_container_width=True)
        st.caption(f"Black-Scholes reference: {greeks.price:.4f}")

    st.subheader("PDE (Crank-Nicolson) — American option")
    pde = crank_nicolson_price(S, K, T, r, sigma, option_type, "american", M=250, N=250)
    st.caption(f"PDE American {option_type}: **{pde.price:.4f}** (tree above: {amer_tree:.4f})")

    st.subheader("Exotics (Monte Carlo)")
    e1, e2 = st.columns(2)
    with e1:
        barrier = e1.number_input("Barrier (down-and-out call)", value=80.0, max_value=S - 1)
        barrier_result = barrier_down_and_out_call(S, K, T, r, sigma, barrier, n_paths=100_000, n_steps=100, seed=1)
        e1.metric("Down-and-out call", f"{barrier_result.price:.4f}", f"± {barrier_result.std_error:.4f}")
    with e2:
        asian_result = asian_arithmetic_call(S, K, T, r, sigma, n_paths=100_000, n_steps=100, seed=1)
        e2.metric("Arithmetic Asian call", f"{asian_result.price:.4f}", f"± {asian_result.std_error:.4f}")


# ---------------------------------------------------------------- Part B
with tabs[2]:
    st.header("Part B — Beyond Black-Scholes")
    st.markdown("The smile Black-Scholes (flat, by construction) cannot produce — same option chain, two mechanisms.")

    b1, b2, b3 = st.columns(3)
    rho = b1.slider("Heston ρ (correlation)", -0.95, 0.0, -0.7, step=0.05)
    xi = b2.slider("Heston ξ (vol-of-vol)", 0.1, 0.8, 0.4, step=0.05)
    jump_intensity = b3.slider("Merton jump intensity (λ)", 0.0, 2.0, 0.8, step=0.1)

    if st.button("Generate smile comparison", key="b_smile_button"):
        with st.spinner("Simulating Heston + pricing Merton across strikes..."):
            S0, T0, r0 = 100, 1.0, 0.05
            strikes = [70, 80, 90, 100, 110, 120, 130]
            heston_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=rho)
            heston_results = heston_smile(S0, strikes, T0, r0, heston_params, n_paths=100_000, seed=10)
            merton_results = merton_smile(S0, strikes, T0, r0, sigma=0.20, jump_intensity=jump_intensity, jump_mean=-0.08, jump_vol=0.2)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.axhline(0.20, color="crimson", linestyle="--", label="Black-Scholes (flat)")
        ax.plot(strikes, [iv for _, _, iv in heston_results], marker="o", label=f"Heston (ρ={rho})")
        ax.plot(strikes, [iv for _, _, iv in merton_results], marker="s", label=f"Merton (λ={jump_intensity})")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied volatility")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Discrete delta-hedging: P&L variance vs. rebalancing frequency")
    if st.button("Run hedging frequency study", key="b_hedge_button"):
        with st.spinner("Simulating hedged portfolios at several rebalancing frequencies..."):
            results = hedging_frequency_study(100, 100, 1.0, 0.05, 0.2, frequencies=(12, 52, 252, 1000), n_paths=15_000, seed=0)
        hedge_df = pd.DataFrame([{"Rebalances/year": r.rebalances_per_year, "P&L mean": r.pnl_mean, "P&L std dev": r.pnl_std} for r in results])
        st.dataframe(hedge_df.style.format({"P&L mean": "{:.4f}", "P&L std dev": "{:.4f}"}), use_container_width=True)
        st.caption("Std dev should roughly halve every time frequency roughly quadruples (O(1/√n) scaling).")


# ---------------------------------------------------------------- Part D
with tabs[3]:
    st.header("Part D — Portfolio Theory & Risk")

    @st.cache_data(show_spinner="Fetching market data (falls back to synthetic data if offline)...")
    def _load_prices():
        prices, is_synthetic = fetch_prices(period="5y")
        return prices, is_synthetic

    prices, is_synthetic = _load_prices()
    returns = prices_to_returns(prices)
    if is_synthetic:
        st.warning("Live market data unavailable — using clearly-labeled SYNTHETIC price data instead.")
    else:
        st.success(f"Real market data: {prices.index.min().date()} to {prices.index.max().date()} ({len(prices)} trading days)")

    st.subheader("Efficient frontier")
    mv = min_variance_portfolio(returns)
    ms = max_sharpe_portfolio(returns, rf=0.04)
    frontier = efficient_frontier(returns, n_points=30)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(frontier["volatility"], frontier["return"], color="steelblue", label="Efficient frontier")
    ax.scatter([mv.volatility], [mv.expected_return], color="crimson", s=80, zorder=5, label="Min-variance")
    ax.scatter([ms.volatility], [ms.expected_return], color="darkorange", s=80, zorder=5, label="Max-Sharpe (rf=4%)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    w1, w2 = st.columns(2)
    w1.write("**Min-variance weights**")
    w1.dataframe(mv.weights.round(3))
    w2.write("**Max-Sharpe weights**")
    w2.dataframe(ms.weights.round(3))

    st.subheader("VaR: three ways, backtested with Kupiec")
    if st.button("Run VaR backtest (rolling 250-day window)", key="d_var_button"):
        portfolio_returns = returns.mean(axis=1)
        rows = []
        with st.spinner("Backtesting historical, parametric, and Monte Carlo VaR..."):
            for method in ["historical", "parametric", "monte_carlo"]:
                bt = rolling_var_backtest(portfolio_returns, method, window=250, confidence=0.95, seed=1)
                kt = kupiec_test(bt["breached"], confidence=0.95)
                rows.append({"Method": method, "Breach rate": kt.breach_rate, "Expected rate": kt.expected_rate,
                             "Kupiec p-value": kt.p_value, "Rejected at 5%?": kt.reject_at_5pct})
        st.dataframe(pd.DataFrame(rows).style.format({"Breach rate": "{:.3f}", "Expected rate": "{:.3f}", "Kupiec p-value": "{:.3f}"}), use_container_width=True)

    st.subheader("CVaR (Expected Shortfall)")
    portfolio_returns = returns.mean(axis=1)
    cv1, cv2 = st.columns(2)
    cv1.metric("Historical VaR (95%)", f"{historical_var(portfolio_returns):.4f}")
    cv1.metric("Historical CVaR (95%)", f"{historical_cvar(portfolio_returns):.4f}")
    cv2.metric("Parametric VaR (95%)", f"{parametric_var(portfolio_returns):.4f}")
    cv2.metric("Parametric CVaR (95%)", f"{parametric_cvar(portfolio_returns):.4f}")

    st.subheader("Stress test: real historical crisis window")
    crisis_start, crisis_end = "2025-03-15", "2025-05-15"
    try:
        scenario = crisis_period_total_return(prices, crisis_start, crisis_end)
        stress = apply_historical_scenario(ms.weights, scenario)
        st.metric(f"Max-Sharpe portfolio under {crisis_start} → {crisis_end}", f"{stress.portfolio_shock:+.2%}")
    except ValueError:
        st.info("Crisis window not covered by the currently loaded price history (this happens with synthetic fallback data).")


# ---------------------------------------------------------------- Part E
with tabs[4]:
    st.header("Part E — AI-Augmented Quant Research")

    st.subheader("ML-based signal generation — an honest comparison")
    if st.button("Train and compare signal models", key="e_signal_button"):
        with st.spinner("Building factor panel and training XGBoost..."):
            panel = build_factor_panel(prices)
            baseline, ml, model = compare_baseline_vs_ml(panel)
            _, test = chronological_split(panel)
            importance_df, _ = explain_model(model, test[FEATURE_COLUMNS])

        sig_df = pd.DataFrame([
            {"Model": baseline.name, "OOS IC": baseline.ic, "IC p-value": baseline.ic_p_value, "Long/short Sharpe": baseline.long_short_sharpe},
            {"Model": ml.name, "OOS IC": ml.ic, "IC p-value": ml.ic_p_value, "Long/short Sharpe": ml.long_short_sharpe},
        ])
        st.dataframe(sig_df.style.format({"OOS IC": "{:.4f}", "IC p-value": "{:.3f}", "Long/short Sharpe": "{:.3f}"}), use_container_width=True)
        if ml.ic < baseline.ic:
            st.info("Honest result: the ML model does **not** beat the simple hand-crafted baseline here — reported as-is, not hidden.")

        st.write("**SHAP feature importance — return-signal model**")
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#4C72B0")
        ax.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Model interpretability — volatility forecasting")
    if st.button("Train volatility model and explain with SHAP", key="e_vol_button"):
        with st.spinner("Training and explaining..."):
            vol_panel = build_volatility_panel(prices)
            vol_model, _, vol_test = train_volatility_model(vol_panel)
            vol_importance, _ = explain_model(vol_model, vol_test[RISK_FEATURE_COLUMNS])
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(vol_importance["feature"][::-1], vol_importance["mean_abs_shap"][::-1], color="#55A868")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("What drives the volatility forecast?")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Trailing realized volatility dominating the forecast reproduces volatility clustering — a well-known real stylized fact, a sanity check that the model learned something real.")

    st.subheader("LLM research assistant")
    st.caption("Research-idea generator over a small synthetic news corpus — not a source of truth. See ai_augmented/README.md for documented failure modes.")

    @st.cache_resource(show_spinner="Loading the research assistant (embedding model)...")
    def _load_assistant():
        from ai_augmented.research_assistant.pipeline import LIVE_MODE_AVAILABLE, ResearchAssistant
        return ResearchAssistant(), LIVE_MODE_AVAILABLE

    assistant, live_available = _load_assistant()
    if live_available:
        st.success("ANTHROPIC_API_KEY detected — live mode available.")
    else:
        st.info("No ANTHROPIC_API_KEY — running in mock mode (extractive, zero cost).")

    question = st.text_input("Ask about AAPL/MSFT/GOOGL/AMZN or market conditions:", key="e_ra_question")
    if question:
        with st.spinner("Retrieving..."):
            answer = assistant.ask(question)
        st.markdown(answer.answer)
        if answer.citations:
            with st.expander("Sources"):
                for c in answer.citations:
                    st.markdown(f"- {c}")


# ---------------------------------------------------------------- Part F
with tabs[5]:
    st.header("Part F — C++ Performance Component")
    st.markdown(
        """
The Part A Monte Carlo pricer's inner loop, rewritten in C++ (pybind11), benchmarked against
Python/NumPy. **Building C++ live in this hosted environment isn't attempted here** — the numbers
below are the actual measured results from building and running it locally (see
[`perf/README.md`](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit/perf)
for the full write-up and how to reproduce it).
        """
    )
    perf_df = pd.DataFrame([
        {"Paths": "1,000,000", "Python/NumPy": "0.0136s", "C++ single-threaded": "0.0257s", "C++ multi-threaded": "0.0042s", "Speedup vs. Python": "3.3x"},
        {"Paths": "5,000,000", "Python/NumPy": "0.0897s", "C++ single-threaded": "0.1277s", "C++ multi-threaded": "0.0196s", "Speedup vs. Python": "4.6x"},
        {"Paths": "20,000,000", "Python/NumPy": "0.3646s", "C++ single-threaded": "0.5112s", "C++ multi-threaded": "0.0803s", "Speedup vs. Python": "4.5x"},
    ])
    st.dataframe(perf_df, use_container_width=True)
    st.warning(
        "Honest finding: the naive single-threaded C++ loop is **slower** than vectorized NumPy at every "
        "scale — NumPy's own RNG and array ops are SIMD-vectorized C code, so a scalar C++ loop has no "
        "structural advantage. Only the multi-threaded (`std::thread`) version, which actually parallelizes "
        "across cores, delivers a genuine 3-4.6x speedup."
    )
    fig_path = Path(__file__).resolve().parent.parent / "perf" / "figures" / "cpp_speedup.png"
    if fig_path.exists():
        st.image(str(fig_path))


# ---------------------------------------------------------------- Part C
with tabs[6]:
    st.header("Part C — Fixed Income & Rates")

    st.subheader("Bond pricing, duration, convexity")
    bc1, bc2, bc3, bc4 = st.columns(4)
    face = bc1.number_input("Face value", value=100.0)
    coupon_rate = bc2.number_input("Coupon rate", value=0.05, format="%.3f")
    ytm = bc3.number_input("Yield to maturity", value=0.05, format="%.3f")
    maturity = bc4.number_input("Maturity (years)", value=10.0, min_value=0.5)

    metrics = bond_risk_metrics(face, coupon_rate, ytm, maturity, freq=2)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{metrics.price:.4f}")
    m2.metric("Macaulay duration", f"{metrics.macaulay_duration:.3f}")
    m3.metric("Modified duration", f"{metrics.modified_duration:.3f}")
    m4.metric("Convexity", f"{metrics.convexity:.3f}")

    dy = st.slider("Yield shock (bps)", -300, 300, 100, step=25) / 10000
    estimated = estimate_price_change(metrics, dy)
    exact = bond_price(face, coupon_rate, ytm + dy, maturity, freq=2) - metrics.price
    e1, e2 = st.columns(2)
    e1.metric("Duration+convexity estimate", f"{estimated:+.4f}")
    e2.metric("Exact repriced change", f"{exact:+.4f}")

    st.subheader("Yield curve bootstrapping")
    default_instruments = pd.DataFrame([
        {"Maturity (years)": 0.5, "Par yield": 0.045, "Freq": 1},
        {"Maturity (years)": 1.0, "Par yield": 0.047, "Freq": 1},
        {"Maturity (years)": 2.0, "Par yield": 0.049, "Freq": 2},
        {"Maturity (years)": 3.0, "Par yield": 0.050, "Freq": 2},
        {"Maturity (years)": 5.0, "Par yield": 0.052, "Freq": 2},
        {"Maturity (years)": 10.0, "Par yield": 0.053, "Freq": 2},
    ])
    edited = st.data_editor(default_instruments, num_rows="dynamic", key="c_curve_editor")
    instruments = [ParInstrument(row["Maturity (years)"], row["Par yield"], int(row["Freq"])) for _, row in edited.iterrows()]
    try:
        curve = bootstrap_zero_curve(instruments)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(edited["Maturity (years)"], edited["Par yield"], marker="s", label="Input par yields")
        ax.plot([t for t, _ in curve], [z for _, z in curve], marker="o", label="Bootstrapped zero curve")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    except ValueError as exc:
        st.error(f"Bootstrapping failed: {exc}")

    st.subheader("Vasicek short-rate model")
    v1, v2, v3, v4 = st.columns(4)
    r0 = v1.number_input("r0", value=0.03, format="%.3f")
    kappa = v2.number_input("κ (mean reversion)", value=0.5, format="%.2f")
    theta = v3.number_input("θ (long-run mean)", value=0.04, format="%.3f")
    vol = v4.number_input("σ", value=0.02, format="%.3f")

    vasicek_params = VasicekParams(kappa=kappa, theta=theta, sigma=vol)
    paths = simulate_vasicek_paths(r0, 10, vasicek_params, n_paths=25, n_steps=300, seed=1)
    time_grid = np.linspace(0, 10, 301)
    maturities = np.linspace(0.25, 15, 30)
    yields = [vasicek_yield(r0, Tm, vasicek_params) for Tm in maturities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for path in paths:
        ax1.plot(time_grid, path, alpha=0.4, linewidth=0.8)
    ax1.axhline(theta, color="black", linestyle="--", label="θ")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Short rate")
    ax1.legend()
    ax2.plot(maturities, yields, color="steelblue")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Yield")
    st.pyplot(fig)
    plt.close(fig)
