# Quant Finance Toolkit

A derivatives pricing, portfolio risk, and AI-augmented research toolkit,
built to demonstrate both classical quantitative finance fundamentals and
the AI/ML engineering skillset that increasingly complements them in modern
quant research and trading teams.

Sibling repo to [`quant-research-lab`](../quant-research-lab) (a
falsification-first systematic strategy research repo — walk-forward
validation, deflated Sharpe, permutation tests). That repo focuses on
*strategy research discipline*; this one focuses on *pricing, risk, and the
AI layer on top of them*. Cross-linked, not merged, on purpose — different
scope, same standard of honesty about what actually works.

---

## Why this exists

Two considerations shaped this repo directly:

1. **Core quantitative finance competency.** Options pricing theory, PDEs,
   numerical methods, and stochastic calculus are the foundation of
   quantitative research and analytics work, and C++ remains the standard
   for performance-critical pricing and risk infrastructure. Part A and
   Part F exist specifically to demonstrate those fundamentals solidly.
2. **AI/ML as a genuine differentiator, not bonus content.** Systematic
   trading and research desks are increasingly building in-house AI
   capability for signal generation, model interpretability, and risk
   tooling — a skillset that's rarer among candidates with strong classical
   derivatives math alone. Part E treats that combination as core to this
   toolkit, not an afterthought, which is what separates it from a generic
   derivatives-pricing course repo.

## Structure and status

| Part | Module | Status | What it proves |
|---|---|---|---|
| A | [`pricing/`](pricing/README.md) | ✅ Complete | Options pricing: closed-form, tree, Monte Carlo + variance reduction, PDE, exotics |
| D | [`risk/`](risk/README.md) | ✅ Complete | Markowitz optimization, VaR (3 methods) + Kupiec backtest, CVaR, stress testing |
| E | [`ai_augmented/`](ai_augmented/README.md) | ✅ Complete | ML signal generation (honestly benchmarked), SHAP interpretability, LLM research assistant |
| B | [`advanced_models/`](advanced_models/README.md) | ✅ Complete | Heston stochastic vol + Merton jump-diffusion smiles vs. Black-Scholes' flat one, discrete delta-hedging P&L study |
| F | [`perf/`](perf/README.md) | ✅ Complete | C++ Monte Carlo kernel (single- and multi-threaded) via pybind11, benchmarked against Python/NumPy |
| C | [`fixed_income/`](fixed_income/README.md) | ✅ Complete | Bond pricing/duration/convexity, yield curve bootstrapping, Vasicek short-rate model |

All six parts are complete, tested, and cross-validated against each other.

Built in that order deliberately — Part A is the most foundational and
comes first; Part D reuses the statistical-validation muscle already built
for `quant-research-lab`; Part E is treated as core, not "if there's time."
Parts B, F, and C add depth and breadth on top of a foundation that's
already solid, tested, and documented.

## The cross-module narrative

Every module states its own assumptions and, critically, **where it breaks
— usually demonstrated by a later module, not just asserted**:

- Part A's Black-Scholes pricer assumes constant volatility. Part B plots
  the Heston model's and the Merton jump-diffusion model's implied
  volatility smiles against Black-Scholes' flat one, on the same option
  chain — the limitation made visible, not just stated.
- Part A's binomial tree is only trustworthy once it's shown to *converge*
  to Black-Scholes as steps increase — that convergence proof is in the
  README with a real plot, not left as an exercise for the reader.
- Part D's VaR section reports the honest, textbook-defying finding that all
  three VaR methods performed similarly on this specific dataset — and the
  more robust finding underneath it: **every method's breach rate roughly
  triples during a real high-volatility window**, a structural procyclicality
  problem shared by all trailing-window risk models, not a one-method flaw.
- Part E's ML signal model is reported *not* beating a simple hand-crafted
  baseline out-of-sample — an honest, common real-world outcome, kept in the
  README rather than replaced with a cherry-picked comparison that flatters
  the ML model.
- Part F's naive, single-threaded C++ rewrite of Part A's Monte Carlo loop is
  reported *slower* than vectorized NumPy — "rewrite it in C++" is not
  automatically a speedup once the Python version already calls into
  SIMD-vectorized C internally. The multi-threaded C++ version, which
  actually parallelizes across cores, is what delivers the genuine 3-4.6x
  speedup — the more precise, defensible claim.

## Quick start

```bash
git clone <this-repo-url> && cd quant-finance-toolkit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Part A — derivatives pricing (no network needed)
python -m pytest tests/test_black_scholes.py tests/test_binomial_tree.py \
    tests/test_monte_carlo.py tests/test_pde_solver.py tests/test_exotics.py -v
python -m pricing.generate_figures

# Part D — risk management (fetches real market data; falls back to
# clearly-labeled synthetic data if no network access)
python -m pytest tests/test_var.py tests/test_cvar.py tests/test_portfolio.py -v
python -m risk.generate_figures

# Part E — AI-augmented research (downloads a small embedding model on first run)
python -m pytest tests/test_ai_augmented.py tests/test_research_assistant.py -v
python -m ai_augmented.generate_figures

# Part B — beyond Black-Scholes (no network needed)
python -m pytest tests/test_heston.py tests/test_merton.py tests/test_delta_hedging.py -v
python -m advanced_models.generate_figures

# Part F — C++ performance component (needs a C++ compiler; skips cleanly without one)
cd perf && python setup.py build_ext --inplace && cd ..
python -m pytest tests/test_perf.py -v
python -m perf.benchmark

# Part C — fixed income (no network needed)
python -m pytest tests/test_bonds.py tests/test_yield_curve.py tests/test_short_rate.py -v
python -m fixed_income.generate_figures

# Or just run everything:
python -m pytest tests/ -v
```

Each module's own README (linked in the table above) has the full detail:
the model, its assumptions, a worked example, and where it breaks down.

## Skills demonstrated

| Core competency | Where it's shown here |
|---|---|
| Options pricing theory, numerical methods (trees, PDE, Monte Carlo) | `pricing/` — five independent pricing methods, cross-validated against each other |
| Stochastic calculus, PDEs | `pricing/pde_solver.py` (Crank-Nicolson), `advanced_models/` (Heston SDEs, Merton jump-diffusion) |
| C++ for performance-critical pricing infrastructure | `perf/` — the Part A Monte Carlo kernel rewritten in C++ (single- and multi-threaded) via pybind11, benchmarked against Python with an honest result |
| Risk management, VaR, portfolio construction | `risk/` — Markowitz optimization, three VaR methods with a real Kupiec backtest, CVaR, stress testing |
| AI/ML engineering applied to quant research | `ai_augmented/` — ML signal generation (honestly benchmarked against a baseline), SHAP interpretability on two different models, an LLM research assistant with documented failure modes |
| Model interpretability | `ai_augmented/interpretability.py` — SHAP applied to both the signal model and a purpose-built risk (volatility) model |
| Ability to validate and stress-test models rigorously | `risk/var.py`'s Kupiec backtest; `quant-research-lab`'s walk-forward validation and permutation tests (sibling repo) |
| Fixed income, rates | `fixed_income/` — bond risk metrics, yield curve bootstrapping, Vasicek short-rate model |

## Disclaimer

All market data is real (fetched from Yahoo Finance) unless a figure or
output explicitly says "SYNTHETIC" — which happens automatically if live
data is unavailable, never silently. The `ai_augmented/research_assistant/`
corpus is entirely fictional analyst-note-style text written for this
project, not real reporting. Nothing in this repository is investment advice.
