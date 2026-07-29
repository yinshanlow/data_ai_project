# Ops Copilot — Multi-Agent Orchestration over RetailPulse

A third portfolio project that deliberately does **not** stand alone: it's an
orchestrator that routes between the other two projects — [`sg-data-ai-demo`](../sg-data-ai-demo)'s
customer analytics and [`genai-power-platform-agent`](../genai-power-platform-agent)'s
HR/IT policy RAG assistant — using real Claude tool-use, not a hardcoded if/else
classifier. The point of this project is to demonstrate **integration and
orchestration thinking**, the skill that separates a Solutions Architect trajectory
from "can build one more PoC."

**Why this exists:** the first two projects prove SQL/Python/ML fundamentals and
GenAI+Power Platform integration separately. Real enterprise "agentic AI" pitches in
2026 aren't about a single chatbot — they're about one front door that intelligently
dispatches across a company's existing systems. This project is that pattern, built
small enough to actually run and test end to end.

---

## The business narrative

> RetailPulse (same fictional e-commerce company as Project 1) rolls out an internal
> "Ops Copilot." A regional manager asks *"what's the churn risk for CUST100005?"* —
> routed to the analytics tools, which query the actual scored customer data from
> Project 1. An employee asks *"how many days of annual leave do I get?"* — routed to
> the HR/IT policy service from Project 2. The copilot doesn't know in advance which
> kind of question is coming; an orchestrator decides, live, using Claude's tool-use
> API — the same mechanism a production "agentic AI" deployment would use.

---

## Architecture

```mermaid
flowchart TB
    U[User question] --> O[Orchestrator<br/>Claude tool-use, or a<br/>rule-based router in mock mode]
    O -->|ask_hr_it_policy| A1[HR/IT Policy Tool<br/>HTTP call to Project 2's rag/api.py]
    O -->|get_customer_churn_risk| A2[Analytics Tool<br/>reads RetailPulse's customer_scored.csv]
    O -->|get_country_kpis| A3[Analytics Tool<br/>country-level aggregates]
    O -->|get_churn_risk_drivers| A4[Analytics Tool<br/>feature importance + model metrics]
    O -->|no matching tool| F[Honest fallback:<br/>"outside what this copilot covers"]
    A1 --> R[Tool results]
    A2 --> R
    A3 --> R
    A4 --> R
    R --> O
    O --> S[Final answer +<br/>full tool-call trace]
    S --> UI[Streamlit chat UI]
    S --> API[orchestrator/api.py FastAPI /ask]
    API --> PA[Power Automate / Teams stub]
```

### Two design decisions worth calling out

**Four granular tools, not two "mega" specialist agents.** The original spec framed
this as "an HR/IT agent and an analytics agent." In practice, exposing Claude to four
small, precisely-scoped tools (`ask_hr_it_policy`, `get_customer_churn_risk`,
`get_country_kpis`, `get_churn_risk_drivers`) produces better real tool-use than
collapsing three different analytics lookups behind one dispatcher — Claude's
native tool selection *is* the router, so there's no reason to build a second,
hidden router inside a single "analytics" tool.

**The HR/IT policy tool is an HTTP call to Project 2's actual FastAPI service, not a
Python import of its internals.** Project 2's `rag/api.py` was built specifically so
more than one caller could reuse it (that was the whole premise of its Power Automate
stub). This project is the second real caller — proof that the service boundary
was worth building, not just decoration. The trade-off: local testing needs Project
2's service running as well (see Quick start below), which is a bit more operational
overhead than a single-process demo, but it's the architecturally honest choice.

**Analytics tools are a small fixed set of typed functions, not open text-to-SQL.**
Same reasoning as Project 2's grounding gate: an LLM (or a mock-mode regex) freely
constructing queries against customer data is a real injection/hallucination surface
for a project this size. `get_customer_churn_risk`, `get_country_kpis`, and
`get_churn_risk_drivers` cover the realistic query shapes without that risk.

---

## Quick start (no API key needed)

This needs **two services running** — that's the cost of the HTTP-reuse design above.

```bash
# Terminal 1 — Project 2's policy service (already built, being reused here)
cd genai-power-platform-agent
source .venv/bin/activate            # see that project's README to set it up first
uvicorn rag.api:app --port 8000

# Terminal 2 — this project
cd ops-copilot-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m eval.run_eval              # runs the 20-question eval set, mock mode, no API key
streamlit run app/streamlit_app.py
```

Everything above runs in **mock mode** by default — a rule-based keyword router with
zero API calls and zero cost, same zero-cost-by-default philosophy as Project 2.

### Enabling live mode (optional)

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY=sk-ant-... in .env
```

With a key set, the orchestrator uses real Claude tool-use to decide routing —
including handling compound questions across both domains in one turn, which mock
mode cannot do (see the honest failure case below).

---

## Evaluation

`eval/eval_questions.json` has 20 questions across 7 categories: `hr_it`,
`customer_lookup`, `invalid_customer` (hallucination check), `country_kpi`,
`churn_drivers`, `out_of_scope`, `compound` (needs two tools), and `ambiguous`
(sounds like it could be analytics but only the policy tool can actually answer it).

Latest run (mock mode):

| Metric | Result |
|---|---|
| Routing accuracy (expected tool(s) called) | 90% (18/20) |
| Keyword grounding accuracy | 88% (15/17) |
| Hallucination flags | 0 |
| Correct refusal rate (out-of-scope) | 100% (2/2) |
| Compound-question full-success rate | 0% (0/2) |

Full per-question table: [`eval/eval_results.md`](eval/eval_results.md). The 10%
routing gap and the 12% keyword gap are the exact same two questions — both are the
deliberately compound ones, discussed below, not scattered failures.

---

## Honest failure case — what I caught and fixed, and what's a real remaining limitation

**Caught and fixed (real bugs, not intentional trade-offs).** The first version of the
mock router's keyword list included the bare word `"policy"` as an HR/IT signal, and
matched country names as plain substrings. Running the eval set exposed two concrete
failures: *"What's Meridian Holdings' stock buyback policy for shareholders?"* got
routed to the HR/IT policy tool purely because it contains the word "policy," and
*"What's the weather like in Singapore today?"* got routed to `get_country_kpis`
purely because it mentions Singapore. Neither question has anything to do with either
tool's actual purpose. Fix: dropped the generic `"policy"` keyword in favor of
specific policy-domain terms, and required a country match to co-occur with an actual
analytics-context word (customer, revenue, churn, KPI, etc.) before routing to
`get_country_kpis`. Re-running the eval after the fix: correct refusal rate on
out-of-scope questions went from 0% → 100%, with routing accuracy improving from 75%
→ 90% overall.

**The honest limitation that remains.** Mock mode's router picks exactly one tool per
question, by design — it's a simple rule-based classifier, not a reasoning system. On
the two deliberately compound questions in the eval set (*"How many days of annual
leave do I get, and what's the churn risk for CUST100000?"*), mock mode answers only
the half it matches first and silently drops the other half. This is not a bug to
patch with more keyword rules — it's the actual capability gap between a keyword
router and real LLM tool-use, and it's the clearest evidence for why live mode's
tool-use loop (which can call multiple tools in one turn and synthesize both results)
is the real product here, not the free fallback. **This project's live mode was
implemented but not benchmarked against a real Anthropic API key while building it**
(no key was used, to avoid incurring cost) — if you add one via `.env`, running
`python -m eval.run_eval --mode live` and comparing the compound-question success rate
against mock mode's 0% is the single most interesting thing to check.

---

## Power Automate integration

See [`power_automate/README.md`](power_automate/README.md) and
[`power_automate/ops_copilot_flow_definition.json`](power_automate/ops_copilot_flow_definition.json).
Same integration pattern as Project 2 — a Teams/Copilot Studio flow calling a FastAPI
`/ask` endpoint — now proven to reuse cleanly across a copilot spanning two domains
instead of one. Still explicitly **not deployed** (no licensed Power Platform
environment available).

---

## Mapping this project to what employers actually ask for

| Job requirement (from real SG postings) | Where it's shown here |
|---|---|
| "Agentic AI / multi-agent orchestration" — the fastest-growing GenAI line item in 2026 SG postings | `orchestrator/agent.py`'s live-mode tool-use loop |
| "Experience with LLM tool-use / function calling" | `orchestrator/tool_schemas.py`, `orchestrator/agent.py` |
| "Ability to design system integrations across existing platforms" — a Solutions Architect-level ask | The HTTP-reuse decision documented above; `tools/policy.py` |
| "Evaluation and testing discipline for agentic systems, not just single-turn chatbots" | `eval/run_eval.py`'s routing-accuracy + compound-question breakout |
| "Can communicate trade-offs and limitations to both technical and business stakeholders" | The honest failure case section above |

---

## What's not done / possible extensions

- Live mode's compound-question handling is implemented but unbenchmarked (see above)
  — the most valuable next step if a key becomes available.
- No conversation memory, same as Project 2 — each question is independent.
- The mock router's keyword list is inherently brittle; it's a deliberately simple
  fallback, not a second real classifier. It will keep needing small fixes as new
  edge cases turn up — that's expected of a keyword heuristic, not a sign the design
  is wrong.
- Natural extension: a genuine multi-turn conversation ("show me CUST100005's risk" →
  "now show me Malaysia's KPIs for comparison") would need session-level memory in the
  orchestrator, not just per-question routing.

---

## Disclaimer

All company names, customer data, and evaluation questions in this repository are
synthetic or reused from this repo's own earlier synthetic datasets. No real company,
employer, or customer data is used anywhere in this project.
