# Power Automate / Copilot Studio Integration Stub

## Why this one is different from Project 2's

Project 2's stub proved the *shape* of the integration: a Power Automate flow calling
a FastAPI service, posting an Adaptive Card back to Teams. This one proves the
*reuse* — the exact same shape now fronts a copilot that spans two backend domains
(HR/IT policy and customer analytics) instead of one, and the flow definition doesn't
change at all to accommodate that. The routing complexity is entirely inside the
copilot; Power Automate's job stays exactly as simple as it was for a single-purpose
service.

As with Project 2: this is a hand-written, schema-accurate Workflow Definition
Language document, **not exported from a live Power Platform environment** (still no
licensed environment available to do that). See the honest gaps section below.

## What the flow does

```mermaid
sequenceDiagram
    participant User as Employee/Manager (Teams)
    participant Teams as Teams Channel / Copilot Studio topic
    participant Flow as Power Automate flow
    participant API as orchestrator/api.py (FastAPI)
    participant Agent as Orchestrator (Claude tool-use)
    participant HR as HR/IT policy service (Project 2)
    participant AN as Analytics tools (RetailPulse data)

    User->>Teams: "What's the churn risk for CUST100005?"
    Teams->>Flow: Trigger: new channel message / recognized topic
    Flow->>API: POST /ask { question, mode: "live" }
    API->>Agent: agent.ask(question)
    Agent->>AN: get_customer_churn_risk(...) [if analytics question]
    Agent->>HR: ask_hr_it_policy(...) [if policy question]
    AN-->>Agent: tool result
    HR-->>Agent: tool result
    Agent-->>API: answer + tool_calls trace
    API-->>Flow: JSON response
    Flow->>Flow: Build Adaptive Card (answer + which tool(s) handled it)
    Flow->>Teams: Post Adaptive Card as reply
    Teams-->>User: Answer, attributed to the right backend system
```

## Running the piece that actually exists

```bash
# Terminal 1 — the HR/IT policy service this copilot depends on:
cd ../genai-power-platform-agent
source .venv/bin/activate
uvicorn rag.api:app --port 8000

# Terminal 2 — the Ops Copilot's own service:
cd ../ops-copilot-agent
source .venv/bin/activate
uvicorn orchestrator.api:app --port 8100
```

```bash
curl -s -X POST http://localhost:8100/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the churn risk for CUST100005?"}'
```

## Honest gaps in this stub

- Same `shared_teams` connection placeholder issue as Project 2 — needs a real,
  licensed Teams connector to actually deploy.
- No auth between the flow and either FastAPI service — same production caveat as
  Project 2 (would need API Management or a shared secret, not exposed unauthenticated).
- This flow now has a *dependency chain*: Power Automate → Ops Copilot service →
  (sometimes) the HR/IT policy service. In a real deployment this ordering/availability
  dependency would need its own monitoring — if the policy service is down, the
  copilot's `ask_hr_it_policy` tool call fails gracefully (see
  `tools/policy.py::ask_hr_it_policy`), but that failure mode isn't modeled in this
  flow definition itself.
