"""Claude tool-use schemas for the orchestrator's live mode.

Four tools, not two "mega" specialist tools — Claude routes to whichever
granular function actually answers the question (and can call more than one
for a compound question). See the README for why this is a better fit for
real tool-use than collapsing everything behind a single
"ask_customer_analytics" dispatcher.
"""

TOOLS = [
    {
        "name": "ask_hr_it_policy",
        "description": (
            "Answer a question about company HR or IT policy — leave, expense claims, "
            "IT equipment/laptop requests, remote/hybrid work, new hire onboarding, "
            "password and account security, business travel, performance reviews, "
            "parental leave, or workplace conduct — by querying the policy knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The employee's HR/IT policy question, verbatim or lightly cleaned up.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_customer_churn_risk",
        "description": (
            "Look up a specific RetailPulse customer's churn risk score and key RFM "
            "(recency/frequency/monetary) stats by their customer ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID, e.g. CUST100123",
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_country_kpis",
        "description": (
            "Get aggregate customer KPIs (customer count, average churn probability, "
            "high-risk customer count, total revenue) for one of RetailPulse's markets: "
            "Singapore, Malaysia, Indonesia, or Thailand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "One of: Singapore, Malaysia, Indonesia, Thailand.",
                }
            },
            "required": ["country"],
        },
    },
    {
        "name": "get_churn_risk_drivers",
        "description": (
            "Get the churn model's top feature importances and overall accuracy/ROC-AUC. "
            "Use this for general questions like 'why are customers churning' or "
            "'how accurate is the churn model', not for a specific customer or country."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """You are RetailPulse's internal Ops Copilot, used by employees and managers.

You have four tools covering two different domains:
- ask_hr_it_policy: for employee HR/IT policy questions.
- get_customer_churn_risk, get_country_kpis, get_churn_risk_drivers: for customer/business analytics questions.

Rules:
1. Only answer using information returned by your tools. Do not use outside knowledge.
2. If a question needs more than one tool (e.g. it asks about both a policy and analytics data), call all the tools you need before answering, and clearly attribute each part of your answer to what it came from.
3. If a question doesn't fit either domain, or none of your tools can answer it, say so plainly rather than guessing.
4. Keep answers concise and practical.
"""
