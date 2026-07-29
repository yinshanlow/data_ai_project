# Evaluation Results

Mode used: **mock**

- Questions evaluated: 20
- Routing accuracy (expected tool(s) called): **90%**
- Keyword grounding accuracy (answer contains expected fact): **88%**
- Hallucination flags raised: **0**
- Correct refusal rate on out-of-scope questions: **100%**
- Compound-question full-success rate (both expected tools called): **0%**

## Per-question results

| ID | Category | Question | Expected tools | Tools called | Routing correct | Keyword hit | Hallucination |
|---|---|---|---|---|---|---|---|
| 1 | hr_it | How many days of annual leave do I get? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 2 | hr_it | How long do I have to submit an expense claim? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 3 | hr_it | What happens if I lose my company laptop? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 4 | hr_it | How many weeks of maternity leave do I get? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 5 | hr_it | How often do I need to change my corporate password? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 6 | hr_it | How many days a week do I need to be in the office? | ask_hr_it_policy | ask_hr_it_policy | True | True | False |
| 7 | customer_lookup | What is the churn risk for customer CUST100000? | get_customer_churn_risk | get_customer_churn_risk | True | True | False |
| 8 | customer_lookup | Can you look up CUST100005 and tell me their churn probability? | get_customer_churn_risk | get_customer_churn_risk | True | True | False |
| 9 | customer_lookup | Is CUST100004 at risk of churning? | get_customer_churn_risk | get_customer_churn_risk | True | True | False |
| 10 | invalid_customer | What's the churn risk for customer CUST999999? | get_customer_churn_risk | get_customer_churn_risk | True | True | False |
| 11 | country_kpi | Give me the customer KPIs for Singapore. | get_country_kpis | get_country_kpis | True | True | False |
| 12 | country_kpi | How many high-risk customers do we have in Malaysia? | get_country_kpis | get_country_kpis | True | True | False |
| 13 | country_kpi | What's our total revenue from Indonesia? | get_country_kpis | get_country_kpis | True | True | False |
| 14 | churn_drivers | Why are customers churning? | get_churn_risk_drivers | get_churn_risk_drivers | True | True | False |
| 15 | churn_drivers | How accurate is the churn model? | get_churn_risk_drivers | get_churn_risk_drivers | True | True | False |
| 16 | out_of_scope | What's the weather like in Singapore today? | (none) | (none) | True | None | False |
| 17 | out_of_scope | What's Meridian Holdings' stock buyback policy for shareholders? | (none) | (none) | True | None | False |
| 18 | compound | How many days of annual leave do I get, and what's the churn risk for CUST100000? | ask_hr_it_policy, get_customer_churn_risk | get_customer_churn_risk | False | False | False |
| 19 | compound | What's our remote work policy, and how many customers do we have in Malaysia? | ask_hr_it_policy, get_country_kpis | get_country_kpis | False | False | False |
| 20 | ambiguous | Tell me about our onboarding numbers and process. | ask_hr_it_policy | ask_hr_it_policy | True | None | False |
