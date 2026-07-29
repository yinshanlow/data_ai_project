# Evaluation Results

Mode used: **mock**

- Questions evaluated: 20 (18 in-scope, 2 deliberately out-of-scope)
- Retrieval accuracy (expected doc in top-k): **100%**
- Keyword grounding accuracy (answer contains expected fact): **100%**
- Correct refusal rate on out-of-scope questions: **100%**
- Hallucination flags raised: **0**

## Per-question results

| ID | Question | In scope | Retrieval hit | Keyword hit | Grounded | Best distance | Hallucination flag |
|---|---|---|---|---|---|---|---|
| 1 | How many days of annual leave am I entitled to in my first two years? | True | True | True | True | 0.742 | False |
| 2 | How far in advance do I need to submit a leave request for a 5-day holiday? | True | True | True | True | 0.751 | False |
| 3 | How many unused annual leave days can I carry into next year? | True | True | True | True | 0.706 | False |
| 4 | What is the deadline for submitting an expense claim after I incur the cost? | True | True | True | True | 0.782 | False |
| 5 | Can I claim alcohol on my expenses? | True | True | True | True | 0.991 | False |
| 6 | How long does it take to get reimbursed after my expense claim is approved? | True | True | True | True | 0.461 | False |
| 7 | What happens if I lose my company laptop? | True | True | True | True | 0.968 | False |
| 8 | How often is my work laptop refreshed? | True | True | True | True | 0.592 | False |
| 9 | How many days a week do I need to be in the office under the hybrid policy? | True | True | True | True | 0.664 | False |
| 10 | Am I eligible for hybrid work while I'm still on probation? | True | True | True | True | 0.456 | False |
| 11 | How long is the probation period for a new hire? | True | True | True | True | 0.435 | False |
| 12 | What mandatory training do I need to finish in my first week? | True | True | True | True | 0.894 | False |
| 13 | How often do I need to change my corporate account password? | True | True | True | True | 0.463 | False |
| 14 | Who do I report a suspected phishing email to? | True | True | True | True | 0.81 | False |
| 15 | What class of flight can I book for a 10-hour international trip if I'm a manager? | True | True | True | True | 0.824 | False |
| 16 | How many performance review cycles happen per year and when? | True | True | True | True | 0.804 | False |
| 17 | How many weeks of paid maternity leave do I get? | True | True | True | True | 0.491 | False |
| 18 | Can I raise a workplace harassment concern anonymously? | True | True | True | True | 0.972 | False |
| 19 | What is the company's reimbursement policy for leasing a personal car in Jakarta? | False | None | None | False | 1.138 | False |
| 20 | What's Meridian Holdings' stock buyback policy for shareholders? | False | None | None | False | 1.003 | False |
