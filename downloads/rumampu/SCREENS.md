# RuMampu — Screen inventory (Iteration 1 + preview)

Format: screen → what the user is deciding here → the single primary action.

| # | Screen | The user is deciding… | Primary action |
|---|---|---|---|
| 1 | Onboarding (3 cards) | whether this app is for them | **Start** (or Skip — both land on Home) |
| 2 | Home | what to do next: add records or test a house | **Test a house** / **Re-test the house** |
| — | Money (hub) | which money fact to update | tap a section row |
| 3 | Money — Income | whether to record last week's earnings | **Add income** |
| 4 | Money — Work costs | what it really costs them to earn | edit a cost (net income updates live) |
| 5 | Money — Commitments | what already leaves the account monthly | tap a preset chip |
| 6 | Money — Income pattern | how much their income actually moves | read; edit nothing (figures only) |
| 7 | Money — Coverage check | whether the recorded months are the *right* months | answer **Yes / No / Not sure** |
| 18 | Money — Your record *(journey stage 10, Monitor)* | whether the record is growing and which tests to keep | add entries elsewhere; review kept tests |
| 8 | Test — The house | which house, and on what financing | **Total monthly cost →** (escape hatch: "I already know my monthly payment") |
| 9 | Test — Total monthly cost | whether they accept what the house *really* costs monthly | **Run the test →** |
| 10 | Test — Pre-housing check *(fires only when the weakest month is short before housing)* | whether to fix income/costs before testing any price | **Go to Money** |
| 11 | Test — Result | what the count of short months means for them | tap **How this was worked out** |
| 12 | Test — Carrying range | how their entered price sits against what the record carried | read; back to Result |
| 13 | Test — Compare payments | which of three payments they want to sit with | edit a payment figure |
| 14 | Test — If income drops | how fragile the result is to a weaker version of the same record | tap **−10% / −20% / Custom** |
| — | Prepare (hub) | which preparation front to work on | tap a section row |
| 15 | Prepare — Upfront cash | whether the one-time cash gap is closable | edit a fee line |
| 16 | Prepare — Cash buffer | how much starting cash their own record implies | read; back |
| 17 | Prepare — Documents & financing | what to assemble, and what RuMampu can't tell them | tick through the checklist |
| P1 | Preview — Switch mode | (team demo) how estimates become actuals | **I've bought the home** |
| P2 | Preview — This month | (team demo) cash left or short this month, actuals only | read |
| P3 | Preview — Earlier test vs what happened | (team demo) how the earlier count compares with lived months | read |

Notes

- **Reaching screen 10:** it renders only when it fires. To demo: Money → Commitments → raise rent to RM 9,000 → open Test. Restore rent and the Test tab returns to the Result.
- **Preview entry** is the badged row at the foot of Prepare — deliberately not in the tab bar.
- **Journey-map additions (23 Aug):** past-month bulk entry on Income (stage 1), largest gap on the shock screen (stage 8), month-by-month strip on Income pattern (stage 4), "Keep this test" on Result feeding the Your-record screen (stage 10), and the keep-recording line on the after-buying comparison (stage 15).
- The **language switcher** (EN · BM · 中文) is in every header, onboarding included.
- `audit.mjs` in this folder is the self-audit: `node audit.mjs` (needs Node; DOM checks additionally need the `playwright` package and Chromium — set `NODE_PATH` to a global install if required).
