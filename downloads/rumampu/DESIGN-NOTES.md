# RuMampu — Design Notes (v2 rebuild)

**Direction in one line:** a quiet ledger on warm paper, with exactly one loud element — the **waterline**: your recorded months drawn as ink bars, and the home's total monthly cost drawn as a line that visibly cuts across them.

## Type

No CDN is allowed, so fonts ship in-file or come from the system.

- **Display: Space Grotesk Bold**, subsetted to Latin + digits and embedded as a WOFF2 data-URI (~15 KB). Used only for the one big answer per screen, large ringgit figures, and tab labels. Tabular lining figures on — columns of money must align. Deliberately not a high-contrast serif; its slightly technical character against the cream reads "instrument", not "lifestyle brand".
- **Body: system stack** (`system-ui, Roboto, "Segoe UI", sans-serif`). Free, instant on a cheap Android; Malay shares Latin, and 中文 falls back to the system CJK face — embedding a CJK subset in a single file is not viable, and Roboto/Noto is what Farid's phone renders best anyway.

Scale (px / line-height): **40/44** display (max one per screen) · **26/32** headline · **19/26** title · **16/24** body · **13/18** secondary · **11/14** provenance caps, tracking 0.08em.

## Space

4px base. Steps: **4 · 8 · 12 · 16 · 24 · 32 · 48.** Screen gutter 20. Card padding 16. Tap targets ≥48. One column, always — this app is used one-handed in a parked car.

## Colour — how the six are spent

| Token | Spent on |
|---|---|
| `--ink` #3C5152 | All text, chrome, and **every neutral data bar — including covered months** |
| `--paper` #FFFFFF | The page (changed from the brief's #E1DFC6 cream at client request, 23 Aug 2026). Cards are a cool ink-tinted gray, #EFF3F2, with a hairline edge |
| `--brand` #4A9195 | One primary button per screen, active tab, links, focus ring |
| `--confirm` #32B14A | The save toast and entry-accepted tick. Nothing else, ever |
| `--caution` #FEC844 | Unknown values, coverage gaps, the "test hasn't seen enough" state |
| `--short` #F1592A | The shortfall itself — one component per screen, nothing else |

**Outcome is encoded by geometry, not hue.** A month is a bar; the cost is a line; a short month is a bar that fails to reach the line — and only the *gap segment* fills `--short`. Covered months stay ink and look deliberately boring. Colour is thereby freed to mean exactly two things: "short" and "RuMampu doesn't know". No traffic light can form because green never touches an outcome and yellow never touches a judgement.

## Signature element — the waterline

One chart grammar reused everywhere (pattern, result, compare, shocks, this-month preview): vertical ink bars = income after work costs per recorded month; a single horizontal rule = total monthly home cost. The rule overshoots the plot into the right margin — the only element in the app allowed to break the grid; that's the boldness budget, spent once. Unknown months render as `--caution` outlined bars. Three sizes: strip (Home), full (Result), triple (Compare).

## IA

Four tabs: **Home** (S2 snapshot) · **Money** (S3 income, S4 work costs, S5 commitments, S6 pattern, S7 coverage) · **Test** (S8 house, S9 total cost, S10 pre-housing gate, S11 result, S12 carrying range, S13 compare, S14 shocks) · **Prepare** (S15 upfront cash, S16 buffer, S17 documents/SJKP). Onboarding = three skippable cards over Home on first launch. The three Iteration-3 preview screens sit behind a clearly badged row at the foot of Prepare — not in the tab bar. Language switcher (EN · BM · 中文) lives in every screen header.

**Provenance treatment:** every figure carries an 11px caps label in ink at 64% opacity with a constant glyph — ● User Data · ○ Official Source · ▸ Derived Calculation · ▒ Stress-Test Assumption. Glyphs stay stable across languages; tapping the label opens a one-line source sheet. Monochrome, so it can sit on every number without shouting.

## The three hardest calls

1. **Refusing the traffic light by moving outcome into geometry.** The obvious design colours covered months green — and instantly becomes a verdict machine. Instead outcome lives in bar-versus-line position, `--confirm` is exiled to the save toast, and the covered state is intentionally unremarkable. Cost: the result screen is less immediately "scannable" than a green/red grid. Accepted, because scannable here means judged.
2. **Provenance on ~every number inside a 60-word screen budget.** Words-only labels blew the budget in all three languages; icons-only failed comprehension. The call: constant glyph + short localized word at 64% opacity, with the full explanation behind a tap. Provenance is metadata, so it earns quiet, not silence.
3. **Carrying range without the upward nudge.** The range renders as a flat ink band beneath the waterline with the user's entered payment as the only emphasized mark. No arrows, no delta figure, and when the entered price sits below the band there is **no comparative sentence at all** — the band is simply what the record carried. Label: "Indicative only. Not a valuation or an offer." The anti-nudge is enforced by omission, not by disclaimer.

## Wireframes (390px)

**Home (house entered)** — one answer above the fold:

```
┌────────────────────────────┐
│ RuMampu             EN ▾   │
│                            │
│  2 of your 6 recorded      │  ← 40px display, ink
│  months would have         │
│  run short.                │
│  ▸ DERIVED CALCULATION     │
│                            │
│  Largest gap RM 340        │  ← 19px
│  ▸ DERIVED CALCULATION     │
│                            │
│  ▂▄▃█▅▆ ──────             │  ← coverage strip; gap
│  Feb – Jul recorded        │    hatched in --caution
│  ● USER DATA               │
│                            │
│ [ Re-test the house      ] │  ← --brand, full width
├────────────────────────────┤
│ Home │ Money │ Test │ Prep │
└────────────────────────────┘
```

**S11 Result** — headline first, waterline, reasoning behind a tap:

```
┌────────────────────────────┐
│ ← Test              EN ▾   │
│                            │
│  2 of your 6 recorded      │
│  months would have         │
│  run short.                │
│  ▸ DERIVED CALCULATION     │
│  Largest gap RM 340        │
│                            │
│      █  █     █  █         │  ← bars: ink
│  ────┼──┼──░──┼──┼──░── ─▶ │  ← waterline: total cost,
│      F  M  A  M  J  J      │    overshoots right edge
│   ░ = gap below line       │    gap fill: --short only
│                            │
│  ( How this was worked     │  ← plain link, not button
│    out )                   │
│                            │
│  Carrying range          → │
│  Compare payments        → │
│  If income drops         → │
└────────────────────────────┘
```

**S12 Carrying range** — band + pin, no nudge:

```
┌────────────────────────────┐
│ ← Result            EN ▾   │
│                            │
│  Your recorded months      │
│  carried payments of       │
│  RM 980 – RM 1,310         │  ← 26px, ink
│  a month.                  │
│  ▸ DERIVED CALCULATION     │
│                            │
│  ├────▓▓▓▓▓▓▓▓────┤        │  ← ink band
│         ▲ RM 1,150         │  ← entered payment,
│           your test        │    the only emphasis
│                            │
│  At your rate and tenure   │
│  that is about             │
│  RM 192k – RM 246k         │
│  ▒ STRESS-TEST ASSUMPTION  │
│                            │
│  Indicative only. Not a    │
│  valuation or an offer.    │
└────────────────────────────┘
```
