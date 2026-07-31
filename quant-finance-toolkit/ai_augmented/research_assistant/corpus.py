"""Synthetic financial news/research corpus for the RAG research assistant.

All items are fictional analyst-note-style commentary written for this
project — not excerpts of real reporting, and not claims about any real
company's actual results. Ticker names are used only to give the retrieval
demo something realistic to search over.
"""
from dataclasses import dataclass


@dataclass
class NewsItem:
    id: str
    ticker: str  # ticker this item is about, or "MACRO" for market-wide commentary
    date: str
    headline: str
    text: str


CORPUS: list[NewsItem] = [
    NewsItem(
        id="aapl-services-1",
        ticker="AAPL",
        date="2026-06-02",
        headline="Services segment continues to outpace hardware growth",
        text=(
            "Subscription and services revenue grew faster than device sales again this "
            "quarter, continuing a multi-year shift in revenue mix. Gross margin on "
            "services is structurally higher than on hardware, so this mix shift has been "
            "a steady tailwind for blended margins even as unit growth in the core device "
            "line has flattened in mature markets."
        ),
    ),
    NewsItem(
        id="aapl-supply-chain-1",
        ticker="AAPL",
        date="2026-05-14",
        headline="Component cost pressure flagged by supply-chain analysts",
        text=(
            "Several component suppliers have signaled tighter pricing on advanced memory "
            "and display parts heading into the next product cycle. If sustained, this "
            "would compress hardware gross margin unless offset by pricing action or further "
            "mix shift toward services, which several analysts view as the more likely "
            "management response given historical pricing discipline."
        ),
    ),
    NewsItem(
        id="msft-cloud-1",
        ticker="MSFT",
        date="2026-06-10",
        headline="Cloud infrastructure bookings remain a key growth driver",
        text=(
            "Enterprise cloud infrastructure and AI-workload bookings grew at a healthy "
            "clip again, with management citing continued capacity expansion to meet "
            "demand for large-model training and inference workloads. Capital expenditure "
            "guidance was raised accordingly, which some analysts flagged as a near-term "
            "free-cash-flow headwind even as it supports the longer-term growth case."
        ),
    ),
    NewsItem(
        id="msft-regulatory-1",
        ticker="MSFT",
        date="2026-04-22",
        headline="Regulatory scrutiny of cloud bundling practices continues",
        text=(
            "Regulators in multiple jurisdictions continue to examine bundling practices "
            "around productivity software and cloud services. No formal remedies have been "
            "announced, and management has characterized the inquiries as part of ongoing "
            "engagement rather than an active enforcement action, but the overhang has been "
            "cited by some analysts as a source of multiple compression risk."
        ),
    ),
    NewsItem(
        id="googl-ads-1",
        ticker="GOOGL",
        date="2026-06-05",
        headline="Advertising revenue growth reaccelerates on retail spend",
        text=(
            "Search and shopping ad revenue growth reaccelerated this quarter, helped by "
            "stronger retail advertiser spend heading into the seasonally important second "
            "half. Management noted broad-based strength across verticals rather than "
            "concentration in any single advertiser category."
        ),
    ),
    NewsItem(
        id="googl-ai-competition-1",
        ticker="GOOGL",
        date="2026-05-28",
        headline="AI-native search competitors named as a watch item",
        text=(
            "Several sell-side notes flagged newer AI-native answer engines as a long-term "
            "competitive watch item for core search, while acknowledging that measurable "
            "query or revenue impact has not shown up in the numbers to date. The company's "
            "own AI-generated answer features were described as a defensive integration "
            "rather than a proven new growth driver so far."
        ),
    ),
    NewsItem(
        id="amzn-logistics-1",
        ticker="AMZN",
        date="2026-06-08",
        headline="Fulfillment network redesign continues to lift retail margins",
        text=(
            "Regionalized fulfillment center placement continued to reduce average delivery "
            "distance and outbound shipping cost per unit, a multi-year initiative that has "
            "been a steady contributor to retail segment margin improvement. Management "
            "reiterated that further efficiency gains are expected but at a slower pace as "
            "the easiest network redesign wins have already been captured."
        ),
    ),
    NewsItem(
        id="amzn-consumer-spend-1",
        ticker="AMZN",
        date="2026-04-30",
        headline="Analysts watching discretionary spend trends closely",
        text=(
            "With consumer discretionary spending showing signs of moderation in recent "
            "macro data, analysts are watching for any read-through to online retail unit "
            "growth. Management characterized demand as resilient so far but acknowledged "
            "a cautious posture on discretionary-category inventory buying for the back half."
        ),
    ),
    NewsItem(
        id="macro-rates-1",
        ticker="MACRO",
        date="2026-06-15",
        headline="Rate-path uncertainty remains the dominant macro driver for equities",
        text=(
            "With inflation prints coming in mixed, market pricing for the pace of further "
            "policy rate cuts has been volatile, and that uncertainty has been a larger "
            "driver of single-day equity moves than company-specific news for much of the "
            "past quarter. Rate-sensitive growth names have shown the largest swings on "
            "days with major macro data releases."
        ),
    ),
    NewsItem(
        id="macro-volatility-regime-1",
        ticker="MACRO",
        date="2026-05-10",
        headline="Realized volatility spiked sharply in the spring risk-off episode",
        text=(
            "Equity market realized volatility rose sharply during a March-May risk-off "
            "episode driven by trade-policy headlines, with cross-asset correlations "
            "increasing as well — a pattern consistent with prior broad-based de-risking "
            "events rather than an isolated single-sector selloff. Volatility has partially "
            "normalized since, though remains above the levels seen earlier in the year."
        ),
    ),
]
