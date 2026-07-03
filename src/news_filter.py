"""
Per-ticker news filter — STUB for the stocks bot.

The crypto bot used CryptoPanic here (crypto-only source — useless for AAPL).
Market-wide risk events are still covered by the Groq macro news agent
(src/news_agent.py) and the ForexFactory economic-calendar warning.

Kept as a pass-through so the main.py pipeline stays unchanged; wire a real
per-ticker source later (e.g. Finnhub/AlphaVantage earnings & headlines) —
biggest win would be an EARNINGS-DATE blackout (skip signals the day of/before
the ticker's earnings report — gap risk no SMC setup can price in).
"""


def check_news_sentiment(symbol: str) -> dict:
    """Always safe — per-ticker stock news source not wired yet."""
    return {"safe": True, "reason": ""}
