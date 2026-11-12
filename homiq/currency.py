"""INR-to-USD display conversion with live-rate and configured fallback support."""

import json
import os
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_FALLBACK_USD_PER_INR = 0.012
MIN_VALID_USD_PER_INR = 0.001
MAX_VALID_USD_PER_INR = 0.1
EXCHANGE_RATE_URL = "https://api.frankfurter.dev/v1/latest?from=INR&to=USD"


@dataclass(frozen=True)
class ExchangeRate:
    usd_per_inr: float
    rate_date: str
    source: str
    error: str = ""


def _configured_fallback() -> float:
    raw_rate = os.environ.get(
        "HOMIQ_USD_PER_INR", str(DEFAULT_FALLBACK_USD_PER_INR)
    )
    try:
        rate = float(raw_rate)
    except ValueError as exc:
        raise ValueError("HOMIQ_USD_PER_INR must be a positive numeric rate.") from exc
    if not MIN_VALID_USD_PER_INR <= rate <= MAX_VALID_USD_PER_INR:
        raise ValueError(
            "HOMIQ_USD_PER_INR is outside the configured sanity bounds "
            f"({MIN_VALID_USD_PER_INR}–{MAX_VALID_USD_PER_INR})."
        )
    return rate


def get_exchange_rate() -> ExchangeRate:
    """Fetch the latest daily INR/USD rate, reporting configured fallback use."""
    fallback = _configured_fallback()
    try:
        with urlopen(EXCHANGE_RATE_URL, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rate = float(payload["rates"]["USD"])
        rate_date = str(payload["date"])
        if not MIN_VALID_USD_PER_INR <= rate <= MAX_VALID_USD_PER_INR:
            raise ValueError("The live INR/USD rate failed sanity validation.")
        return ExchangeRate(rate, rate_date, "Frankfurter")
    except (URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return ExchangeRate(
            fallback,
            "configured fallback",
            "HOMIQ_USD_PER_INR",
            str(exc),
        )


def inr_to_usd(value_inr: float, usd_per_inr: float) -> float:
    """Convert an INR amount to USD for display; model values remain in INR."""
    return float(value_inr) * usd_per_inr


def format_usd(value_usd: float) -> str:
    return f"${value_usd:,.0f}"
