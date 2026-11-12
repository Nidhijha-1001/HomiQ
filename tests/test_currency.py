import os
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import URLError

from homiq.currency import (
    ExchangeRate,
    _configured_fallback,
    format_usd,
    get_exchange_rate,
    inr_to_usd,
)


class CurrencyTests(unittest.TestCase):
    def test_inr_conversion_and_usd_formatting(self):
        self.assertEqual(inr_to_usd(1_000_000, 0.01), 10_000)
        self.assertEqual(format_usd(10_000), "$10,000")

    def test_live_rate_is_used_when_service_returns_valid_data(self):
        response = BytesIO(b'{"date":"2026-11-11","rates":{"USD":0.0105}}')
        with patch("homiq.currency.urlopen", return_value=response):
            quote = get_exchange_rate()

        self.assertEqual(
            quote,
            ExchangeRate(0.0105, "2026-11-11", "Frankfurter"),
        )

    def test_configured_fallback_is_explicit_when_live_service_fails(self):
        with patch.dict(os.environ, {"HOMIQ_USD_PER_INR": "0.011"}):
            with patch("homiq.currency.urlopen", side_effect=URLError("offline")):
                quote = get_exchange_rate()

        self.assertEqual(quote.usd_per_inr, 0.011)
        self.assertEqual(quote.source, "HOMIQ_USD_PER_INR")
        self.assertTrue(quote.error)

    def test_invalid_fallback_rates_raise_a_clear_error(self):
        with patch.dict(os.environ, {"HOMIQ_USD_PER_INR": "0"}):
            with self.assertRaises(ValueError):
                _configured_fallback()


if __name__ == "__main__":
    unittest.main()
