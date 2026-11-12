import unittest

import pandas as pd

from homiq.analytics import (
    actual_vs_predicted_chart,
    correlation_heatmap,
    feature_importance_chart,
    filter_market_data,
    price_distribution_chart,
)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.market = pd.DataFrame(
            {
                "location": ["Whitefield", "Whitefield", "Hebbal"],
                "area_type": ["Super built-up Area", "Plot Area", "Built-up Area"],
                "bhk": [2, 4, 3],
                "total_sqft": [1000, 1800, 1400],
                "bath": [2, 3, 2],
                "balcony": [1, 2, 1],
                "price_inr": [5_000_000, 12_000_000, 8_000_000],
            }
        )

    def test_market_filters_intersect_location_type_and_bhk(self):
        filtered = filter_market_data(
            self.market,
            location="Whitefield",
            area_type="Plot Area",
            bhk_range=(3, 5),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["price_inr"], 12_000_000)

    def test_price_charts_convert_inr_values_to_usd(self):
        test = self.market.assign(
            ActualPrice=self.market["price_inr"],
            PredictedPrice=[5_500_000, 11_000_000, 8_500_000],
        )
        prediction_chart = actual_vs_predicted_chart(test, usd_per_inr=0.01)
        distribution_chart = price_distribution_chart(
            self.market["price_inr"],
            usd_per_inr=0.01,
            predicted_price_inr=6_000_000,
        )

        self.assertEqual(list(prediction_chart.data[0].x), [50_000, 120_000, 80_000])
        self.assertEqual(list(prediction_chart.data[0].y), [55_000, 110_000, 85_000])
        self.assertEqual(
            list(distribution_chart.data[0].x), [50_000, 120_000, 80_000]
        )
        self.assertEqual(distribution_chart.layout.shapes[0].x0, 60_000)

    def test_correlation_and_importance_charts_are_data_backed(self):
        heatmap = correlation_heatmap(
            self.market, ["total_sqft", "bhk", "price_inr"]
        )
        importance = feature_importance_chart(
            pd.DataFrame(
                [
                    {
                        "feature": "total_sqft",
                        "mae_increase": 1_000_000,
                        "mae_increase_std": 100_000,
                    }
                ]
            ),
            usd_per_inr=0.01,
        )

        self.assertEqual(len(heatmap.data), 1)
        self.assertIn("Listed price (USD)", list(heatmap.data[0].x))
        self.assertEqual(list(importance.data[0].x), [10_000])
        self.assertEqual(list(importance.data[0].error_x.array), [1_000])


if __name__ == "__main__":
    unittest.main()
