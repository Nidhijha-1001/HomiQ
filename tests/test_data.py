import tempfile
import unittest
from pathlib import Path

import pandas as pd

from homiq.data import TARGET, load_indian_housing


class IndianHousingDataTests(unittest.TestCase):
    def test_load_normalizes_area_and_converts_lakh_prices_to_inr(self):
        source = pd.DataFrame(
            [
                {
                    "area_type": "Super built-up Area",
                    "location": " Whitefield ",
                    "size": "2 BHK",
                    "total_sqft": "1050 - 1200",
                    "bath": 2,
                    "balcony": 1,
                    "price": 90.5,
                },
                {
                    "area_type": "Built-up Area",
                    "location": "Hebbal",
                    "size": "1 Bedroom",
                    "total_sqft": "34.46Sq. Meter",
                    "bath": 1,
                    "balcony": None,
                    "price": 1.0,
                },
                {
                    "area_type": "Built-up Area",
                    "location": "Hebbal",
                    "size": "1 RK",
                    "total_sqft": "1000",
                    "bath": 1,
                    "balcony": 0,
                    "price": 20.0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indian_housing.csv"
            source.to_csv(path, index=False)
            frame = load_indian_housing(path)

        self.assertEqual(len(frame), 2)
        whitefield = frame.loc[frame["location"] == "Whitefield"].iloc[0]
        hebbal = frame.loc[frame["location"] == "Hebbal"].iloc[0]
        self.assertEqual(whitefield["total_sqft"], 1125)
        self.assertEqual(whitefield[TARGET], 9_050_000)
        self.assertAlmostEqual(hebbal["total_sqft"], 34.46 * 10.7639104167)
        self.assertEqual(hebbal[TARGET], 100_000)
        self.assertTrue(pd.isna(hebbal["balcony"]))


if __name__ == "__main__":
    unittest.main()
