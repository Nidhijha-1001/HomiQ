import unittest

import numpy as np
import pandas as pd

from homiq.modeling import build_pipeline, evaluate_regression


class ModelingTests(unittest.TestCase):
    def test_pipeline_imputes_missing_values_and_ignores_unknown_categories(self):
        X_train = pd.DataFrame(
            {
                "area": [900.0, 1100.0, np.nan, 1500.0, 1700.0],
                "location": ["Whitefield", "Whitefield", "Hebbal", None, "Hebbal"],
            }
        )
        y_train = pd.Series([120_000, 145_000, 135_000, 180_000, 200_000])
        model = build_pipeline(["area"], ["location"])
        model.fit(X_train, y_train)

        X_new = pd.DataFrame({"area": [np.nan], "location": ["unseen"]})
        prediction = model.predict(X_new)

        self.assertEqual(len(prediction), 1)
        self.assertTrue(np.isfinite(prediction[0]))

    def test_evaluation_reports_all_expected_regression_metrics(self):
        metrics = evaluate_regression(
            pd.Series([100.0, 200.0]), np.array([110.0, 180.0])
        )

        self.assertEqual(set(metrics), {"mse", "rmse", "mae", "r2"})
        self.assertAlmostEqual(metrics["mae"], 15.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250.0))


if __name__ == "__main__":
    unittest.main()
