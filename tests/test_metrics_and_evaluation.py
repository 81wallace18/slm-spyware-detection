import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = None
    pd = None


@unittest.skipIf(np is None or pd is None, "numpy/pandas not installed")
class MetricsAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        from scripts.run_evaluation import generate_comparison_table
        from src.metrics import compute_anomaly_metrics

        self.generate_comparison_table = generate_comparison_table
        self.compute_anomaly_metrics = compute_anomaly_metrics

    def test_anomaly_metrics_reports_real_fpr(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.8, 0.7])

        result = self.compute_anomaly_metrics(y_true, scores, {0.5: 0.75})["fpr_0.5"]

        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fpr_actual"], 0.5)

    def test_comparison_table_leaves_missing_metrics_blank(self):
        df = pd.DataFrame([
            {"model": "RF", "pr_auc": 0.9, "f1_macro": 0.8},
            {"model": "SLM benign-only", "pr_auc": 0.7, "tpr": 0.6},
        ])

        comparison = self.generate_comparison_table(df)
        benign = comparison[comparison["model"] == "SLM benign-only"].iloc[0]

        self.assertEqual(benign["f1_macro"], "")
        self.assertEqual(benign["tpr"], "0.6000")


if __name__ == "__main__":
    unittest.main()
