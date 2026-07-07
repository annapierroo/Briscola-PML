import unittest
from types import SimpleNamespace

from scripts.run_comparison import _run_validation_case, summarize_rows


class ComparisonScriptTest(unittest.TestCase):
    def test_summarize_rows_groups_by_feature_set_and_profile(self) -> None:
        rows = [
            {
                "feature_set": "core",
                "profile": "conservative",
                "feature_count": 4,
                "theta_l2_error": 1.0,
                "heldout_loglik_delta": 10.0,
                "heldout_mean_logp_delta": 0.1,
                "calibration_ece": 0.01,
                "mean_theta_heldout_loglik_delta": 9.0,
                "posterior_predictive_heldout_loglik_delta": 10.0,
                "mean_theta_heldout_mean_logp_delta": 0.09,
                "posterior_predictive_heldout_mean_logp_delta": 0.1,
                "mean_theta_calibration_ece": 0.02,
                "posterior_predictive_calibration_ece": 0.01,
                "importance_ess": 100.0,
                "final_elbo": -20.0,
            },
            {
                "feature_set": "core",
                "profile": "conservative",
                "feature_count": 4,
                "theta_l2_error": 3.0,
                "heldout_loglik_delta": 20.0,
                "heldout_mean_logp_delta": 0.2,
                "calibration_ece": 0.03,
                "mean_theta_heldout_loglik_delta": 18.0,
                "posterior_predictive_heldout_loglik_delta": 20.0,
                "mean_theta_heldout_mean_logp_delta": 0.18,
                "posterior_predictive_heldout_mean_logp_delta": 0.2,
                "mean_theta_calibration_ece": 0.04,
                "posterior_predictive_calibration_ece": 0.03,
                "importance_ess": 200.0,
                "final_elbo": -10.0,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["feature_set"], "core")
        self.assertEqual(summary[0]["profile"], "conservative")
        self.assertEqual(summary[0]["runs"], 2)
        self.assertEqual(summary[0]["theta_l2_error_mean"], 2.0)
        self.assertEqual(summary[0]["heldout_loglik_delta_mean"], 15.0)
        self.assertEqual(summary[0]["mean_theta_heldout_loglik_delta_mean"], 13.5)
        self.assertEqual(summary[0]["posterior_predictive_heldout_loglik_delta_mean"], 15.0)
        self.assertGreater(summary[0]["theta_l2_error_std"], 0.0)

    def test_summarize_rows_can_skip_importance_reference(self) -> None:
        rows = [
            {
                "feature_set": "compact",
                "profile": "aggressive",
                "feature_count": 5,
                "theta_l2_error": 1.5,
                "heldout_loglik_delta": 12.0,
                "heldout_mean_logp_delta": 0.12,
                "calibration_ece": 0.02,
                "mean_theta_heldout_loglik_delta": 11.0,
                "posterior_predictive_heldout_loglik_delta": 12.0,
                "mean_theta_heldout_mean_logp_delta": 0.11,
                "posterior_predictive_heldout_mean_logp_delta": 0.12,
                "mean_theta_calibration_ece": 0.03,
                "posterior_predictive_calibration_ece": 0.02,
                "final_elbo": -30.0,
            }
        ]

        summary = summarize_rows(rows, include_importance_reference=False)

        self.assertEqual(summary[0]["theta_l2_error_mean"], 1.5)
        self.assertNotIn("importance_ess_mean", summary[0])

    def test_validation_case_records_theta_components(self) -> None:
        spec = SimpleNamespace(
            feature_set="core",
            profile="greedy_points",
            seed=0,
            data_source="matched",
            num_games=2,
            train_fraction=0.75,
            max_train_observations=10,
            max_test_observations=5,
            split_unit="game",
            theta_scale=1.0,
            train_mode="conditional",
            eval_mode="conditional",
            calibration_mode="absolute",
            vi_steps=1,
            learning_rate=0.03,
            prior_std=1.0,
            elbo_samples=1,
            posterior_samples=0,
            importance_samples=8,
            reference_observations=2,
            skip_importance_reference=True,
            calibration_bins=2,
            skip_calibration=True,
            calibration_observations=0,
            progress_interval=1,
        )

        try:
            row = _run_validation_case(spec, show_vi_progress=False)
        except ModuleNotFoundError as exc:
            if "PyTorch" not in str(exc):
                raise
            self.skipTest("PyTorch is not installed")

        self.assertEqual(len(row["theta"]), row["feature_count"])
        self.assertIn("theta_true", row)
        self.assertIn("theta_posterior_mean", row)
        self.assertEqual(row["theta"][0]["feature"], "is_trump")


if __name__ == "__main__":
    unittest.main()
