import unittest

from scripts.run_comparison import summarize_rows


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


if __name__ == "__main__":
    unittest.main()
