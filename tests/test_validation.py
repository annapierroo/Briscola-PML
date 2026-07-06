import math
import unittest

import experiments
from experiments import (
    calibration_curve,
    collect_matched_model_observations,
    collect_observations,
    heldout_predictive_evaluation,
    importance_sampling_reference,
    train_test_split,
)
from inference import LikelihoodMode, marginal_card_probability
from opponents import GREEDY_POINTS_THETA, RANDOM_THETA, RandomOpponent, ThetaSoftmaxOpponent


class ValidationTest(unittest.TestCase):
    def test_experiments_package_exports_validation_helpers(self) -> None:
        self.assertIs(experiments.calibration_curve, calibration_curve)
        self.assertIs(experiments.train_test_split, train_test_split)

    def test_validation_utilities_run_on_small_dataset(self) -> None:
        observations = collect_observations(
            observed_model=ThetaSoftmaxOpponent(GREEDY_POINTS_THETA, seed=1),
            observer_model=RandomOpponent(seed=2),
            num_games=1,
            seed=3,
        )
        train, test = train_test_split(observations, train_fraction=0.8)

        predictive = heldout_predictive_evaluation(
            test,
            posterior_mean=GREEDY_POINTS_THETA,
            baseline_theta=RANDOM_THETA,
        )
        calibration = calibration_curve(test, GREEDY_POINTS_THETA, num_bins=5)
        reference = importance_sampling_reference(train[:2], num_samples=8, seed=4)

        self.assertEqual(len(train) + len(test), len(observations))
        self.assertTrue(math.isfinite(predictive.posterior_log_likelihood))
        self.assertEqual(len(calibration.bins), 5)
        self.assertGreater(reference.effective_sample_size, 0.0)

    def test_matched_model_generator_produces_compatible_observations(self) -> None:
        observations = collect_matched_model_observations(
            GREEDY_POINTS_THETA,
            num_games=1,
            seed=7,
        )
        first = observations[0]

        probability = marginal_card_probability(
            first.chosen_card,
            first.public_state,
            first.observer_hand,
            GREEDY_POINTS_THETA,
            observed_player=first.player,
            mode=LikelihoodMode.ABSOLUTE,
        )

        self.assertEqual(len(observations), 20)
        self.assertIn(first.chosen_card, first.opponent_hand)
        self.assertGreater(probability, 0.0)


if __name__ == "__main__":
    unittest.main()
