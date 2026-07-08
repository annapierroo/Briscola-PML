import unittest

from experiments import play_synthetic_game
from inference import (
    fit_variational_posterior,
    log_sequential_likelihood_torch,
    prepare_sequential_games,
    sequential_log_likelihood,
)
from opponents import GREEDY_POINTS_THETA, RANDOM_THETA, RandomOpponent, ThetaSoftmaxOpponent

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class VariationalInferenceTest(unittest.TestCase):
    def test_fit_variational_posterior_returns_expected_shapes(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(GREEDY_POINTS_THETA, seed=4),
            observer_model=RandomOpponent(seed=5),
            seed=6,
            observer_player=0,
            observed_player=1,
        )

        posterior = fit_variational_posterior(
            result.observations[:2],
            num_steps=3,
            learning_rate=0.01,
            num_elbo_samples=1,
            seed=7,
        )

        self.assertEqual(len(posterior.mean), len(RANDOM_THETA))
        self.assertEqual(len(posterior.std), len(RANDOM_THETA))
        self.assertEqual(len(posterior.elbo_history), 3)
        self.assertEqual(posterior.best_elbo, max(posterior.elbo_history))
        self.assertGreaterEqual(posterior.best_step, 1)
        self.assertLessEqual(posterior.best_step, 3)
        self.assertEqual(posterior.training_likelihood, "sequential")

    def test_sequential_torch_likelihood_matches_hand_filter(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(GREEDY_POINTS_THETA, seed=8),
            observer_model=RandomOpponent(seed=9),
            seed=10,
            observer_player=0,
            observed_player=1,
        )

        observations = result.observations[:3]
        prepared = prepare_sequential_games(observations)
        theta = torch.tensor(GREEDY_POINTS_THETA, dtype=torch.float64)
        torch_log_likelihood = float(log_sequential_likelihood_torch(prepared, theta))
        filter_log_likelihood = sequential_log_likelihood(
            observations,
            GREEDY_POINTS_THETA,
        )

        self.assertAlmostEqual(torch_log_likelihood, filter_log_likelihood)

    def test_fit_variational_posterior_with_sequential_likelihood(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(GREEDY_POINTS_THETA, seed=11),
            observer_model=RandomOpponent(seed=12),
            seed=13,
            observer_player=0,
            observed_player=1,
        )

        posterior = fit_variational_posterior(
            result.observations[:2],
            num_steps=2,
            learning_rate=0.01,
            num_elbo_samples=1,
            seed=14,
        )

        self.assertEqual(posterior.training_likelihood, "sequential")
        self.assertEqual(len(posterior.elbo_history), 2)


if __name__ == "__main__":
    unittest.main()
