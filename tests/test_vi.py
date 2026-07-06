import unittest

from experiments import play_synthetic_game
from inference import fit_variational_posterior
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


if __name__ == "__main__":
    unittest.main()
