import math
import unittest

from experiments import play_synthetic_game
from inference import LikelihoodMode, marginal_card_probability, marginal_log_likelihood
from opponents import GREEDY_POINTS_THETA, RANDOM_THETA, RandomOpponent, ThetaSoftmaxOpponent


class InferenceTest(unittest.TestCase):
    def test_marginal_likelihood_runs_on_collected_observations(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(GREEDY_POINTS_THETA, seed=1),
            observer_model=RandomOpponent(seed=2),
            seed=3,
            observer_player=0,
            observed_player=1,
        )
        observation = result.observations[0]

        conditional = marginal_card_probability(
            observation.chosen_card,
            observation.public_state,
            observation.observer_hand,
            RANDOM_THETA,
            observed_player=observation.player,
            mode=LikelihoodMode.CONDITIONAL,
        )
        absolute = marginal_card_probability(
            observation.chosen_card,
            observation.public_state,
            observation.observer_hand,
            RANDOM_THETA,
            observed_player=observation.player,
            mode=LikelihoodMode.ABSOLUTE,
        )
        log_likelihood = marginal_log_likelihood(
            result.observations[:3],
            GREEDY_POINTS_THETA,
            mode=LikelihoodMode.CONDITIONAL,
        )

        self.assertGreater(conditional, 0.0)
        self.assertLessEqual(absolute, conditional)
        self.assertTrue(math.isfinite(log_likelihood))


if __name__ == "__main__":
    unittest.main()
