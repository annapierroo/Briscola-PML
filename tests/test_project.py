import math
import unittest

from experiments import (
    collect_observations,
    heldout_predictive_evaluation,
    play_synthetic_game,
    train_test_split,
)
from game import BriscolaGame, Card, PlayedCard, Rank, Suit, full_deck, trick_winner
from inference import (
    compatible_hands_containing,
    compatible_unknown_cards,
    fit_variational_posterior,
    known_opponent_cards,
    log_sequential_likelihood_torch,
    marginal_card_probability,
    prepare_sequential_games,
    sequential_log_likelihood,
)
from opponents import (
    AGGRESSIVE_THETA,
    CORE_FEATURE_NAMES,
    GREEDY_POINTS_THETA,
    RANDOM_THETA,
    STYLE_FEATURE_NAMES,
    STYLE_GREEDY_POINTS_THETA,
    RandomOpponent,
    ThetaSoftmaxOpponent,
    card_features,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def deck_with_prefix(prefix: list[Card]) -> tuple[Card, ...]:
    rest = [card for card in full_deck() if card not in prefix]
    return tuple(prefix + rest)


class ProjectTest(unittest.TestCase):
    def test_briscola_rules_and_full_game_score(self) -> None:
        self.assertEqual(
            trick_winner(
                PlayedCard(0, Card(Rank.THREE, Suit.CUPS)),
                PlayedCard(1, Card(Rank.ACE, Suit.CUPS)),
                Suit.COINS,
            ),
            1,
        )
        self.assertEqual(
            trick_winner(
                PlayedCard(0, Card(Rank.ACE, Suit.CUPS)),
                PlayedCard(1, Card(Rank.TWO, Suit.COINS)),
                Suit.COINS,
            ),
            1,
        )

        game = BriscolaGame.new(seed=7, first_player=0)
        while not game.finished:
            player = game.public_state().current_player
            assert player is not None
            game.play_card(player, game.legal_moves()[0])

        self.assertEqual(len(game.trick_history), 20)
        self.assertEqual(sum(game.scores), 120)

    def test_synthetic_games_give_observations(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(
                AGGRESSIVE_THETA,
                seed=2,
                feature_names=CORE_FEATURE_NAMES,
            ),
            observer_model=RandomOpponent(seed=1),
            seed=11,
            observer_player=0,
            observed_player=1,
        )
        observations = collect_observations(
            observed_model=RandomOpponent(seed=3),
            observer_model=RandomOpponent(seed=4),
            num_games=2,
            seed=12,
        )

        self.assertEqual(sum(result.final_scores), 120)
        self.assertEqual(len(result.observations), 20)
        self.assertTrue(all(move.player == 1 for move in result.observations))
        self.assertTrue(all(move.chosen_card in move.legal_cards for move in result.observations))
        self.assertEqual(len(observations), 40)

    def test_style_features_feed_the_softmax_opponent(self) -> None:
        game = BriscolaGame.from_deck(
            deck_with_prefix(
                [
                    Card(Rank.ACE, Suit.CUPS),
                    Card(Rank.TWO, Suit.SWORDS),
                    Card(Rank.FOUR, Suit.CLUBS),
                    Card(Rank.THREE, Suit.COINS),
                    Card(Rank.FIVE, Suit.CUPS),
                    Card(Rank.SIX, Suit.CLUBS),
                    Card(Rank.SEVEN, Suit.COINS),
                ]
            ),
            first_player=0,
        )
        view = game.player_view(0)

        features = card_features(
            view.hand[0],
            view.hand,
            view.public_state,
            player=0,
            feature_names=STYLE_FEATURE_NAMES,
        )
        probabilities = ThetaSoftmaxOpponent(
            STYLE_GREEDY_POINTS_THETA,
            feature_names=STYLE_FEATURE_NAMES,
        ).probabilities(view)

        self.assertEqual(len(features), len(STYLE_GREEDY_POINTS_THETA))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(
            probabilities[Card(Rank.ACE, Suit.CUPS)],
            probabilities[Card(Rank.TWO, Suit.SWORDS)],
        )

    def test_hidden_hand_belief_supports_marginal_prediction(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(
                GREEDY_POINTS_THETA,
                seed=1,
                feature_names=CORE_FEATURE_NAMES,
            ),
            observer_model=RandomOpponent(seed=2),
            seed=3,
            observer_player=0,
            observed_player=1,
        )
        move = result.observations[0]

        unknown_cards = compatible_unknown_cards(
            move.public_state,
            move.observer_hand,
            opponent_player=move.player,
        )
        required_cards = known_opponent_cards(
            move.public_state,
            move.observer_hand,
            opponent_player=move.player,
        )
        hands = compatible_hands_containing(
            unknown_cards,
            move.public_state.hand_sizes[move.player],
            move.chosen_card,
            required_cards=required_cards,
        )
        probability = marginal_card_probability(
            move.chosen_card,
            move.public_state,
            move.observer_hand,
            RANDOM_THETA,
            observed_player=move.player,
            feature_names=CORE_FEATURE_NAMES,
        )

        self.assertIn(move.chosen_card, unknown_cards)
        self.assertTrue(hands)
        self.assertGreater(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_sequential_likelihood_and_vi_run(self) -> None:
        result = play_synthetic_game(
            observed_model=ThetaSoftmaxOpponent(
                GREEDY_POINTS_THETA,
                seed=8,
                feature_names=CORE_FEATURE_NAMES,
            ),
            observer_model=RandomOpponent(seed=9),
            seed=10,
            observer_player=0,
            observed_player=1,
        )
        observations = result.observations[:3]
        prepared = prepare_sequential_games(
            observations,
            feature_names=CORE_FEATURE_NAMES,
        )
        theta = torch.tensor(GREEDY_POINTS_THETA, dtype=torch.float64)

        torch_value = float(log_sequential_likelihood_torch(prepared, theta))
        direct_value = sequential_log_likelihood(
            observations,
            GREEDY_POINTS_THETA,
            feature_names=CORE_FEATURE_NAMES,
        )
        posterior = fit_variational_posterior(
            observations[:2],
            feature_names=CORE_FEATURE_NAMES,
            num_steps=2,
            learning_rate=0.01,
            num_elbo_samples=1,
            seed=14,
        )

        self.assertAlmostEqual(torch_value, direct_value)
        self.assertEqual(len(posterior.mean), len(GREEDY_POINTS_THETA))
        self.assertEqual(posterior.training_likelihood, "sequential")
        self.assertEqual(len(posterior.elbo_history), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_validation_split_and_heldout_score(self) -> None:
        observations = collect_observations(
            observed_model=ThetaSoftmaxOpponent(
                GREEDY_POINTS_THETA,
                seed=1,
                feature_names=CORE_FEATURE_NAMES,
            ),
            observer_model=RandomOpponent(seed=2),
            num_games=3,
            seed=3,
        )
        train, test = train_test_split(observations, train_fraction=0.75)
        prediction = heldout_predictive_evaluation(
            test,
            posterior_mean=GREEDY_POINTS_THETA,
            posterior_std=tuple(0.1 for _ in GREEDY_POINTS_THETA),
            posterior_samples=5,
            seed=4,
            baseline_theta=RANDOM_THETA,
            feature_names=CORE_FEATURE_NAMES,
        )

        train_games = {move.game_id for move in train}
        test_games = {move.game_id for move in test}

        self.assertTrue(train_games.isdisjoint(test_games))
        self.assertEqual(len(train) + len(test), len(observations))
        self.assertTrue(math.isfinite(prediction.posterior_log_likelihood))
        self.assertEqual(prediction.posterior_samples, 5)
        self.assertEqual(prediction.likelihood, "posterior_predictive_sequential")


if __name__ == "__main__":
    unittest.main()
