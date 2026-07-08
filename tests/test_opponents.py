import unittest

from game import BriscolaGame, Card, Rank, Suit, full_deck
from opponents import (
    COMPACT_FEATURE_NAMES,
    COMPACT_GREEDY_POINTS_THETA,
    GREEDY_POINTS_THETA,
    RANDOM_THETA,
    STYLE_FEATURE_NAMES,
    STYLE_GREEDY_POINTS_THETA,
    ThetaSoftmaxOpponent,
    TRUMP_COUNT_FEATURE_NAMES,
    TRUMP_COUNT_GREEDY_POINTS_THETA,
    card_features,
)


def deck_with_prefix(prefix: list[Card]) -> tuple[Card, ...]:
    remaining = [card for card in full_deck() if card not in prefix]
    return tuple(prefix + remaining)


class OpponentModelTest(unittest.TestCase):
    def test_feature_vector_and_softmax_model_are_usable(self) -> None:
        prefix = [
            Card(Rank.ACE, Suit.CUPS),
            Card(Rank.TWO, Suit.SWORDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.THREE, Suit.COINS),
            Card(Rank.FIVE, Suit.CUPS),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.COINS),
        ]
        game = BriscolaGame.from_deck(deck_with_prefix(prefix), first_player=0)
        view = game.player_view(0)

        features = card_features(view.hand[0], view.hand, view.public_state, player=0)
        compact_features = card_features(
            view.hand[0],
            view.hand,
            view.public_state,
            player=0,
            feature_names=COMPACT_FEATURE_NAMES,
        )
        trump_count_features = card_features(
            view.hand[0],
            view.hand,
            view.public_state,
            player=0,
            feature_names=TRUMP_COUNT_FEATURE_NAMES,
        )
        style_features = card_features(
            view.hand[0],
            view.hand,
            view.public_state,
            player=0,
            feature_names=STYLE_FEATURE_NAMES,
        )
        random_probabilities = ThetaSoftmaxOpponent(RANDOM_THETA).probabilities(view)
        greedy_probabilities = ThetaSoftmaxOpponent(GREEDY_POINTS_THETA).probabilities(view)
        compact_probabilities = ThetaSoftmaxOpponent(
            COMPACT_GREEDY_POINTS_THETA,
            feature_names=COMPACT_FEATURE_NAMES,
        ).probabilities(view)
        trump_count_probabilities = ThetaSoftmaxOpponent(
            TRUMP_COUNT_GREEDY_POINTS_THETA,
            feature_names=TRUMP_COUNT_FEATURE_NAMES,
        ).probabilities(view)
        style_probabilities = ThetaSoftmaxOpponent(
            STYLE_GREEDY_POINTS_THETA,
            feature_names=STYLE_FEATURE_NAMES,
        ).probabilities(view)

        self.assertEqual(len(features), len(RANDOM_THETA))
        self.assertEqual(len(compact_features), len(COMPACT_GREEDY_POINTS_THETA))
        self.assertEqual(len(trump_count_features), len(TRUMP_COUNT_GREEDY_POINTS_THETA))
        self.assertEqual(len(style_features), len(STYLE_GREEDY_POINTS_THETA))
        self.assertAlmostEqual(sum(random_probabilities.values()), 1.0)
        self.assertAlmostEqual(sum(greedy_probabilities.values()), 1.0)
        self.assertAlmostEqual(sum(compact_probabilities.values()), 1.0)
        self.assertAlmostEqual(sum(trump_count_probabilities.values()), 1.0)
        self.assertAlmostEqual(sum(style_probabilities.values()), 1.0)
        self.assertGreater(
            greedy_probabilities[Card(Rank.ACE, Suit.CUPS)],
            greedy_probabilities[Card(Rank.TWO, Suit.SWORDS)],
        )


if __name__ == "__main__":
    unittest.main()
