import math
import unittest

from game import BriscolaGame, Card, Rank, Suit, full_deck
from inference.beliefs import (
    compatible_hands,
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
)


def deck_with_prefix(prefix: list[Card]) -> tuple[Card, ...]:
    remaining = [card for card in full_deck() if card not in prefix]
    return tuple(prefix + remaining)


class BeliefEnumerationTest(unittest.TestCase):
    def test_unknown_cards_exclude_observer_known_cards(self) -> None:
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

        unknown_at_start = compatible_unknown_cards(
            game.public_state(),
            tuple(game.hands[0]),
        )

        self.assertEqual(len(unknown_at_start), 36)
        self.assertTrue(all(card not in unknown_at_start for card in game.hands[0]))
        self.assertNotIn(game.trump_card, unknown_at_start)

        led_card = game.hands[0][0]
        game.play_card(0, led_card)
        unknown_after_lead = compatible_unknown_cards(
            game.public_state(),
            tuple(game.hands[0]),
        )

        self.assertEqual(len(unknown_after_lead), 36)
        self.assertNotIn(led_card, unknown_after_lead)

    def test_compatible_hand_enumeration_and_counts(self) -> None:
        cards = full_deck()[:4]
        observed_card = cards[0]

        hands = compatible_hands(cards, hand_size=2)
        containing = compatible_hands_containing(
            cards,
            hand_size=3,
            observed_card=observed_card,
        )

        self.assertEqual(len(hands), math.comb(4, 2))
        self.assertEqual(len(containing), math.comb(3, 2))
        self.assertTrue(all(observed_card in hand for hand in containing))
        self.assertEqual(hand_count(4, 2), math.comb(4, 2))
        self.assertEqual(hand_count(2, 3), 0)

    def test_negative_hand_size_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            compatible_hands(full_deck(), hand_size=-1)
        with self.assertRaises(ValueError):
            compatible_hands_containing(full_deck(), -1, full_deck()[0])
        with self.assertRaises(ValueError):
            hand_count(4, -1)


if __name__ == "__main__":
    unittest.main()
