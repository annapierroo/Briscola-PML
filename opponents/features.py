"""Feature extraction for the softmax opponent model"""

from __future__ import annotations

from collections.abc import Sequence

from game import Card, PlayedCard, PlayerId, PublicState, trick_winner

MAX_CARD_POINTS = 11
TOTAL_CARDS = 40

CORE_FEATURE_NAMES: tuple[str, ...] = (
    "is_trump",
    "points_normalized",
    "wins_current_trick",
    "lowest_card_in_suit",
)

INTERACTIVE_FEATURE_NAMES: tuple[str, ...] = (
    "is_trump",
    "points_normalized",
    "wins_current_trick",
    "lowest_card_in_suit",
    "trump_progress",
    "points_progress",
    "trump_on_table_points",
    "greedy_take",
)

FEATURE_NAMES = INTERACTIVE_FEATURE_NAMES


def card_features(
    card: Card,
    hand: tuple[Card, ...],
    public_state: PublicState,
    player: PlayerId | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[float, ...]:
    """Feature vector for playing this card in the current state"""

    values = feature_dict(card, hand, public_state, player)
    unknown_names = tuple(name for name in feature_names if name not in values)
    if unknown_names:
        raise ValueError(f"unknown feature names: {unknown_names}")

    return tuple(float(values[name]) for name in feature_names)


def feature_dict(
    card: Card,
    hand: tuple[Card, ...],
    public_state: PublicState,
    player: PlayerId | None = None,
) -> dict[str, float]:
    """All feature values, keyed by name"""

    if card not in hand:
        raise ValueError(f"card {card} is not in the candidate hand")
    if len(public_state.current_trick) > 1:
        raise ValueError("current_trick cannot contain more than one card before a move")

    is_trump = float(card.suit == public_state.trump_suit)
    points_normalized = card.points / MAX_CARD_POINTS
    wins_current_trick = float(_wins_current_trick(card, public_state, player))
    lowest_card_in_suit = float(_is_lowest_card_in_suit(card, hand))
    current_trick_points_normalized = (
        sum(played.card.points for played in public_state.current_trick) / MAX_CARD_POINTS
    )
    played_cards_normalized = len(public_state.played_cards) / TOTAL_CARDS

    trump_progress = is_trump * played_cards_normalized
    points_progress = points_normalized * played_cards_normalized
    trump_on_table_points = is_trump * current_trick_points_normalized
    greedy_take = wins_current_trick * current_trick_points_normalized

    return {
        "is_trump": is_trump,
        "points_normalized": points_normalized,
        "wins_current_trick": wins_current_trick,
        "lowest_card_in_suit": lowest_card_in_suit,
        "trump_progress": trump_progress,
        "points_progress": points_progress,
        "trump_on_table_points": trump_on_table_points,
        "greedy_take": greedy_take,
    }


def _wins_current_trick(
    card: Card,
    public_state: PublicState,
    player: PlayerId | None,
) -> bool:
    if not public_state.current_trick:
        return False

    first = public_state.current_trick[0]
    acting_player = public_state.current_player if player is None else player
    if acting_player is None:
        raise ValueError("player is required when evaluating a second move")

    second = PlayedCard(player=acting_player, card=card)
    return trick_winner(first, second, public_state.trump_suit) == acting_player


def _is_lowest_card_in_suit(card: Card, hand: tuple[Card, ...]) -> bool:
    same_suit = [candidate for candidate in hand if candidate.suit == card.suit]
    return card.strength == min(candidate.strength for candidate in same_suit)
