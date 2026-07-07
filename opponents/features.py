"""Feature extraction for the softmax opponent model"""

from __future__ import annotations

from collections.abc import Sequence

from game import Card, PlayedCard, PlayerId, PublicState, Rank, full_deck, trick_winner

MAX_CARD_POINTS = 11
# Strength is encoded from 0 to 9 in the Briscola rank order.
MAX_CARD_STRENGTH = 9
TOTAL_CARDS = 40
TOTAL_TRUMPS = 10
# At most nine stronger cards can share the same suit.
MAX_HIGHER_SAME_SUIT = 9

CORE_FEATURE_NAMES: tuple[str, ...] = (
    "is_trump",
    "points_normalized",
    "wins_current_trick",
    "lowest_card_in_suit",
)

TRUMP_COUNT_FEATURE_NAMES: tuple[str, ...] = (
    "trumps_remaining_after",
    "points_normalized",
    "wins_current_trick",
    "lowest_card_in_suit",
)

COMPACT_FEATURE_NAMES: tuple[str, ...] = (
    "is_trump",
    "points_normalized",
    "wins_current_trick",
    "lowest_card_in_suit",
    "is_smooth",
)

EXTENDED_FEATURE_NAMES: tuple[str, ...] = (
    *CORE_FEATURE_NAMES,
    "strength_normalized",
    "is_ace",
    "is_three",
    "is_load",
    "is_smooth",
    "current_trick_points_normalized",
    "score_difference_normalized",
    "played_cards_normalized",
    "stock_empty",
    "late_game",
    "higher_same_suit_unseen_normalized",
    "higher_trumps_unseen_normalized",
)

FEATURE_NAMES = CORE_FEATURE_NAMES


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

    acting_player = _acting_player(public_state, player)
    is_trump = float(card.suit == public_state.trump_suit)
    points_normalized = card.points / MAX_CARD_POINTS
    wins_current_trick = float(_wins_current_trick(card, public_state, player))
    lowest_card_in_suit = float(_is_lowest_card_in_suit(card, hand))
    plays_first = float(len(public_state.current_trick) == 0)
    trumps_remaining_after = float(
        sum(1 for other in hand if other != card and other.suit == public_state.trump_suit)
    )
    strength_normalized = card.strength / MAX_CARD_STRENGTH
    is_ace = float(card.rank == Rank.ACE)
    is_three = float(card.rank == Rank.THREE)
    is_load = float(card.rank in (Rank.ACE, Rank.THREE))
    is_smooth = float(card.points == 0 and card.suit != public_state.trump_suit)
    current_trick_points_normalized = (
        sum(played.card.points for played in public_state.current_trick) / MAX_CARD_POINTS
    )
    score_difference_normalized = (
        public_state.scores[acting_player] - public_state.scores[1 - acting_player]
    ) / 120.0
    played_cards_normalized = len(public_state.played_cards) / TOTAL_CARDS
    stock_empty = float(public_state.stock_size == 0)
    late_game = float(public_state.tricks_played >= 14)
    observed_cards = _observed_cards(hand, public_state)
    higher_same_suit_unseen_normalized = (
        _higher_same_suit_unseen(card, observed_cards) / MAX_HIGHER_SAME_SUIT
    )
    higher_trumps_unseen_normalized = (
        _higher_trumps_unseen(card, public_state, observed_cards) / TOTAL_TRUMPS
    )

    return {
        "is_trump": is_trump,
        "points_normalized": points_normalized,
        "wins_current_trick": wins_current_trick,
        "lowest_card_in_suit": lowest_card_in_suit,
        "plays_first": plays_first,
        "trumps_remaining_after": trumps_remaining_after,
        "strength_normalized": strength_normalized,
        "is_ace": is_ace,
        "is_three": is_three,
        "is_load": is_load,
        "is_smooth": is_smooth,
        "current_trick_points_normalized": current_trick_points_normalized,
        "score_difference_normalized": score_difference_normalized,
        "played_cards_normalized": played_cards_normalized,
        "stock_empty": stock_empty,
        "late_game": late_game,
        "higher_same_suit_unseen_normalized": higher_same_suit_unseen_normalized,
        "higher_trumps_unseen_normalized": higher_trumps_unseen_normalized,
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


def _acting_player(public_state: PublicState, player: PlayerId | None) -> PlayerId:
    acting_player = public_state.current_player if player is None else player
    if acting_player is None:
        raise ValueError("player is required when the game is finished")
    return acting_player


def _is_lowest_card_in_suit(card: Card, hand: tuple[Card, ...]) -> bool:
    same_suit = [candidate for candidate in hand if candidate.suit == card.suit]
    return card.strength == min(candidate.strength for candidate in same_suit)


def _observed_cards(hand: tuple[Card, ...], public_state: PublicState) -> set[Card]:
    observed = set(hand)
    observed.update(played.card for played in public_state.current_trick)
    observed.update(played.card for played in public_state.played_cards)
    if public_state.trump_card_in_stock:
        observed.add(public_state.trump_card)
    return observed


def _higher_same_suit_unseen(card: Card, observed_cards: set[Card]) -> int:
    return sum(
        1
        for candidate in full_deck()
        if candidate.suit == card.suit
        and candidate.strength > card.strength
        and candidate not in observed_cards
    )


def _higher_trumps_unseen(
    card: Card,
    public_state: PublicState,
    observed_cards: set[Card],
) -> int:
    return sum(
        1
        for candidate in full_deck()
        if candidate.suit == public_state.trump_suit
        and candidate not in observed_cards
        and (card.suit != public_state.trump_suit or candidate.strength > card.strength)
    )
