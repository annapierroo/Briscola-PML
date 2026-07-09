"""Uniform local beliefs over possible opponent hands"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
import math

from game import Card, PublicState, full_deck


def compatible_unknown_cards(
    public_state: PublicState,
    observer_hand: tuple[Card, ...],
    *,
    opponent_player: int | None = None,
) -> tuple[Card, ...]:
    """Cards that could still belong to the opponent"""

    # Baseline belief: remove known cards, but not filter by draw order
    known_cards = set(observer_hand)
    known_cards.update(played.card for played in public_state.played_cards)
    known_cards.update(played.card for played in public_state.current_trick)
    if public_state.trump_card_in_stock:
        known_cards.add(public_state.trump_card)
    if opponent_player is not None:
        for player, cards in enumerate(public_state.known_cards_by_player):
            if player != opponent_player:
                known_cards.update(cards)

    return tuple(card for card in full_deck() if card not in known_cards)


def known_opponent_cards(
    public_state: PublicState,
    observer_hand: tuple[Card, ...],
    *,
    opponent_player: int | None,
) -> tuple[Card, ...]:
    """Publicly known cards that must be in the opponent hand"""

    if opponent_player is None:
        return ()

    unavailable = set(observer_hand)
    unavailable.update(played.card for played in public_state.played_cards)
    unavailable.update(played.card for played in public_state.current_trick)
    return tuple(
        card
        for card in public_state.known_cards_by_player[opponent_player]
        if card not in unavailable
    )


def compatible_hands(
    unknown_cards: Iterable[Card],
    hand_size: int,
    *,
    required_cards: Iterable[Card] = (),
) -> tuple[tuple[Card, ...], ...]:
    """All hands of the requested size from the unknown-card pool"""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    cards = tuple(unknown_cards)
    required = _validated_required_cards(cards, required_cards)
    if required is None:
        return ()
    if hand_size > len(cards) or len(required) > hand_size:
        return ()
    remaining_cards = tuple(card for card in cards if card not in required)
    rest_size = hand_size - len(required)
    return tuple(
        (*required, *rest)
        for rest in combinations(remaining_cards, rest_size)
    )


def compatible_hands_containing(
    unknown_cards: Iterable[Card],
    hand_size: int,
    observed_card: Card,
    *,
    required_cards: Iterable[Card] = (),
) -> tuple[tuple[Card, ...], ...]:
    """All compatible hands that contain the observed card"""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    cards = tuple(unknown_cards)
    if observed_card not in cards or hand_size == 0:
        return ()

    required = _validated_required_cards(cards, required_cards)
    if required is None:
        return ()
    ordered_required = (observed_card,) + tuple(
        card for card in required if card != observed_card
    )
    if len(ordered_required) > hand_size:
        return ()

    remaining_cards = tuple(card for card in cards if card not in ordered_required)
    rest_size = hand_size - len(ordered_required)
    if rest_size > len(remaining_cards):
        return ()

    return tuple(
        (*ordered_required, *rest)
        for rest in combinations(remaining_cards, rest_size)
    )


def hand_count(num_cards: int, hand_size: int, required_count: int = 0) -> int:
    """Number of hands in a uniform local belief"""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    if required_count < 0:
        raise ValueError("required_count must be non-negative")
    if (
        hand_size > num_cards
        or required_count > hand_size
        or required_count > num_cards
    ):
        return 0
    return math.comb(num_cards - required_count, hand_size - required_count)


def _validated_required_cards(
    cards: tuple[Card, ...],
    required_cards: Iterable[Card],
) -> tuple[Card, ...] | None:
    required = tuple(dict.fromkeys(required_cards))
    if any(card not in cards for card in required):
        return None
    return required
