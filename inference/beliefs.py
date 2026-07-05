"""Uniform local beliefs over possible opponent hands."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
import math

from game import Card, PublicState, full_deck


def compatible_unknown_cards(
    public_state: PublicState,
    observer_hand: tuple[Card, ...],
) -> tuple[Card, ...]:
    """Cards that could still belong to the opponent.

    This implements the simplifying assumption from the project formulation:
    local opponent hands are treated as uniformly distributed over the cards
    not known to the observer. Exact draw-order filtering is deliberately left
    for a later extension.
    """

    known_cards = set(observer_hand)
    known_cards.update(played.card for played in public_state.played_cards)
    known_cards.update(played.card for played in public_state.current_trick)
    if public_state.trump_card_in_stock:
        known_cards.add(public_state.trump_card)

    return tuple(card for card in full_deck() if card not in known_cards)


def compatible_hands(
    unknown_cards: Iterable[Card],
    hand_size: int,
) -> tuple[tuple[Card, ...], ...]:
    """All hands of the requested size from the unknown-card pool."""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    cards = tuple(unknown_cards)
    if hand_size > len(cards):
        return ()
    return tuple(combinations(cards, hand_size))


def compatible_hands_containing(
    unknown_cards: Iterable[Card],
    hand_size: int,
    observed_card: Card,
) -> tuple[tuple[Card, ...], ...]:
    """All compatible hands that contain the observed card."""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    cards = tuple(unknown_cards)
    if observed_card not in cards or hand_size == 0:
        return ()
    if hand_size == 1:
        return ((observed_card,),)

    remaining_cards = tuple(card for card in cards if card != observed_card)
    if hand_size - 1 > len(remaining_cards):
        return ()
    return tuple(
        (observed_card, *rest)
        for rest in combinations(remaining_cards, hand_size - 1)
    )


def hand_count(num_cards: int, hand_size: int) -> int:
    """Number of hands in a uniform local belief."""

    if hand_size < 0:
        raise ValueError("hand_size must be non-negative")
    if hand_size > num_cards:
        return 0
    return math.comb(num_cards, hand_size)
