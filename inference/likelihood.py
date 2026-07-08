"""Marginal likelihood over hidden opponent hands"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from game import Card, PlayerId, PublicState
from inference.beliefs import (
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
    known_opponent_cards,
)
from opponents import FEATURE_NAMES, card_features


def local_card_probability(
    observed_card: Card,
    hand: tuple[Card, ...],
    public_state: PublicState,
    theta: Sequence[float],
    *,
    player: PlayerId | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    temperature: float = 1.0,
) -> float:
    """Probability of one card given one candidate hand"""

    if observed_card not in hand:
        return 0.0
    scores = {
        card: _dot(
            theta,
            card_features(
                card,
                hand,
                public_state,
                player,
                feature_names=feature_names,
            ),
        )
        for card in hand
    }
    return _softmax_probability(scores, observed_card, temperature)


def marginal_card_probability(
    observed_card: Card,
    public_state: PublicState,
    observer_hand: tuple[Card, ...],
    theta: Sequence[float],
    *,
    observed_player: PlayerId | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    temperature: float = 1.0,
    max_exact_hands: int | None = None,
    mc_samples: int | None = None,
    seed: int | None = None,
) -> float:
    """Absolute marginal probability of the observed card under public information."""

    _validate_theta(theta, feature_names)

    opponent_player = _target_player(public_state, observed_player)
    unknown_cards = compatible_unknown_cards(
        public_state,
        observer_hand,
        opponent_player=opponent_player,
    )
    required_cards = known_opponent_cards(
        public_state,
        observer_hand,
        opponent_player=opponent_player,
    )
    hand_size = _opponent_hand_size(public_state, opponent_player)
    if observed_card not in unknown_cards or any(
        card not in unknown_cards for card in required_cards
    ):
        return 0.0

    required_with_observed = _required_cards_with_observed(required_cards, observed_card)
    containing_count = hand_count(
        len(unknown_cards),
        hand_size,
        required_count=len(required_with_observed),
    )
    if containing_count == 0:
        return 0.0

    use_mc = (
        mc_samples is not None
        and mc_samples > 0
        and max_exact_hands is not None
        and containing_count > max_exact_hands
    )
    if use_mc:
        average_probability = _mc_average_probability(
            observed_card=observed_card,
            unknown_cards=unknown_cards,
            required_cards=required_cards,
            hand_size=hand_size,
            public_state=public_state,
            theta=theta,
            player=opponent_player,
            feature_names=feature_names,
            temperature=temperature,
            num_samples=mc_samples,
            seed=seed,
        )
    else:
        average_probability = _exact_average_probability(
            observed_card=observed_card,
            unknown_cards=unknown_cards,
            required_cards=required_cards,
            hand_size=hand_size,
            public_state=public_state,
            theta=theta,
            player=opponent_player,
            feature_names=feature_names,
            temperature=temperature,
        )

    total_count = hand_count(
        len(unknown_cards),
        hand_size,
        required_count=len(required_cards),
    )
    if total_count == 0:
        return 0.0
    # Restore P(card in hand | public information) for a normalized predictive probability.
    return average_probability * containing_count / total_count


def marginal_log_likelihood(
    observations: Sequence[object],
    theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    temperature: float = 1.0,
    max_exact_hands: int | None = None,
    mc_samples: int | None = None,
    seed: int | None = None,
) -> float:
    """Sum log marginal probabilities over collected observations"""

    total = 0.0
    for index, observation in enumerate(observations):
        probability = marginal_card_probability(
            observed_card=observation.chosen_card,
            public_state=observation.public_state,
            observer_hand=observation.observer_hand,
            theta=theta,
            observed_player=observation.player,
            feature_names=feature_names,
            temperature=temperature,
            max_exact_hands=max_exact_hands,
            mc_samples=mc_samples,
            seed=None if seed is None else seed + index,
        )
        if probability <= 0.0:
            return -math.inf
        total += math.log(probability)
    return total


def _exact_average_probability(
    *,
    observed_card: Card,
    unknown_cards: tuple[Card, ...],
    required_cards: tuple[Card, ...],
    hand_size: int,
    public_state: PublicState,
    theta: Sequence[float],
    player: PlayerId | None,
    feature_names: Sequence[str],
    temperature: float,
) -> float:
    hands = compatible_hands_containing(
        unknown_cards,
        hand_size,
        observed_card,
        required_cards=required_cards,
    )
    if not hands:
        return 0.0

    return sum(
        local_card_probability(
            observed_card,
            hand,
            public_state,
            theta,
            player=player,
            feature_names=feature_names,
            temperature=temperature,
        )
        for hand in hands
    ) / len(hands)


def _mc_average_probability(
    *,
    observed_card: Card,
    unknown_cards: tuple[Card, ...],
    required_cards: tuple[Card, ...],
    hand_size: int,
    public_state: PublicState,
    theta: Sequence[float],
    player: PlayerId | None,
    feature_names: Sequence[str],
    temperature: float,
    num_samples: int,
    seed: int | None,
) -> float:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if hand_size <= 0:
        return 0.0

    rng = random.Random(seed)
    base_hand = _required_cards_with_observed(required_cards, observed_card)
    draw_count = hand_size - len(base_hand)
    if draw_count < 0:
        return 0.0
    remaining_cards = [card for card in unknown_cards if card not in base_hand]
    if draw_count > len(remaining_cards):
        return 0.0
    total = 0.0
    for _ in range(num_samples):
        rest = tuple(rng.sample(remaining_cards, draw_count))
        hand = (*base_hand, *rest)
        total += local_card_probability(
            observed_card,
            hand,
            public_state,
            theta,
            player=player,
            feature_names=feature_names,
            temperature=temperature,
        )
    return total / num_samples


def _target_player(
    public_state: PublicState,
    observed_player: PlayerId | None,
) -> PlayerId | None:
    return public_state.current_player if observed_player is None else observed_player


def _opponent_hand_size(
    public_state: PublicState,
    observed_player: PlayerId | None,
) -> int:
    if observed_player is not None:
        return public_state.hand_sizes[observed_player]
    current_player = public_state.current_player
    if current_player is None:
        raise ValueError("observed_player is required when no current player is set")
    return public_state.hand_sizes[current_player]


def _required_cards_with_observed(
    required_cards: tuple[Card, ...],
    observed_card: Card,
) -> tuple[Card, ...]:
    return (observed_card,) + tuple(card for card in required_cards if card != observed_card)


def _softmax_probability(
    scores: dict[Card, float],
    selected_card: Card,
    temperature: float,
) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    max_score = max(scores.values())
    selected_weight = math.exp((scores[selected_card] - max_score) / temperature)
    normalizer = sum(
        math.exp((score - max_score) / temperature)
        for score in scores.values()
    )
    return selected_weight / normalizer


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _validate_theta(theta: Sequence[float], feature_names: Sequence[str]) -> None:
    if len(theta) != len(feature_names):
        raise ValueError("theta length must match feature_names length")
