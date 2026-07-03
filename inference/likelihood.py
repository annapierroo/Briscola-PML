"""Marginal likelihood over hidden opponent hands"""

from __future__ import annotations

from enum import Enum
import math
import random
from collections.abc import Sequence

from game import Card, PlayerId, PublicState
from inference.beliefs import (
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
)
from opponents import FEATURE_NAMES, card_features


class LikelihoodMode(str, Enum):
    """Whether to include the chance that the card is in hand"""

    CONDITIONAL = "conditional" # p(playing card | card in had, public state, theta)
    ABSOLUTE = "absolute" # p(card in hand) * p(playing card | card in hand, public state, theta)


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
    mode: LikelihoodMode = LikelihoodMode.CONDITIONAL,
    temperature: float = 1.0,
    max_exact_hands: int | None = None,
    mc_samples: int | None = None,
    seed: int | None = None,
) -> float:
    """Marginal probability of the observed card under a uniform hand belief"""

    mode = LikelihoodMode(mode)
    _validate_theta(theta, feature_names)

    unknown_cards = compatible_unknown_cards(public_state, observer_hand)
    hand_size = _opponent_hand_size(public_state, observed_player)
    containing_count = hand_count(len(unknown_cards) - 1, hand_size - 1)
    if observed_card not in unknown_cards or containing_count == 0:
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
            hand_size=hand_size,
            public_state=public_state,
            theta=theta,
            player=observed_player,
            feature_names=feature_names,
            temperature=temperature,
            num_samples=mc_samples,
            seed=seed,
        )
    else:
        average_probability = _exact_average_probability(
            observed_card=observed_card,
            unknown_cards=unknown_cards,
            hand_size=hand_size,
            public_state=public_state,
            theta=theta,
            player=observed_player,
            feature_names=feature_names,
            temperature=temperature,
        )

    if mode == LikelihoodMode.CONDITIONAL:
        return average_probability

    total_count = hand_count(len(unknown_cards), hand_size)
    return average_probability * containing_count / total_count


def marginal_log_likelihood(
    observations: Sequence[object],
    theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    mode: LikelihoodMode = LikelihoodMode.CONDITIONAL,
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
            mode=mode,
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
    hand_size: int,
    public_state: PublicState,
    theta: Sequence[float],
    player: PlayerId | None,
    feature_names: Sequence[str],
    temperature: float,
) -> float:
    hands = compatible_hands_containing(unknown_cards, hand_size, observed_card)
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
    remaining_cards = [card for card in unknown_cards if card != observed_card]
    total = 0.0
    for _ in range(num_samples):
        rest = tuple(rng.sample(remaining_cards, hand_size - 1))
        hand = (observed_card, *rest)
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
