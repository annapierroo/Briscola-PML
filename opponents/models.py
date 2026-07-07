"""Synthetic opponent models used to generate observed Briscola moves"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol, Sequence

from game import Card, PlayerView
from opponents.features import (
    COMPACT_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    FEATURE_NAMES,
    TRUMP_COUNT_FEATURE_NAMES,
    card_features,
)


def theta_from_weights(
    feature_names: Sequence[str] = FEATURE_NAMES,
    **weights: float,
) -> tuple[float, ...]:
    """Builds a theta vector from named feature weights"""

    unknown_features = set(weights) - set(feature_names)
    if unknown_features:
        raise ValueError(f"unknown theta weights: {sorted(unknown_features)}")
    return tuple(float(weights.get(name, 0.0)) for name in feature_names)


def zero_theta(feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[float, ...]:
    """Return a theta vector that induces a uniform softmax"""

    return tuple(0.0 for _ in feature_names)


# Hand-tuned profiles used to generate synthetic data.
RANDOM_THETA = zero_theta()
AGGRESSIVE_THETA = theta_from_weights(
    is_trump=0.4,
    points_normalized=1.4,
    wins_current_trick=2.2,
    lowest_card_in_suit=-0.2,
)
CONSERVATIVE_THETA = theta_from_weights(
    is_trump=-1.8,
    points_normalized=-1.1,
    wins_current_trick=0.7,
    lowest_card_in_suit=1.2,
)
GREEDY_POINTS_THETA = theta_from_weights(
    is_trump=0.1,
    points_normalized=3.0,
    wins_current_trick=0.4,
    lowest_card_in_suit=-0.2,
)

TRUMP_COUNT_RANDOM_THETA = zero_theta(TRUMP_COUNT_FEATURE_NAMES)
TRUMP_COUNT_AGGRESSIVE_THETA = theta_from_weights(
    TRUMP_COUNT_FEATURE_NAMES,
    trumps_remaining_after=-0.3,
    points_normalized=1.4,
    wins_current_trick=2.2,
    lowest_card_in_suit=-0.2,
)
TRUMP_COUNT_CONSERVATIVE_THETA = theta_from_weights(
    TRUMP_COUNT_FEATURE_NAMES,
    trumps_remaining_after=0.8,
    points_normalized=-1.1,
    wins_current_trick=0.7,
    lowest_card_in_suit=1.2,
)
TRUMP_COUNT_GREEDY_POINTS_THETA = theta_from_weights(
    TRUMP_COUNT_FEATURE_NAMES,
    points_normalized=3.0,
    wins_current_trick=0.4,
    lowest_card_in_suit=-0.2,
)

COMPACT_RANDOM_THETA = zero_theta(COMPACT_FEATURE_NAMES)
COMPACT_AGGRESSIVE_THETA = theta_from_weights(
    COMPACT_FEATURE_NAMES,
    is_trump=0.4,
    points_normalized=1.4,
    wins_current_trick=2.2,
    lowest_card_in_suit=-0.2,
    is_smooth=-0.3,
)
COMPACT_CONSERVATIVE_THETA = theta_from_weights(
    COMPACT_FEATURE_NAMES,
    is_trump=-1.8,
    points_normalized=-1.1,
    wins_current_trick=0.7,
    lowest_card_in_suit=1.2,
    is_smooth=0.8,
)
COMPACT_GREEDY_POINTS_THETA = theta_from_weights(
    COMPACT_FEATURE_NAMES,
    is_trump=0.1,
    points_normalized=3.0,
    wins_current_trick=0.4,
    lowest_card_in_suit=-0.2,
    is_smooth=-0.4,
)

EXTENDED_RANDOM_THETA = zero_theta(EXTENDED_FEATURE_NAMES)
EXTENDED_AGGRESSIVE_THETA = theta_from_weights(
    EXTENDED_FEATURE_NAMES,
    is_trump=0.3,
    points_normalized=1.2,
    wins_current_trick=2.0,
    lowest_card_in_suit=-0.2,
    strength_normalized=0.7,
    is_ace=0.4,
    is_three=0.3,
    is_load=0.4,
    is_smooth=-0.3,
    current_trick_points_normalized=0.9,
    score_difference_normalized=-0.1,
    stock_empty=0.2,
    late_game=0.4,
    higher_same_suit_unseen_normalized=-0.2,
    higher_trumps_unseen_normalized=-0.3,
)
EXTENDED_CONSERVATIVE_THETA = theta_from_weights(
    EXTENDED_FEATURE_NAMES,
    is_trump=-1.4,
    points_normalized=-0.9,
    wins_current_trick=0.5,
    lowest_card_in_suit=1.0,
    strength_normalized=-0.4,
    is_ace=-0.5,
    is_three=-0.4,
    is_load=-0.5,
    is_smooth=0.8,
    current_trick_points_normalized=0.2,
    score_difference_normalized=0.1,
    stock_empty=0.4,
    late_game=0.5,
    higher_same_suit_unseen_normalized=0.3,
    higher_trumps_unseen_normalized=0.5,
)
EXTENDED_GREEDY_POINTS_THETA = theta_from_weights(
    EXTENDED_FEATURE_NAMES,
    is_trump=0.1,
    points_normalized=2.8,
    wins_current_trick=0.3,
    lowest_card_in_suit=-0.2,
    strength_normalized=0.2,
    is_ace=0.4,
    is_three=0.4,
    is_load=0.4,
    is_smooth=-0.4,
    current_trick_points_normalized=0.2,
)


class OpponentModel(Protocol):
    """Interface shared by synthetic move generators"""

    def choose_card(self, view: PlayerView) -> Card:
        """Choose one card from the player's current hand"""


@dataclass(slots=True)
class RandomOpponent:
    """Uniform random opponent over legal cards in hand"""

    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_card(self, view: PlayerView) -> Card:
        _validate_non_empty_hand(view)
        return self._rng.choice(view.hand)


@dataclass(slots=True)
class ThetaSoftmaxOpponent:
    """Opponent that samples moves with softmax(theta dot phi)"""

    theta: Sequence[float]
    seed: int | None = None
    temperature: float = 1.0
    feature_names: Sequence[str] = FEATURE_NAMES
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.feature_names = tuple(self.feature_names)
        if len(self.theta) != len(self.feature_names):
            raise ValueError(f"theta must have length {len(self.feature_names)}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.theta = tuple(float(value) for value in self.theta)
        self._rng = random.Random(self.seed)

    def scores(self, view: PlayerView) -> dict[Card, float]:
        _validate_non_empty_hand(view)
        return {
            card: _dot(
                self.theta,
                card_features(
                    card,
                    view.hand,
                    view.public_state,
                    view.player,
                    feature_names=self.feature_names,
                ),
            )
            for card in view.hand
        }

    def probabilities(self, view: PlayerView) -> dict[Card, float]:
        return softmax(self.scores(view), temperature=self.temperature)

    def choose_card(self, view: PlayerView) -> Card:
        probabilities = self.probabilities(view)
        threshold = self._rng.random()
        cumulative = 0.0

        for card, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                return card

        return next(reversed(probabilities))


def softmax(scores: dict[Card, float], temperature: float = 1.0) -> dict[Card, float]:
    """Return numerically stable softmax probabilities for card scores"""

    if not scores:
        raise ValueError("scores cannot be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    max_score = max(scores.values())
    weights = {
        card: math.exp((score - max_score) / temperature)
        for card, score in scores.items()
    }
    normalizer = sum(weights.values())
    return {card: weight / normalizer for card, weight in weights.items()}


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _validate_non_empty_hand(view: PlayerView) -> None:
    if not view.hand:
        raise ValueError("cannot choose a card from an empty hand")
