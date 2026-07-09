"""Opponent models used to generate synthetic games"""

from opponents.features import (
    CORE_FEATURE_NAMES,
    FEATURE_NAMES,
    INTERACTIVE_FEATURE_NAMES,
    card_features,
    feature_dict,
)
from opponents.models import (
    AGGRESSIVE_THETA,
    CONSERVATIVE_THETA,
    GREEDY_POINTS_THETA,
    INTERACTIVE_AGGRESSIVE_THETA,
    INTERACTIVE_CONSERVATIVE_THETA,
    INTERACTIVE_GREEDY_POINTS_THETA,
    INTERACTIVE_RANDOM_THETA,
    RANDOM_THETA,
    RandomOpponent,
    ThetaSoftmaxOpponent,
    theta_from_weights,
    zero_theta,
)

__all__ = [
    "AGGRESSIVE_THETA",
    "CONSERVATIVE_THETA",
    "CORE_FEATURE_NAMES",
    "FEATURE_NAMES",
    "GREEDY_POINTS_THETA",
    "INTERACTIVE_AGGRESSIVE_THETA",
    "INTERACTIVE_CONSERVATIVE_THETA",
    "INTERACTIVE_FEATURE_NAMES",
    "INTERACTIVE_GREEDY_POINTS_THETA",
    "INTERACTIVE_RANDOM_THETA",
    "RANDOM_THETA",
    "RandomOpponent",
    "ThetaSoftmaxOpponent",
    "card_features",
    "feature_dict",
    "theta_from_weights",
    "zero_theta",
]
