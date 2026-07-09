"""Inference helpers for sequential hidden-hand filtering"""

from inference.beliefs import (
    compatible_hands,
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
    known_opponent_cards,
)
from inference.vi import (
    PreparedSequentialGame,
    PreparedSequentialObservation,
    VariationalPosterior,
    diag_gaussian_kl,
    fit_variational_posterior,
    log_sequential_game_probability_torch,
    log_sequential_likelihood_torch,
    prepare_sequential_games,
    sequential_log_likelihood,
)

__all__ = [
    "PreparedSequentialGame",
    "PreparedSequentialObservation",
    "VariationalPosterior",
    "compatible_hands",
    "compatible_hands_containing",
    "compatible_unknown_cards",
    "diag_gaussian_kl",
    "fit_variational_posterior",
    "hand_count",
    "known_opponent_cards",
    "log_sequential_game_probability_torch",
    "log_sequential_likelihood_torch",
    "prepare_sequential_games",
    "sequential_log_likelihood",
]
