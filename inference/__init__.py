"""Inference helpers for hidden-hand marginalization"""

from inference.beliefs import (
    compatible_hands,
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
    known_opponent_cards,
)
from inference.likelihood import (
    LikelihoodMode,
    local_card_probability,
    marginal_card_probability,
    marginal_log_likelihood,
)

from inference.vi import (
    PreparedObservation,
    PreparedSequentialGame,
    PreparedSequentialObservation,
    VariationalPosterior,
    diag_gaussian_kl,
    fit_variational_posterior,
    log_likelihood_torch,
    log_marginal_probability_torch,
    log_sequential_game_probability_torch,
    log_sequential_likelihood_torch,
    prepare_observations,
    prepare_sequential_games,
    sequential_log_likelihood,
)

__all__ = [
    "LikelihoodMode",
    "PreparedObservation",
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
    "local_card_probability",
    "log_likelihood_torch",
    "log_marginal_probability_torch",
    "log_sequential_game_probability_torch",
    "log_sequential_likelihood_torch",
    "marginal_card_probability",
    "marginal_log_likelihood",
    "prepare_observations",
    "prepare_sequential_games",
    "sequential_log_likelihood",
]
