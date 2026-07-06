"""Inference helpers for hidden-hand marginalization"""

from inference.beliefs import (
    compatible_hands,
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
)
from inference.likelihood import (
    LikelihoodMode,
    local_card_probability,
    marginal_card_probability,
    marginal_log_likelihood,
)

from inference.vi import (
    PreparedObservation,
    VariationalPosterior,
    diag_gaussian_kl,
    fit_variational_posterior,
    log_likelihood_torch,
    log_marginal_probability_torch,
    prepare_observations,
)

__all__ = [
    "CalibrationBin",
    "CalibrationResult",
    "ImportanceSamplingResult",
    "MoveChooser",
    "OpponentMoveObservation",
    "PredictiveResult",
    "RecoveryResult",
    "SyntheticGameResult",
    "calibration_curve",
    "collect_matched_model_observations",
    "collect_observations",
    "heldout_predictive_evaluation",
    "importance_sampling_reference",
    "play_synthetic_game",
    "run_recovery_experiment",
    "train_test_split",
]
