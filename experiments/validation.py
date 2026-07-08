"""Validation experiments for synthetic opponent modelling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import random

from experiments.episode_collection import OpponentMoveObservation, collect_observations
from game import Card
from inference import (
    compatible_unknown_cards,
    fit_variational_posterior,
    known_opponent_cards,
    local_card_probability,
    sequential_log_likelihood,
)
from opponents import FEATURE_NAMES, RandomOpponent, ThetaSoftmaxOpponent, zero_theta


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Theta recovery summary."""

    true_theta: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_std: tuple[float, ...]
    per_feature_error: tuple[float, ...]
    l2_error: float
    within_two_std: tuple[bool, ...]
    num_observations: int
    elbo_history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PredictiveResult:
    """Held-out posterior predictive comparison."""

    posterior_log_likelihood: float
    baseline_log_likelihood: float
    posterior_mean_log_probability: float
    baseline_mean_log_probability: float
    num_observations: int
    posterior_samples: int
    likelihood: str = "posterior_predictive_sequential"


def collect_matched_model_observations(
    theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    num_games: int,
    seed: int = 0,
    observer_player: int = 0,
    observed_player: int = 1,
    theta_name: str | None = "matched_model",
) -> tuple[OpponentMoveObservation, ...]:
    """Generate local observations from the same hand belief used by inference."""

    feature_names = tuple(feature_names)
    if len(theta) != len(feature_names):
        raise ValueError("theta length must match feature_names length")

    contexts = collect_observations(
        observed_model=RandomOpponent(seed=seed + 1),
        observer_model=RandomOpponent(seed=seed + 2),
        num_games=num_games,
        seed=seed + 3,
        observer_player=observer_player,
        observed_player=observed_player,
        theta_name="context",
    )
    rng = random.Random(seed + 4)
    return tuple(
        _resample_observation_from_local_belief(
            observation,
            theta,
            feature_names=feature_names,
            rng=rng,
            theta_name=theta_name,
        )
        for observation in contexts
    )


def run_recovery_experiment(
    true_theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    num_games: int = 3,
    seed: int = 0,
    vi_steps: int = 300,
    learning_rate: float = 0.03,
    num_elbo_samples: int = 1,
) -> RecoveryResult:
    """Generate synthetic games and fit the VI posterior."""

    feature_names = tuple(feature_names)
    true_theta = tuple(float(value) for value in true_theta)
    observed_model = ThetaSoftmaxOpponent(
        true_theta,
        seed=seed + 1,
        feature_names=feature_names,
    )
    observations = collect_observations(
        observed_model=observed_model,
        observer_model=RandomOpponent(seed=seed + 2),
        num_games=num_games,
        seed=seed + 3,
        observer_player=0,
        observed_player=1,
        theta_name="validation",
    )
    posterior = fit_variational_posterior(
        observations,
        feature_names=feature_names,
        num_steps=vi_steps,
        learning_rate=learning_rate,
        num_elbo_samples=num_elbo_samples,
        seed=seed + 4,
    )
    errors = tuple(
        estimate - truth
        for estimate, truth in zip(posterior.mean, true_theta)
    )
    return RecoveryResult(
        true_theta=true_theta,
        posterior_mean=posterior.mean,
        posterior_std=posterior.std,
        per_feature_error=errors,
        l2_error=_l2_norm(errors),
        within_two_std=tuple(
            abs(error) <= 2.0 * std
            for error, std in zip(errors, posterior.std)
        ),
        num_observations=len(observations),
        elbo_history=posterior.elbo_history,
    )


def train_test_split(
    observations: Sequence[OpponentMoveObservation],
    *,
    train_fraction: float = 0.75,
) -> tuple[tuple[OpponentMoveObservation, ...], tuple[OpponentMoveObservation, ...]]:
    """Deterministic game-level split that preserves observation order."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(observations) < 2:
        raise ValueError("at least two observations are required")
    game_ids = tuple(dict.fromkeys(observation.game_id for observation in observations))
    if len(game_ids) < 2:
        raise ValueError("game split requires observations from at least two games")

    split_index = max(1, min(len(game_ids) - 1, int(len(game_ids) * train_fraction)))
    train_game_ids = set(game_ids[:split_index])
    train = tuple(
        observation
        for observation in observations
        if observation.game_id in train_game_ids
    )
    test = tuple(
        observation
        for observation in observations
        if observation.game_id not in train_game_ids
    )
    return train, test


def heldout_predictive_evaluation(
    observations: Sequence[OpponentMoveObservation],
    posterior_mean: Sequence[float],
    *,
    posterior_std: Sequence[float],
    posterior_samples: int = 20,
    seed: int = 0,
    baseline_theta: Sequence[float] | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> PredictiveResult:
    """Compare the VI posterior predictive against a zero-theta baseline."""

    feature_names = tuple(feature_names)
    posterior_mean = _checked_theta(
        posterior_mean,
        feature_names=feature_names,
        name="posterior_mean",
    )
    posterior_std = _checked_theta(
        posterior_std,
        feature_names=feature_names,
        name="posterior_std",
    )
    if any(value < 0.0 for value in posterior_std):
        raise ValueError("posterior_std cannot contain negative values")
    if posterior_samples <= 0:
        raise ValueError("posterior_samples must be positive")
    if baseline_theta is None:
        baseline_theta = zero_theta(feature_names)
    baseline_theta = _checked_theta(
        baseline_theta,
        feature_names=feature_names,
        name="baseline_theta",
    )

    theta_samples = _sample_posterior_thetas(
        posterior_mean,
        posterior_std,
        num_samples=posterior_samples,
        seed=seed,
    )
    posterior_log_likelihood = _logmeanexp(
        tuple(
            sequential_log_likelihood(
                observations,
                theta,
                feature_names=feature_names,
            )
            for theta in theta_samples
        )
    )
    baseline_log_likelihood = sequential_log_likelihood(
        observations,
        baseline_theta,
        feature_names=feature_names,
    )
    count = len(observations)
    return PredictiveResult(
        posterior_log_likelihood=posterior_log_likelihood,
        baseline_log_likelihood=baseline_log_likelihood,
        posterior_mean_log_probability=_safe_mean_log_probability(
            posterior_log_likelihood,
            count,
        ),
        baseline_mean_log_probability=_safe_mean_log_probability(
            baseline_log_likelihood,
            count,
        ),
        num_observations=count,
        posterior_samples=posterior_samples,
    )


def _checked_theta(
    theta: Sequence[float],
    *,
    feature_names: tuple[str, ...],
    name: str,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in theta)
    if len(values) != len(feature_names):
        raise ValueError(f"{name} length must match feature_names length")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _sample_posterior_thetas(
    posterior_mean: tuple[float, ...],
    posterior_std: tuple[float, ...],
    *,
    num_samples: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed)
    return tuple(
        tuple(
            rng.gauss(mean, std)
            for mean, std in zip(posterior_mean, posterior_std)
        )
        for _ in range(num_samples)
    )


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one log value is required")
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values) / len(values)
    )


def _resample_observation_from_local_belief(
    observation: OpponentMoveObservation,
    theta: Sequence[float],
    *,
    feature_names: Sequence[str],
    rng: random.Random,
    theta_name: str | None,
) -> OpponentMoveObservation:
    hand = _sample_uniform_compatible_hand(observation, rng)
    chosen_card = _sample_card_from_hand(
        hand,
        observation,
        theta,
        feature_names=feature_names,
        rng=rng,
    )
    return replace(
        observation,
        opponent_hand=hand,
        legal_cards=hand,
        chosen_card=chosen_card,
        theta_name=theta_name,
    )


def _sample_uniform_compatible_hand(
    observation: OpponentMoveObservation,
    rng: random.Random,
) -> tuple[Card, ...]:
    unknown_cards = compatible_unknown_cards(
        observation.public_state,
        observation.observer_hand,
        opponent_player=observation.player,
    )
    required_cards = known_opponent_cards(
        observation.public_state,
        observation.observer_hand,
        opponent_player=observation.player,
    )
    hand_size = observation.public_state.hand_sizes[observation.player]
    if hand_size < 0:
        raise ValueError("hand size cannot be negative")
    if hand_size > len(unknown_cards):
        raise ValueError("no compatible hands are available for this observation")
    if any(card not in unknown_cards for card in required_cards):
        raise ValueError("required public cards are not compatible with this observation")
    draw_count = hand_size - len(required_cards)
    if draw_count < 0:
        raise ValueError("too many required public cards for this hand size")
    remaining_cards = tuple(card for card in unknown_cards if card not in required_cards)
    return (*required_cards, *rng.sample(remaining_cards, draw_count))


def _sample_card_from_hand(
    hand: tuple[Card, ...],
    observation: OpponentMoveObservation,
    theta: Sequence[float],
    *,
    feature_names: Sequence[str],
    rng: random.Random,
) -> Card:
    probabilities = tuple(
        local_card_probability(
            card,
            hand,
            observation.public_state,
            theta,
            player=observation.player,
            feature_names=feature_names,
        )
        for card in hand
    )
    threshold = rng.random()
    cumulative = 0.0
    for card, probability in zip(hand, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return card
    return hand[-1]


def _safe_mean_log_probability(log_likelihood: float, count: int) -> float:
    if count == 0:
        return math.nan
    return log_likelihood / count


def _l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
