"""Validation experiments for synthetic opponent modelling"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from collections.abc import Sequence

from experiments.episode_collection import OpponentMoveObservation, collect_observations
from game import Card
from inference import (
    LikelihoodMode,
    compatible_unknown_cards,
    fit_variational_posterior,
    local_card_probability,
    marginal_card_probability,
    marginal_log_likelihood,
)
from opponents import FEATURE_NAMES, RandomOpponent, ThetaSoftmaxOpponent, zero_theta


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Theta recovery summary"""

    true_theta: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_std: tuple[float, ...]
    per_feature_error: tuple[float, ...]
    l2_error: float
    within_two_std: tuple[bool, ...]
    num_observations: int
    elbo_history: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ImportanceSamplingResult:
    """Crude posterior reference from prior samples"""

    posterior_mean: tuple[float, ...]
    posterior_std: tuple[float, ...]
    effective_sample_size: float
    log_normalizer: float
    num_samples: int


@dataclass(frozen=True, slots=True)
class PredictiveResult:
    """Held-out log-likelihood comparison"""

    posterior_log_likelihood: float
    baseline_log_likelihood: float
    posterior_mean_log_probability: float
    baseline_mean_log_probability: float
    num_observations: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One calibration bucket"""

    lower: float
    upper: float
    count: int
    mean_probability: float
    empirical_frequency: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibration curve and ECE"""

    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    num_events: int


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
    """Generate local observations from the same hand belief used by inference"""

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
    """Generate synthetic games and fit the VI posterior"""

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
        mode=LikelihoodMode.CONDITIONAL,
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
    train_fraction: float = 0.7,
) -> tuple[tuple[OpponentMoveObservation, ...], tuple[OpponentMoveObservation, ...]]:
    """Deterministic split that preserves observation order"""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(observations) < 2:
        raise ValueError("at least two observations are required")

    split_index = max(
        1,
        min(len(observations) - 1, int(len(observations) * train_fraction)),
    )
    return tuple(observations[:split_index]), tuple(observations[split_index:])


def importance_sampling_reference(
    observations: Sequence[OpponentMoveObservation],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    num_samples: int = 500,
    prior_std: float = 1.0,
    seed: int = 0,
    mode: LikelihoodMode = LikelihoodMode.CONDITIONAL,
) -> ImportanceSamplingResult:
    """Approximate posterior moments by sampling theta from the prior"""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")

    feature_names = tuple(feature_names)
    rng = random.Random(seed)
    theta_samples = tuple(
        tuple(rng.gauss(0.0, prior_std) for _ in feature_names)
        for _ in range(num_samples)
    )
    log_weights = tuple(
        marginal_log_likelihood(
            observations,
            theta,
            feature_names=feature_names,
            mode=mode,
        )
        for theta in theta_samples
    )
    normalized_weights, log_normalizer = _normalize_log_weights(log_weights)
    mean = tuple(
        sum(weight * theta[dim] for weight, theta in zip(normalized_weights, theta_samples))
        for dim in range(len(feature_names))
    )
    variances = tuple(
        sum(
            weight * (theta[dim] - mean[dim]) ** 2
            for weight, theta in zip(normalized_weights, theta_samples)
        )
        for dim in range(len(feature_names))
    )
    ess = 1.0 / sum(weight * weight for weight in normalized_weights)

    return ImportanceSamplingResult(
        posterior_mean=mean,
        posterior_std=tuple(math.sqrt(max(variance, 0.0)) for variance in variances),
        effective_sample_size=ess,
        log_normalizer=log_normalizer,
        num_samples=num_samples,
    )


def heldout_predictive_evaluation(
    observations: Sequence[OpponentMoveObservation],
    posterior_mean: Sequence[float],
    *,
    baseline_theta: Sequence[float] | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    mode: LikelihoodMode = LikelihoodMode.ABSOLUTE,
) -> PredictiveResult:
    """Compare posterior mean against a baseline on held-out moves"""

    feature_names = tuple(feature_names)
    if baseline_theta is None:
        baseline_theta = zero_theta(feature_names)
    posterior_log_likelihood = marginal_log_likelihood(
        observations,
        posterior_mean,
        feature_names=feature_names,
        mode=mode,
    )
    baseline_log_likelihood = marginal_log_likelihood(
        observations,
        baseline_theta,
        feature_names=feature_names,
        mode=mode,
    )
    count = len(observations)
    return PredictiveResult(
        posterior_log_likelihood=posterior_log_likelihood,
        baseline_log_likelihood=baseline_log_likelihood,
        posterior_mean_log_probability=_safe_mean_log_probability(posterior_log_likelihood, count),
        baseline_mean_log_probability=_safe_mean_log_probability(baseline_log_likelihood, count),
        num_observations=count,
    )


def calibration_curve(
    observations: Sequence[OpponentMoveObservation],
    theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    num_bins: int = 10,
) -> CalibrationResult:
    """Bin absolute probabilities and empirical frequencies"""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]
    for observation in observations:
        for candidate in compatible_unknown_cards(
            observation.public_state,
            observation.observer_hand,
        ):
            probability = marginal_card_probability(
                candidate,
                observation.public_state,
                observation.observer_hand,
                theta,
                observed_player=observation.player,
                feature_names=feature_names,
                mode=LikelihoodMode.ABSOLUTE,
            )
            bin_index = min(num_bins - 1, int(probability * num_bins))
            label = int(candidate == observation.chosen_card)
            buckets[bin_index].append((probability, label))

    bins = tuple(
        _build_calibration_bin(index, values, num_bins)
        for index, values in enumerate(buckets)
    )
    num_events = sum(bucket.count for bucket in bins)
    ece = (
        sum(
            bucket.count
            * abs(bucket.mean_probability - bucket.empirical_frequency)
            for bucket in bins
        )
        / num_events
        if num_events
        else 0.0
    )
    return CalibrationResult(
        bins=bins,
        expected_calibration_error=ece,
        num_events=num_events,
    )


def _build_calibration_bin(
    index: int,
    values: Sequence[tuple[float, int]],
    num_bins: int,
) -> CalibrationBin:
    lower = index / num_bins
    upper = (index + 1) / num_bins
    if not values:
        return CalibrationBin(
            lower=lower,
            upper=upper,
            count=0,
            mean_probability=0.0,
            empirical_frequency=0.0,
        )

    count = len(values)
    return CalibrationBin(
        lower=lower,
        upper=upper,
        count=count,
        mean_probability=sum(probability for probability, _ in values) / count,
        empirical_frequency=sum(label for _, label in values) / count,
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
    )
    hand_size = observation.public_state.hand_sizes[observation.player]
    if hand_size < 0:
        raise ValueError("hand size cannot be negative")
    if hand_size > len(unknown_cards):
        raise ValueError("no compatible hands are available for this observation")
    return tuple(rng.sample(unknown_cards, hand_size))


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


def _normalize_log_weights(log_weights: Sequence[float]) -> tuple[tuple[float, ...], float]:
    max_log_weight = max(log_weights)
    if max_log_weight == -math.inf:
        raise ValueError("all importance weights are zero")
    raw_weights = tuple(math.exp(value - max_log_weight) for value in log_weights)
    normalizer = sum(raw_weights)
    return (
        tuple(weight / normalizer for weight in raw_weights),
        max_log_weight + math.log(normalizer) - math.log(len(log_weights)),
    )


def _safe_mean_log_probability(log_likelihood: float, count: int) -> float:
    if count == 0:
        return math.nan
    return log_likelihood / count


def _l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
