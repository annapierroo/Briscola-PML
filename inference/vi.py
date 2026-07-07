"""Batch variational inference for theta"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Sequence

from game import Card
from inference.beliefs import (
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
)
from inference.likelihood import LikelihoodMode
from opponents.features import FEATURE_NAMES, card_features

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    """Feature tensor for one observed move"""

    features: object
    log_hand_factor: float
    num_hands: int


@dataclass(frozen=True, slots=True)
class VariationalPosterior:
    """Mean-field Gaussian fit for theta"""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    elbo_history: tuple[float, ...]
    feature_names: tuple[str, ...]
    best_elbo: float
    best_step: int


def prepare_observations(
    observations: Sequence[object],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    mode: LikelihoodMode = LikelihoodMode.CONDITIONAL,
    dtype: object | None = None,
) -> tuple[PreparedObservation, ...]:
    """Precompute hand features once before VI"""

    torch_module = _require_torch()
    dtype = torch_module.float64 if dtype is None else dtype
    return tuple(
        _prepare_observation(
            observation,
            feature_names=feature_names,
            mode=mode,
            dtype=dtype,
        )
        for observation in observations
    )


def log_marginal_probability_torch(
    prepared: PreparedObservation,
    theta: object,
    *,
    temperature: float = 1.0,
) -> object:
    """Differentiable marginal log-probability for one move"""

    torch_module = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    features = prepared.features
    scores = torch_module.einsum("hkd,d->hk", features, theta) / temperature
    log_card_probs = torch_module.log_softmax(scores, dim=1)
    log_average = torch_module.logsumexp(log_card_probs[:, 0], dim=0) - math.log(
        prepared.num_hands
    )
    return log_average + prepared.log_hand_factor


def log_likelihood_torch(
    prepared_observations: Sequence[PreparedObservation],
    theta: object,
    *,
    temperature: float = 1.0,
) -> object:
    """Differentiable log-likelihood for a batch of observations"""

    torch_module = _require_torch()
    if not prepared_observations:
        return torch_module.zeros((), dtype=theta.dtype, device=theta.device)
    return sum(
        log_marginal_probability_torch(
            prepared,
            theta,
            temperature=temperature,
        )
        for prepared in prepared_observations
    )


def diag_gaussian_kl(
    mean: object,
    log_std: object,
    prior_mean: object,
    prior_std: object,
) -> object:
    """KL between two diagonal Gaussians"""

    _require_torch()
    std = log_std.exp()
    prior_var = prior_std.pow(2)
    return (
        torch.log(prior_std)
        - log_std
        + (std.pow(2) + (mean - prior_mean).pow(2)) / (2.0 * prior_var)
        - 0.5
    ).sum()


def fit_variational_posterior(
    observations: Sequence[object],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    mode: LikelihoodMode = LikelihoodMode.CONDITIONAL,
    num_steps: int = 500,
    learning_rate: float = 0.05,
    num_elbo_samples: int = 1,
    prior_mean: Sequence[float] | None = None,
    prior_std: float | Sequence[float] = 1.0,
    initial_std: float = 1.0,
    temperature: float = 1.0,
    seed: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    progress_interval: int = 50,
) -> VariationalPosterior:
    """Fit q(theta) = N(mean, diag(std^2)) with Adam"""

    torch_module = _require_torch()
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if num_elbo_samples <= 0:
        raise ValueError("num_elbo_samples must be positive")
    if initial_std <= 0:
        raise ValueError("initial_std must be positive")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    feature_names = tuple(feature_names)
    dim = len(feature_names)
    prepared = prepare_observations(
        observations,
        feature_names=feature_names,
        mode=mode,
        dtype=torch_module.float64,
    )
    generator = torch_module.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    mean = torch_module.zeros(dim, dtype=torch_module.float64, requires_grad=True)
    log_std = torch_module.full(
        (dim,),
        math.log(initial_std),
        dtype=torch_module.float64,
        requires_grad=True,
    )
    prior_mean_tensor = _as_tensor(
        prior_mean if prior_mean is not None else (0.0,) * dim,
        dim,
        "prior_mean",
    )
    prior_std_tensor = _as_tensor(prior_std, dim, "prior_std")
    if torch_module.any(prior_std_tensor <= 0):
        raise ValueError("prior_std must be positive")

    optimizer = torch_module.optim.Adam((mean, log_std), lr=learning_rate)
    elbo_history: list[float] = []
    best_elbo = -math.inf
    best_step = 0
    best_mean = mean.detach().clone()
    best_log_std = log_std.detach().clone()

    for step_index in range(num_steps):
        optimizer.zero_grad()
        elbo = _estimate_elbo(
            prepared,
            mean,
            log_std,
            prior_mean_tensor,
            prior_std_tensor,
            num_samples=num_elbo_samples,
            generator=generator,
            temperature=temperature,
        )
        elbo_value = float(elbo.detach())
        if elbo_value > best_elbo:
            # Keep the best iterate; stochastic ELBO can get worse near the end.
            best_elbo = elbo_value
            best_step = step_index + 1
            best_mean = mean.detach().clone()
            best_log_std = log_std.detach().clone()

        (-elbo).backward()
        optimizer.step()
        elbo_history.append(elbo_value)

        step = step_index + 1
        if progress_callback is not None and _should_report_progress(
            step,
            num_steps,
            progress_interval,
        ):
            progress_callback(step, num_steps, elbo_value)

    return VariationalPosterior(
        mean=tuple(float(value) for value in best_mean),
        std=tuple(float(value) for value in best_log_std.exp()),
        elbo_history=tuple(elbo_history),
        feature_names=feature_names,
        best_elbo=best_elbo if elbo_history else math.nan,
        best_step=best_step,
    )


def _prepare_observation(
    observation: object,
    *,
    feature_names: Sequence[str],
    mode: LikelihoodMode,
    dtype: object,
) -> PreparedObservation:
    torch_module = _require_torch()
    mode = LikelihoodMode(mode)
    unknown_cards = compatible_unknown_cards(
        observation.public_state,
        observation.observer_hand,
    )
    hand_size = observation.public_state.hand_sizes[observation.player]
    hands = compatible_hands_containing(
        unknown_cards,
        hand_size,
        observation.chosen_card,
    )
    if not hands:
        raise ValueError("observation has no compatible hidden hands")

    # compatible_hands_containing returns hands with the observed card first.
    feature_rows = [
        [
            card_features(
                card,
                hand,
                observation.public_state,
                observation.player,
                feature_names=feature_names,
            )
            for card in hand
        ]
        for hand in hands
    ]
    tensor = torch_module.tensor(feature_rows, dtype=dtype)
    log_hand_factor = 0.0
    if mode == LikelihoodMode.ABSOLUTE:
        total_count = hand_count(len(unknown_cards), hand_size)
        log_hand_factor = math.log(len(hands)) - math.log(total_count)

    return PreparedObservation(
        features=tensor,
        log_hand_factor=log_hand_factor,
        num_hands=len(hands),
    )


def _estimate_elbo(
    prepared_observations: Sequence[PreparedObservation],
    mean: object,
    log_std: object,
    prior_mean: object,
    prior_std: object,
    *,
    num_samples: int,
    generator: object,
    temperature: float,
) -> object:
    torch_module = _require_torch()
    std = log_std.exp()
    log_likelihoods = []
    for _ in range(num_samples):
        epsilon = torch_module.randn(
            mean.shape,
            generator=generator,
            dtype=mean.dtype,
            device=mean.device,
        )
        theta = mean + std * epsilon
        log_likelihoods.append(
            log_likelihood_torch(
                prepared_observations,
                theta,
                temperature=temperature,
            )
        )

    expected_log_likelihood = torch_module.stack(log_likelihoods).mean()
    kl = diag_gaussian_kl(mean, log_std, prior_mean, prior_std)
    return expected_log_likelihood - kl


def _should_report_progress(step: int, total_steps: int, interval: int) -> bool:
    return step == 1 or step == total_steps or step % interval == 0


def _as_tensor(
    value: float | Sequence[float],
    dim: int,
    name: str,
) -> object:
    torch_module = _require_torch()
    if isinstance(value, (int, float)):
        return torch_module.full((dim,), float(value), dtype=torch_module.float64)
    if len(value) != dim:
        raise ValueError(f"{name} must have length {dim}")
    return torch_module.tensor(tuple(float(item) for item in value), dtype=torch_module.float64)


def _require_torch() -> object:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for variational inference. Install it with `pip install torch`"
        )
    return torch
