"""Batch variational inference for theta"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Sequence

from game import Card, full_deck
from inference.beliefs import (
    compatible_hands,
    compatible_hands_containing,
    compatible_unknown_cards,
    hand_count,
    known_opponent_cards,
)
from inference.likelihood import LikelihoodMode
from opponents.features import FEATURE_NAMES, card_features

try:
    import torch
except ModuleNotFoundError:
    torch = None

_DECK_INDEX = {card: index for index, card in enumerate(full_deck())}


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    """Feature tensor for one observed move"""

    features: object
    log_hand_factor: float
    num_hands: int


@dataclass(frozen=True, slots=True)
class PreparedSequentialObservation:
    """Precomputed tensors for one filtering step."""

    num_hands: int
    selected_indices: object
    selected_features: object
    selected_positions: object
    transition_sources: object | None
    transition_targets: object | None
    transition_weights: object | None
    next_num_hands: int | None


@dataclass(frozen=True, slots=True)
class PreparedSequentialGame:
    """Precomputed filtering graph for one game."""

    observations: tuple[PreparedSequentialObservation, ...]


@dataclass(frozen=True, slots=True)
class VariationalPosterior:
    """Mean-field Gaussian fit for theta"""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    elbo_history: tuple[float, ...]
    feature_names: tuple[str, ...]
    best_elbo: float
    best_step: int
    training_likelihood: str = "sequential"


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


def prepare_sequential_games(
    observations: Sequence[object],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    dtype: object | None = None,
) -> tuple[PreparedSequentialGame, ...]:
    """Precompute exact hand-filter graphs for ordered game observations."""

    torch_module = _require_torch()
    dtype = torch_module.float64 if dtype is None else dtype
    return tuple(
        _prepare_sequential_game(
            group,
            feature_names=feature_names,
            dtype=dtype,
        )
        for group in _group_by_game(observations)
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


def log_sequential_game_probability_torch(
    prepared: PreparedSequentialGame,
    theta: object,
    *,
    temperature: float = 1.0,
) -> object:
    """Differentiable log-probability from the sequential hand filter."""

    torch_module = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not prepared.observations:
        return torch_module.zeros((), dtype=theta.dtype, device=theta.device)

    first = prepared.observations[0]
    belief = torch_module.full(
        (first.num_hands,),
        1.0 / first.num_hands,
        dtype=theta.dtype,
        device=theta.device,
    )
    total = torch_module.zeros((), dtype=theta.dtype, device=theta.device)
    tiny = torch_module.finfo(theta.dtype).tiny

    for step in prepared.observations:
        selected_indices = step.selected_indices.to(device=theta.device)
        selected_positions = step.selected_positions.to(device=theta.device)
        features = step.selected_features.to(dtype=theta.dtype, device=theta.device)

        scores = torch_module.einsum("hkd,d->hk", features, theta) / temperature
        log_card_probs = torch_module.log_softmax(scores, dim=1)
        rows = torch_module.arange(
            selected_positions.shape[0],
            dtype=torch_module.long,
            device=theta.device,
        )
        action_probs = log_card_probs[rows, selected_positions].exp()
        weighted = belief[selected_indices] * action_probs
        probability = weighted.sum().clamp_min(tiny)
        total = total + torch_module.log(probability)

        if step.transition_sources is None:
            continue

        posterior = weighted / probability
        transition_sources = step.transition_sources.to(device=theta.device)
        transition_targets = step.transition_targets.to(device=theta.device)
        transition_weights = step.transition_weights.to(
            dtype=theta.dtype,
            device=theta.device,
        )
        next_belief = torch_module.zeros(
            (step.next_num_hands,),
            dtype=theta.dtype,
            device=theta.device,
        )
        next_belief = next_belief.index_add(
            0,
            transition_targets,
            posterior[transition_sources] * transition_weights,
        )
        belief = next_belief / next_belief.sum().clamp_min(tiny)

    return total


def log_sequential_likelihood_torch(
    prepared_games: Sequence[PreparedSequentialGame],
    theta: object,
    *,
    temperature: float = 1.0,
) -> object:
    """Differentiable sequential log-likelihood for several games."""

    torch_module = _require_torch()
    if not prepared_games:
        return torch_module.zeros((), dtype=theta.dtype, device=theta.device)
    return sum(
        log_sequential_game_probability_torch(
            prepared,
            theta,
            temperature=temperature,
        )
        for prepared in prepared_games
    )


def sequential_log_likelihood(
    observations: Sequence[object],
    theta: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    temperature: float = 1.0,
) -> float:
    """Sequential held-out log-likelihood from the hand filter."""

    torch_module = _require_torch()
    feature_names = tuple(feature_names)
    if len(theta) != len(feature_names):
        raise ValueError("theta length must match feature_names length")
    prepared = prepare_sequential_games(
        observations,
        feature_names=feature_names,
        dtype=torch_module.float64,
    )
    theta_tensor = torch_module.tensor(tuple(float(value) for value in theta), dtype=torch_module.float64)
    with torch_module.no_grad():
        return float(
            log_sequential_likelihood_torch(
                prepared,
                theta_tensor,
                temperature=temperature,
            )
        )


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
    prepared = prepare_sequential_games(
        observations,
        feature_names=feature_names,
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
        training_likelihood="sequential",
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
        opponent_player=observation.player,
    )
    required_cards = known_opponent_cards(
        observation.public_state,
        observation.observer_hand,
        opponent_player=observation.player,
    )
    hand_size = observation.public_state.hand_sizes[observation.player]
    hands = compatible_hands_containing(
        unknown_cards,
        hand_size,
        observation.chosen_card,
        required_cards=required_cards,
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
        total_count = hand_count(
            len(unknown_cards),
            hand_size,
            required_count=len(required_cards),
        )
        log_hand_factor = math.log(len(hands)) - math.log(total_count)

    return PreparedObservation(
        features=tensor,
        log_hand_factor=log_hand_factor,
        num_hands=len(hands),
    )


def _prepare_sequential_game(
    observations: Sequence[object],
    *,
    feature_names: Sequence[str],
    dtype: object,
) -> PreparedSequentialGame:
    if not observations:
        raise ValueError("at least one observation is required")

    torch_module = _require_torch()
    hands_by_step = tuple(
        _compatible_hands_for_observation(observation)
        for observation in observations
    )
    hand_maps = tuple(
        {hand: index for index, hand in enumerate(hands)}
        for hands in hands_by_step
    )

    prepared_steps: list[PreparedSequentialObservation] = []
    for index, observation in enumerate(observations):
        hands = hands_by_step[index]
        selected = tuple(
            (hand_index, hand)
            for hand_index, hand in enumerate(hands)
            if observation.chosen_card in hand
        )
        if not selected:
            raise ValueError("observation has no compatible hidden hands")

        selected_indices = tuple(hand_index for hand_index, _ in selected)
        selected_hands = tuple(hand for _, hand in selected)
        selected_positions = tuple(
            hand.index(observation.chosen_card)
            for hand in selected_hands
        )
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
            for hand in selected_hands
        ]

        transition_sources = None
        transition_targets = None
        transition_weights = None
        next_num_hands = None
        if index + 1 < len(observations):
            sources, targets, weights = _prepare_transition_edges(
                selected_hands,
                observation,
                observations[index + 1],
                hand_maps[index + 1],
            )
            transition_sources = torch_module.tensor(
                sources,
                dtype=torch_module.long,
            )
            transition_targets = torch_module.tensor(
                targets,
                dtype=torch_module.long,
            )
            transition_weights = torch_module.tensor(weights, dtype=dtype)
            next_num_hands = len(hands_by_step[index + 1])

        prepared_steps.append(
            PreparedSequentialObservation(
                num_hands=len(hands),
                selected_indices=torch_module.tensor(
                    selected_indices,
                    dtype=torch_module.long,
                ),
                selected_features=torch_module.tensor(feature_rows, dtype=dtype),
                selected_positions=torch_module.tensor(
                    selected_positions,
                    dtype=torch_module.long,
                ),
                transition_sources=transition_sources,
                transition_targets=transition_targets,
                transition_weights=transition_weights,
                next_num_hands=next_num_hands,
            )
        )

    return PreparedSequentialGame(observations=tuple(prepared_steps))


def _compatible_hands_for_observation(observation: object) -> tuple[tuple[Card, ...], ...]:
    unknown_cards = compatible_unknown_cards(
        observation.public_state,
        observation.observer_hand,
    )
    hand_size = observation.public_state.hand_sizes[observation.player]
    hands = compatible_hands(unknown_cards, hand_size)
    if not hands:
        raise ValueError("observation has no compatible hidden hands")
    return tuple(dict.fromkeys(_canonical_hand(hand) for hand in hands))


def _prepare_transition_edges(
    selected_hands: Sequence[tuple[Card, ...]],
    current_observation: object,
    next_observation: object,
    next_hand_map: dict[tuple[Card, ...], int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    next_unknown_cards = compatible_unknown_cards(
        next_observation.public_state,
        next_observation.observer_hand,
    )
    next_hand_size = next_observation.public_state.hand_sizes[next_observation.player]
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for source_index, hand in enumerate(selected_hands):
        reduced_hand = tuple(
            card for card in hand if card != current_observation.chosen_card
        )
        missing_cards = next_hand_size - len(reduced_hand)
        if missing_cards < 0:
            continue

        available_cards = tuple(
            card for card in next_unknown_cards if card not in reduced_hand
        )
        completions = compatible_hands(available_cards, missing_cards)
        if not completions:
            continue

        share = 1.0 / len(completions)
        for completion in completions:
            next_hand = _canonical_hand((*reduced_hand, *completion))
            target_index = next_hand_map.get(next_hand)
            if target_index is None:
                continue
            sources.append(source_index)
            targets.append(target_index)
            weights.append(share)

    if not sources:
        raise ValueError("sequential transition has no compatible next hands")
    return tuple(sources), tuple(targets), tuple(weights)


def _estimate_elbo(
    prepared_observations: Sequence[PreparedSequentialGame],
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
            log_sequential_likelihood_torch(
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


def _group_by_game(observations: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    groups: list[list[object]] = []
    current_group: list[object] = []
    current_game_id: int | None = None

    for observation in observations:
        if current_game_id is None or observation.game_id != current_game_id:
            if current_group:
                groups.append(current_group)
            current_group = [observation]
            current_game_id = observation.game_id
        else:
            current_group.append(observation)

    if current_group:
        groups.append(current_group)
    return tuple(tuple(group) for group in groups)


def _canonical_hand(hand: Sequence[Card]) -> tuple[Card, ...]:
    return tuple(sorted(hand, key=lambda card: _DECK_INDEX[card]))


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
