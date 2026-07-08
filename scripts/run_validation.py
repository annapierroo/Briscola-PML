"""Run one synthetic validation experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    collect_observations,
    heldout_predictive_evaluation,
    train_test_split,
)
from inference import fit_variational_posterior  # noqa: E402
from opponents import (  # noqa: E402
    AGGRESSIVE_THETA,
    COMPACT_AGGRESSIVE_THETA,
    COMPACT_CONSERVATIVE_THETA,
    COMPACT_FEATURE_NAMES,
    COMPACT_GREEDY_POINTS_THETA,
    CONSERVATIVE_THETA,
    CORE_FEATURE_NAMES,
    EXTENDED_AGGRESSIVE_THETA,
    EXTENDED_CONSERVATIVE_THETA,
    EXTENDED_FEATURE_NAMES,
    EXTENDED_GREEDY_POINTS_THETA,
    GREEDY_POINTS_THETA,
    RandomOpponent,
    STYLE_AGGRESSIVE_THETA,
    STYLE_CONSERVATIVE_THETA,
    STYLE_FEATURE_NAMES,
    STYLE_GREEDY_POINTS_THETA,
    ThetaSoftmaxOpponent,
    TRUMP_COUNT_AGGRESSIVE_THETA,
    TRUMP_COUNT_CONSERVATIVE_THETA,
    TRUMP_COUNT_FEATURE_NAMES,
    TRUMP_COUNT_GREEDY_POINTS_THETA,
)

FEATURE_SETS = {
    "compact": COMPACT_FEATURE_NAMES,
    "core": CORE_FEATURE_NAMES,
    "extended": EXTENDED_FEATURE_NAMES,
    "style": STYLE_FEATURE_NAMES,
    "trump_count": TRUMP_COUNT_FEATURE_NAMES,
}

THETA_PROFILES = {
    "compact": {
        "aggressive": COMPACT_AGGRESSIVE_THETA,
        "conservative": COMPACT_CONSERVATIVE_THETA,
        "greedy_points": COMPACT_GREEDY_POINTS_THETA,
    },
    "core": {
        "aggressive": AGGRESSIVE_THETA,
        "conservative": CONSERVATIVE_THETA,
        "greedy_points": GREEDY_POINTS_THETA,
    },
    "extended": {
        "aggressive": EXTENDED_AGGRESSIVE_THETA,
        "conservative": EXTENDED_CONSERVATIVE_THETA,
        "greedy_points": EXTENDED_GREEDY_POINTS_THETA,
    },
    "style": {
        "aggressive": STYLE_AGGRESSIVE_THETA,
        "conservative": STYLE_CONSERVATIVE_THETA,
        "greedy_points": STYLE_GREEDY_POINTS_THETA,
    },
    "trump_count": {
        "aggressive": TRUMP_COUNT_AGGRESSIVE_THETA,
        "conservative": TRUMP_COUNT_CONSERVATIVE_THETA,
        "greedy_points": TRUMP_COUNT_GREEDY_POINTS_THETA,
    },
}


def main() -> None:
    args = _parse_args()
    feature_names = FEATURE_SETS[args.feature_set]
    true_theta = _scale_theta(
        THETA_PROFILES[args.feature_set][args.profile],
        args.theta_scale,
    )

    _print_progress(
        "starting validation "
        f"(feature_set={args.feature_set}, profile={args.profile}, "
        f"data_source={args.data_source}, games={args.num_games}, "
        f"theta_scale={args.theta_scale}, split_unit={args.split_unit}, "
        f"seed={args.seed})"
    )
    _print_progress("collecting observations...")
    observations = _collect_validation_observations(args, true_theta, feature_names)
    _print_progress(f"collected {len(observations)} observations")

    train, test = train_test_split(
        observations,
        train_fraction=args.train_fraction,
        split_unit=args.split_unit,
    )
    _print_progress(f"split data: {len(train)} train, {len(test)} test")

    _print_progress(
        f"fitting variational posterior "
        f"({args.vi_steps} steps, {args.elbo_samples} ELBO samples/step)..."
    )
    posterior = fit_variational_posterior(
        train,
        feature_names=feature_names,
        num_steps=args.vi_steps,
        learning_rate=args.learning_rate,
        num_elbo_samples=args.elbo_samples,
        prior_std=args.prior_std,
        seed=args.seed + 4,
        progress_callback=_print_vi_progress,
        progress_interval=args.progress_interval,
    )
    _print_progress("evaluating held-out predictions...")
    predictive = heldout_predictive_evaluation(
        test,
        posterior_mean=posterior.mean,
        feature_names=feature_names,
    )

    _print_progress("writing report...")
    report = _build_report(
        args=args,
        feature_names=feature_names,
        true_theta=true_theta,
        observations_count=len(observations),
        train_count=len(train),
        test_count=len(test),
        posterior=posterior,
        predictive=predictive,
    )
    _write_report(report, args.output)
    _print_progress("done")
    _print_report(report, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic Briscola opponent-model validation",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="style",
        help="feature set used by the synthetic theta and inference model",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(THETA_PROFILES["core"]),
        default="greedy_points",
        help="synthetic theta profile used by the observed opponent",
    )
    parser.add_argument(
        "--data-source",
        choices=("simulator",),
        default="simulator",
        help="source of synthetic sequential games",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=4,
        help="number of synthetic games to generate",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.75,
        help="fraction of observations used for VI",
    )
    parser.add_argument(
        "--split-unit",
        choices=("game", "observation"),
        default="game",
        help="whether train/test split is done by full games or individual observations",
    )
    parser.add_argument(
        "--theta-scale",
        type=float,
        default=1.0,
        help="multiplier applied to the synthetic theta before generating data",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=300,
        help="number of Adam steps for variational inference",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=1.0,
        help="standard deviation of the zero-mean Gaussian prior over theta",
    )
    parser.add_argument(
        "--elbo-samples",
        type=int,
        default=2,
        help="Monte Carlo samples from q(theta) per ELBO step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base random seed",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="VI steps between progress updates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "validation_report.json",
        help="where to write the JSON report",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_games <= 0:
        raise ValueError("num_games must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not math.isfinite(args.theta_scale) or args.theta_scale < 0.0:
        raise ValueError("theta_scale must be a finite non-negative value")
    if args.vi_steps <= 0:
        raise ValueError("vi_steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not math.isfinite(args.prior_std) or args.prior_std <= 0:
        raise ValueError("prior_std must be a finite positive value")
    if args.elbo_samples <= 0:
        raise ValueError("elbo_samples must be positive")
    if args.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")


def _collect_validation_observations(
    args: argparse.Namespace,
    true_theta: tuple[float, ...],
    feature_names: tuple[str, ...],
) -> tuple:
    return collect_observations(
        observed_model=ThetaSoftmaxOpponent(
            true_theta,
            seed=args.seed + 1,
            feature_names=feature_names,
        ),
        observer_model=RandomOpponent(seed=args.seed + 2),
        num_games=args.num_games,
        seed=args.seed + 3,
        observer_player=0,
        observed_player=1,
        theta_name=f"{args.feature_set}_{args.profile}",
    )


def _build_report(
    *,
    args: argparse.Namespace,
    feature_names: tuple[str, ...],
    true_theta: tuple[float, ...],
    observations_count: int,
    train_count: int,
    test_count: int,
    posterior: Any,
    predictive: Any,
) -> dict[str, Any]:
    posterior_error = tuple(
        estimate - truth
        for estimate, truth in zip(posterior.mean, true_theta)
    )
    return {
        "config": {
            "feature_set": args.feature_set,
            "profile": args.profile,
            "data_source": args.data_source,
            "feature_names": feature_names,
            "num_games": args.num_games,
            "train_fraction": args.train_fraction,
            "split_unit": args.split_unit,
            "theta_scale": args.theta_scale,
            "training_likelihood": "sequential",
            "vi_steps": args.vi_steps,
            "learning_rate": args.learning_rate,
            "prior_std": args.prior_std,
            "elbo_samples": args.elbo_samples,
            "seed": args.seed,
        },
        "data": {
            "observations": observations_count,
            "train_observations": train_count,
            "test_observations": test_count,
        },
        "theta": {
            "true": true_theta,
            "posterior_mean": posterior.mean,
            "posterior_std": posterior.std,
            "posterior_error": posterior_error,
            "posterior_l2_error": _l2_norm(posterior_error),
        },
        "vi": {
            "initial_elbo": posterior.elbo_history[0],
            "final_elbo": posterior.elbo_history[-1],
            "best_elbo": posterior.best_elbo,
            "best_step": posterior.best_step,
            "elbo_history": posterior.elbo_history,
        },
        "heldout": asdict(predictive),
    }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(report), indent=2) + "\n",
        encoding="utf-8",
    )


def _print_progress(message: str) -> None:
    print(f"[run_validation] {message}", file=sys.stderr, flush=True)


def _print_vi_progress(step: int, total_steps: int, elbo: float) -> None:
    _print_progress(f"VI step {step}/{total_steps}: ELBO={elbo:.3f}")


def _print_report(report: dict[str, Any], output: Path) -> None:
    config = report["config"]
    data = report["data"]
    theta = report["theta"]
    vi = report["vi"]
    heldout = report["heldout"]

    print("Synthetic validation")
    print(f"feature set: {config['feature_set']}")
    print(f"profile: {config['profile']}")
    print(f"data source: {config['data_source']}")
    print(f"theta scale: {config['theta_scale']}")
    print(f"split unit: {config['split_unit']}")
    print("training likelihood: sequential")
    print(f"prior std: {config['prior_std']}")
    print(
        f"observations: {data['observations']} total, "
        f"{data['train_observations']} train, "
        f"{data['test_observations']} test"
    )
    print()
    print(f"{'feature':<28} {'true':>9} {'mean':>9} {'std':>9} {'error':>9}")
    for name, truth, mean, std, error in zip(
        config["feature_names"],
        theta["true"],
        theta["posterior_mean"],
        theta["posterior_std"],
        theta["posterior_error"],
    ):
        print(f"{name:<28} {truth:>9.3f} {mean:>9.3f} {std:>9.3f} {error:>9.3f}")
    print()
    print(f"posterior L2 error: {theta['posterior_l2_error']:.3f}")
    print(f"ELBO: {vi['initial_elbo']:.3f} -> {vi['final_elbo']:.3f}")
    print(f"best ELBO: {vi['best_elbo']:.3f} at step {vi['best_step']}")
    print()
    print(f"held-out loglik posterior: {heldout['posterior_log_likelihood']:.3f}")
    print(f"held-out loglik baseline:  {heldout['baseline_log_likelihood']:.3f}")
    print(f"mean logp posterior:       {heldout['posterior_mean_log_probability']:.3f}")
    print(f"mean logp baseline:        {heldout['baseline_mean_log_probability']:.3f}")
    print()
    print(f"report written to: {output}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _l2_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _scale_theta(theta: tuple[float, ...], scale: float) -> tuple[float, ...]:
    return tuple(scale * value for value in theta)


if __name__ == "__main__":
    main()
