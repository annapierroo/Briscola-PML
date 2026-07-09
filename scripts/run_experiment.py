"""Run synthetic Briscola opponent-model experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
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
    CONSERVATIVE_THETA,
    CORE_FEATURE_NAMES,
    GREEDY_POINTS_THETA,
    INTERACTIVE_AGGRESSIVE_THETA,
    INTERACTIVE_CONSERVATIVE_THETA,
    INTERACTIVE_FEATURE_NAMES,
    INTERACTIVE_GREEDY_POINTS_THETA,
    RandomOpponent,
    ThetaSoftmaxOpponent,
)


FEATURE_SETS = {
    "core": CORE_FEATURE_NAMES,
    "interactive": INTERACTIVE_FEATURE_NAMES,
}

THETA_PROFILES = {
    "core": {
        "aggressive": AGGRESSIVE_THETA,
        "conservative": CONSERVATIVE_THETA,
        "greedy_points": GREEDY_POINTS_THETA,
    },
    "interactive": {
        "aggressive": INTERACTIVE_AGGRESSIVE_THETA,
        "conservative": INTERACTIVE_CONSERVATIVE_THETA,
        "greedy_points": INTERACTIVE_GREEDY_POINTS_THETA,
    },
}

DEFAULT_FEATURE_SETS = ("core", "interactive")
DEFAULT_PROFILES = ("aggressive", "conservative", "greedy_points")
DEFAULT_TRAIN_FRACTION = 0.75
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_POSTERIOR_SAMPLES = 20

RUN_FIELDNAMES = (
    "feature_set",
    "profile",
    "seed",
    "opponent_temperature",
    "prior_std",
    "posterior_samples",
    "feature_count",
    "feature_names",
    "theta_true",
    "theta_posterior_mean",
    "theta_posterior_std",
    "theta_error",
    "observations",
    "train_observations",
    "test_observations",
    "theta_l2_error",
    "heldout_posterior_log_likelihood",
    "heldout_baseline_log_likelihood",
    "heldout_loglik_delta",
    "heldout_posterior_mean_logp",
    "heldout_baseline_mean_logp",
    "heldout_mean_logp_delta",
    "heldout_likelihood",
    "initial_elbo",
    "final_elbo",
)
JSON_CSV_FIELDS = {
    "feature_names",
    "theta_true",
    "theta_posterior_mean",
    "theta_posterior_std",
    "theta_error",
}
SUMMARY_METRICS = (
    "theta_l2_error",
    "heldout_loglik_delta",
    "heldout_mean_logp_delta",
    "final_elbo",
)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One synthetic validation configuration."""

    feature_set: str
    profile: str
    seed: int
    num_games: int
    opponent_temperature: float
    vi_steps: int
    prior_std: float
    elbo_samples: int
    posterior_samples: int
    max_train_observations: int = 0
    max_test_observations: int = 0


def main() -> None:
    args = _parse_args()
    args.func(args)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["feature_set"]), str(row["profile"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for feature_set, profile in sorted(grouped):
        group = grouped[(feature_set, profile)]
        item: dict[str, Any] = {
            "feature_set": feature_set,
            "profile": profile,
            "runs": len(group),
            "feature_count": group[0]["feature_count"],
        }
        for metric in SUMMARY_METRICS:
            values = _finite_metric_values(group, metric)
            item[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            item[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else math.nan)
            )
        summary.append(item)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic Briscola opponent-model experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single",
        help="run one validation configuration",
    )
    _add_single_args(single)
    single.set_defaults(func=_run_single)

    compare = subparsers.add_parser(
        "compare",
        help="run a comparison grid over configurations",
    )
    _add_compare_args(compare)
    compare.set_defaults(func=_run_compare)

    args = parser.parse_args()
    _validate_common_args(args)
    if args.command == "compare":
        _validate_compare_args(args)
    return args


def _add_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="core",
        help="feature set used by the synthetic theta and inference model",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(THETA_PROFILES["core"]),
        default="greedy_points",
        help="synthetic theta profile used by the observed opponent",
    )
    _add_common_experiment_args(parser, num_games_default=4, vi_steps_default=300)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base random seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "validation_report.json",
        help="where to write the JSON report",
    )


def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(FEATURE_SETS),
        default=list(DEFAULT_FEATURE_SETS),
        help="feature sets to compare",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(THETA_PROFILES["core"]),
        default=list(DEFAULT_PROFILES),
        help="synthetic theta profiles to compare",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="base random seeds to run",
    )
    _add_common_experiment_args(parser, num_games_default=20, vi_steps_default=1000)
    parser.add_argument(
        "--max-train-observations",
        type=int,
        default=0,
        help="cap training observations used by VI; 0 uses the full train split",
    )
    parser.add_argument(
        "--max-test-observations",
        type=int,
        default=0,
        help="cap held-out observations used by evaluation; 0 uses the full test split",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel worker processes for independent runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "comparison",
        help="directory where comparison outputs are written",
    )
    parser.add_argument(
        "--output-prefix",
        default="validation_comparison",
        help="basename for JSON and CSV outputs",
    )


def _add_common_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    num_games_default: int,
    vi_steps_default: int,
) -> None:
    parser.add_argument(
        "--num-games",
        type=int,
        default=num_games_default,
        help="number of synthetic games to generate",
    )
    parser.add_argument(
        "--opponent-temperature",
        type=float,
        default=1.0,
        help="softmax temperature used by the synthetic opponent and likelihood",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=vi_steps_default,
        help="number of Adam steps for variational inference",
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
        "--posterior-samples",
        type=int,
        default=DEFAULT_POSTERIOR_SAMPLES,
        help="theta samples from q(theta) for held-out posterior prediction",
    )


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.num_games < 2:
        raise ValueError("num_games must be at least 2 for the game-level split")
    if not math.isfinite(args.opponent_temperature) or args.opponent_temperature <= 0.0:
        raise ValueError("opponent_temperature must be a finite positive value")
    if args.vi_steps <= 0:
        raise ValueError("vi_steps must be positive")
    if not math.isfinite(args.prior_std) or args.prior_std <= 0:
        raise ValueError("prior_std must be a finite positive value")
    if args.elbo_samples <= 0:
        raise ValueError("elbo_samples must be positive")
    if args.posterior_samples <= 0:
        raise ValueError("posterior_samples must be positive")


def _validate_compare_args(args: argparse.Namespace) -> None:
    if args.max_train_observations < 0:
        raise ValueError("max_train_observations cannot be negative")
    if args.max_test_observations < 0:
        raise ValueError("max_test_observations cannot be negative")
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")


def _run_single(args: argparse.Namespace) -> None:
    spec = ExperimentSpec(
        feature_set=args.feature_set,
        profile=args.profile,
        seed=args.seed,
        num_games=args.num_games,
        opponent_temperature=args.opponent_temperature,
        vi_steps=args.vi_steps,
        prior_std=args.prior_std,
        elbo_samples=args.elbo_samples,
        posterior_samples=args.posterior_samples,
    )
    _print_progress(
        "single",
        "starting validation "
        f"(feature_set={spec.feature_set}, profile={spec.profile}, "
        f"games={spec.num_games}, "
        f"opponent_temperature={spec.opponent_temperature}, "
        f"posterior_samples={spec.posterior_samples}, seed={spec.seed})",
    )
    row = _run_case(spec, progress_label="single")
    report = _build_single_report(spec, row)
    _write_json(report, args.output)
    _print_progress("single", "done")
    _print_single_report(report, args.output)


def _run_compare(args: argparse.Namespace) -> None:
    run_specs = tuple(_iter_compare_specs(args))
    runs_csv_path = args.output_dir / f"{args.output_prefix}_runs.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_incremental_csv(runs_csv_path, RUN_FIELDNAMES)

    _print_progress("compare", f"starting comparison with {len(run_specs)} runs, jobs={args.jobs}")
    rows = _run_cases(
        run_specs,
        args,
        incremental_runs_csv_path=runs_csv_path,
    )

    summary = summarize_rows(rows)
    output_paths = _write_compare_outputs(
        rows=rows,
        summary=summary,
        args=args,
    )
    _print_summary(summary)
    _print_progress("compare", f"wrote JSON report to {output_paths['json']}")
    _print_progress("compare", f"wrote run CSV to {output_paths['runs_csv']}")
    _print_progress("compare", f"wrote summary CSV to {output_paths['summary_csv']}")


def _iter_compare_specs(args: argparse.Namespace) -> tuple[ExperimentSpec, ...]:
    return tuple(
        ExperimentSpec(
            feature_set=feature_set,
            profile=profile,
            seed=seed,
            num_games=args.num_games,
            opponent_temperature=args.opponent_temperature,
            vi_steps=args.vi_steps,
            prior_std=args.prior_std,
            elbo_samples=args.elbo_samples,
            posterior_samples=args.posterior_samples,
            max_train_observations=args.max_train_observations,
            max_test_observations=args.max_test_observations,
        )
        for feature_set in args.feature_sets
        for profile in args.profiles
        for seed in args.seeds
    )


def _run_case(
    spec: ExperimentSpec,
    *,
    progress_label: str | None = None,
) -> dict[str, Any]:
    feature_names = FEATURE_SETS[spec.feature_set]
    true_theta = THETA_PROFILES[spec.feature_set][spec.profile]

    if progress_label is not None:
        _print_progress(progress_label, "collecting observations...")
    observations = _collect_observations(spec, true_theta, feature_names)

    if progress_label is not None:
        _print_progress(progress_label, f"collected {len(observations)} observations")
    train, test = train_test_split(
        observations,
        train_fraction=DEFAULT_TRAIN_FRACTION,
    )
    train = _limit_observations(train, spec.max_train_observations)
    test = _limit_observations(test, spec.max_test_observations)

    if progress_label is not None:
        _print_progress(progress_label, f"split data: {len(train)} train, {len(test)} test")
        _print_progress(
            progress_label,
            f"fitting variational posterior "
            f"({spec.vi_steps} steps, {spec.elbo_samples} ELBO samples/step)...",
        )
    posterior = fit_variational_posterior(
        train,
        feature_names=feature_names,
        num_steps=spec.vi_steps,
        learning_rate=DEFAULT_LEARNING_RATE,
        num_elbo_samples=spec.elbo_samples,
        prior_std=spec.prior_std,
        temperature=spec.opponent_temperature,
        seed=spec.seed + 4,
    )

    if progress_label is not None:
        _print_progress(progress_label, "evaluating held-out predictions...")
    predictive = heldout_predictive_evaluation(
        test,
        posterior_mean=posterior.mean,
        posterior_std=posterior.std,
        posterior_samples=spec.posterior_samples,
        seed=spec.seed + 5,
        feature_names=feature_names,
        temperature=spec.opponent_temperature,
    )
    posterior_error = tuple(
        estimate - truth
        for estimate, truth in zip(posterior.mean, true_theta)
    )
    theta_rows = _theta_rows(
        feature_names,
        true_theta,
        posterior.mean,
        posterior.std,
    )
    return {
        "feature_set": spec.feature_set,
        "profile": spec.profile,
        "seed": spec.seed,
        "opponent_temperature": spec.opponent_temperature,
        "prior_std": spec.prior_std,
        "posterior_samples": predictive.posterior_samples,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "theta_true": true_theta,
        "theta_posterior_mean": posterior.mean,
        "theta_posterior_std": posterior.std,
        "theta_error": posterior_error,
        "theta": theta_rows,
        "observations": len(observations),
        "train_observations": len(train),
        "test_observations": len(test),
        "theta_l2_error": _l2_norm(posterior_error),
        "heldout_posterior_log_likelihood": predictive.posterior_log_likelihood,
        "heldout_baseline_log_likelihood": predictive.baseline_log_likelihood,
        "heldout_loglik_delta": (
            predictive.posterior_log_likelihood
            - predictive.baseline_log_likelihood
        ),
        "heldout_posterior_mean_logp": predictive.posterior_mean_log_probability,
        "heldout_baseline_mean_logp": predictive.baseline_mean_log_probability,
        "heldout_mean_logp_delta": (
            predictive.posterior_mean_log_probability
            - predictive.baseline_mean_log_probability
        ),
        "heldout_likelihood": predictive.likelihood,
        "initial_elbo": posterior.elbo_history[0],
        "final_elbo": posterior.elbo_history[-1],
        "elbo_history": posterior.elbo_history,
    }


def _collect_observations(
    spec: ExperimentSpec,
    true_theta: tuple[float, ...],
    feature_names: tuple[str, ...],
) -> tuple[Any, ...]:
    return collect_observations(
        observed_model=ThetaSoftmaxOpponent(
            true_theta,
            seed=spec.seed + 1,
            feature_names=feature_names,
            temperature=spec.opponent_temperature,
        ),
        observer_model=RandomOpponent(seed=spec.seed + 2),
        num_games=spec.num_games,
        seed=spec.seed + 3,
        observer_player=0,
        observed_player=1,
        theta_name=f"{spec.feature_set}_{spec.profile}",
    )


def _build_single_report(spec: ExperimentSpec, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": {
            "feature_set": spec.feature_set,
            "profile": spec.profile,
            "data_source": "simulator",
            "feature_names": row["feature_names"],
            "num_games": spec.num_games,
            "train_fraction": DEFAULT_TRAIN_FRACTION,
            "opponent_temperature": spec.opponent_temperature,
            "training_likelihood": "sequential",
            "vi_steps": spec.vi_steps,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "prior_std": spec.prior_std,
            "elbo_samples": spec.elbo_samples,
            "posterior_samples": spec.posterior_samples,
            "seed": spec.seed,
        },
        "data": {
            "observations": row["observations"],
            "train_observations": row["train_observations"],
            "test_observations": row["test_observations"],
        },
        "theta": {
            "true": row["theta_true"],
            "posterior_mean": row["theta_posterior_mean"],
            "posterior_std": row["theta_posterior_std"],
            "posterior_error": row["theta_error"],
            "posterior_l2_error": row["theta_l2_error"],
        },
        "vi": {
            "initial_elbo": row["initial_elbo"],
            "final_elbo": row["final_elbo"],
            "elbo_history": row["elbo_history"],
        },
        "heldout": {
            "posterior_log_likelihood": row["heldout_posterior_log_likelihood"],
            "baseline_log_likelihood": row["heldout_baseline_log_likelihood"],
            "posterior_mean_log_probability": row["heldout_posterior_mean_logp"],
            "baseline_mean_log_probability": row["heldout_baseline_mean_logp"],
            "num_observations": row["test_observations"],
            "posterior_samples": row["posterior_samples"],
            "likelihood": row["heldout_likelihood"],
        },
    }


def _run_cases(
    run_specs: tuple[ExperimentSpec, ...],
    args: argparse.Namespace,
    *,
    incremental_runs_csv_path: Path,
) -> list[dict[str, Any]]:
    if args.jobs == 1:
        rows: list[dict[str, Any]] = []
        for index, spec in enumerate(run_specs, start=1):
            _print_progress("compare", f"run {index}/{len(run_specs)}: {_format_spec(spec)}")
            row = _run_case(spec)
            rows.append(row)
            _append_csv_row(incremental_runs_csv_path, row, RUN_FIELDNAMES)
            _print_progress("compare", f"run {index}/{len(run_specs)} done: {_format_row(row)}")
        return rows

    rows_by_index: dict[int, dict[str, Any]] = {}
    max_workers = min(args.jobs, len(run_specs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(_run_case, spec): (index, spec)
            for index, spec in enumerate(run_specs, start=1)
        }
        for completed, future in enumerate(as_completed(future_to_run), start=1):
            index, spec = future_to_run[future]
            try:
                row = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"run {index}/{len(run_specs)} failed: {_format_spec(spec)}"
                ) from exc
            rows_by_index[index] = row
            _append_csv_row(incremental_runs_csv_path, row, RUN_FIELDNAMES)
            _print_progress(
                "compare",
                f"run {index}/{len(run_specs)} done "
                f"({completed}/{len(run_specs)} completed): {_format_row(row)}",
            )
    return [rows_by_index[index] for index in range(1, len(run_specs) + 1)]


def _write_compare_outputs(
    *,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_prefix}.json"
    runs_csv_path = args.output_dir / f"{args.output_prefix}_runs.csv"
    summary_csv_path = args.output_dir / f"{args.output_prefix}_summary.csv"

    payload = {
        "config": {
            "feature_sets": args.feature_sets,
            "profiles": args.profiles,
            "seeds": args.seeds,
            "data_source": "simulator",
            "num_games": args.num_games,
            "train_fraction": DEFAULT_TRAIN_FRACTION,
            "max_train_observations": args.max_train_observations,
            "max_test_observations": args.max_test_observations,
            "opponent_temperature": args.opponent_temperature,
            "training_likelihood": "sequential",
            "vi_steps": args.vi_steps,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "prior_std": args.prior_std,
            "elbo_samples": args.elbo_samples,
            "posterior_samples": args.posterior_samples,
            "jobs": args.jobs,
        },
        "runs": rows,
        "summary": summary,
    }
    _write_json(payload, json_path)
    _write_csv(runs_csv_path, rows, RUN_FIELDNAMES)
    _write_csv(summary_csv_path, summary, _summary_fieldnames())
    return {
        "json": json_path,
        "runs_csv": runs_csv_path,
        "summary_csv": summary_csv_path,
    }


def _write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_row(row, fieldnames) for row in rows)


def _prepare_incremental_csv(path: Path, fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(
    path: Path,
    row: dict[str, Any],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(_csv_row(row, fieldnames))


def _csv_row(row: dict[str, Any], fieldnames: tuple[str, ...]) -> dict[str, Any]:
    return {
        field: _json_cell(row.get(field)) if field in JSON_CSV_FIELDS else row.get(field)
        for field in fieldnames
    }


def _summary_fieldnames() -> tuple[str, ...]:
    fields = ["feature_set", "profile", "runs", "feature_count"]
    for metric in SUMMARY_METRICS:
        fields.append(f"{metric}_mean")
        fields.append(f"{metric}_std")
    return tuple(fields)


def _finite_metric_values(rows: list[dict[str, Any]], metric: str) -> tuple[float, ...]:
    return tuple(
        value
        for row in rows
        for value in (float(row[metric]),)
        if math.isfinite(value)
    )


def _limit_observations(
    observations: tuple[Any, ...],
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0 or len(observations) <= limit:
        return observations
    return observations[:limit]


def _theta_rows(
    feature_names: tuple[str, ...],
    true_theta: tuple[float, ...],
    posterior_mean: tuple[float, ...],
    posterior_std: tuple[float, ...],
) -> list[dict[str, float | str]]:
    return [
        {
            "feature": feature,
            "true": truth,
            "posterior_mean": mean,
            "posterior_std": std,
            "error": mean - truth,
        }
        for feature, truth, mean, std in zip(
            feature_names,
            true_theta,
            posterior_mean,
            posterior_std,
        )
    ]


def _format_spec(spec: ExperimentSpec) -> str:
    return (
        f"feature_set={spec.feature_set}, profile={spec.profile}, seed={spec.seed}, "
        f"opponent_temperature={spec.opponent_temperature}"
    )


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"delta={row['heldout_loglik_delta']:.3f}, "
        f"mean_delta={row['heldout_mean_logp_delta']:.3f}, "
        f"theta_l2={row['theta_l2_error']:.3f}"
    )


def _print_single_report(report: dict[str, Any], output: Path) -> None:
    config = report["config"]
    data = report["data"]
    theta = report["theta"]
    vi = report["vi"]
    heldout = report["heldout"]

    print("Synthetic validation")
    print(f"feature set: {config['feature_set']}")
    print(f"profile: {config['profile']}")
    print(f"data source: {config['data_source']}")
    print(f"opponent temperature: {config['opponent_temperature']}")
    print("split: game-level 75/25")
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
    print()
    print(f"posterior samples: {heldout['posterior_samples']}")
    print(
        "held-out loglik posterior predictive: "
        f"{heldout['posterior_log_likelihood']:.3f}"
    )
    print(f"held-out loglik baseline:             {heldout['baseline_log_likelihood']:.3f}")
    print(
        "mean logp posterior predictive:       "
        f"{heldout['posterior_mean_log_probability']:.3f}"
    )
    print(f"mean logp baseline:                   {heldout['baseline_mean_log_probability']:.3f}")
    print()
    print(f"report written to: {output}")


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("Validation comparison")
    print(
        f"{'feature_set':<12} {'profile':<14} {'runs':>4} "
        f"{'theta_l2':>9} {'loglik_delta':>13} {'mean_logp_delta':>15}"
    )
    for row in summary:
        print(
            f"{row['feature_set']:<12} "
            f"{row['profile']:<14} "
            f"{row['runs']:>4} "
            f"{row['theta_l2_error_mean']:>9.3f} "
            f"{row['heldout_loglik_delta_mean']:>13.3f} "
            f"{row['heldout_mean_logp_delta_mean']:>15.3f}"
        )


def _print_progress(command: str, message: str) -> None:
    print(f"[run_experiment:{command}] {message}", file=sys.stderr, flush=True)


def _json_cell(value: Any) -> str:
    return json.dumps(_json_ready(value), separators=(",", ":"))


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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
