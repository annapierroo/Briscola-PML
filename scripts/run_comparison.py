"""Compare validation runs across feature sets, profiles, and seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    heldout_predictive_evaluation,
    train_test_split,
)
from inference import fit_variational_posterior  # noqa: E402
from scripts import run_validation as validation  # noqa: E402


DEFAULT_FEATURE_SETS = ("style", "core", "compact", "trump_count", "extended")
DEFAULT_PROFILES = ("aggressive", "conservative", "greedy_points")
RUN_FIELDNAMES = (
    "feature_set",
    "profile",
    "seed",
    "theta_scale",
    "split_unit",
    "prior_std",
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
    "initial_elbo",
    "final_elbo",
    "best_elbo",
    "best_step",
)
SUMMARY_METRICS = (
    "theta_l2_error",
    "heldout_loglik_delta",
    "heldout_mean_logp_delta",
    "final_elbo",
    "best_elbo",
)


def main() -> None:
    args = _parse_args()
    run_specs = tuple(_iter_run_specs(args))
    runs_csv_path = args.output_dir / f"{args.output_prefix}_runs.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_incremental_csv(runs_csv_path, RUN_FIELDNAMES)

    _print_progress(f"starting comparison with {len(run_specs)} runs, jobs={args.jobs}")
    rows = _run_cases(
        run_specs,
        args,
        incremental_runs_csv_path=runs_csv_path,
    )

    summary = summarize_rows(rows)
    output_paths = _write_outputs(
        rows=rows,
        summary=summary,
        args=args,
    )
    _print_summary(summary)
    _print_progress(f"wrote JSON report to {output_paths['json']}")
    _print_progress(f"wrote run CSV to {output_paths['runs_csv']}")
    _print_progress(f"wrote summary CSV to {output_paths['summary_csv']}")


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


def _finite_metric_values(rows: list[dict[str, Any]], metric: str) -> tuple[float, ...]:
    return tuple(
        value
        for row in rows
        for value in (float(row[metric]),)
        if math.isfinite(value)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare synthetic validation runs across configurations",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(validation.FEATURE_SETS),
        default=list(DEFAULT_FEATURE_SETS),
        help="feature sets to compare",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(validation.THETA_PROFILES["core"]),
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
    parser.add_argument(
        "--data-source",
        choices=("simulator",),
        default="simulator",
        help="source of synthetic sequential games",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=20,
        help="number of synthetic games per run",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.75,
        help="fraction of observations used for VI",
    )
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
        "--split-unit",
        choices=("game", "observation"),
        default="game",
        help="whether train/test split is done by full games or individual observations",
    )
    parser.add_argument(
        "--theta-scale",
        type=float,
        default=1.0,
        help="multiplier applied to each synthetic theta before generating data",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=1000,
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
        "--progress-interval",
        type=int,
        default=50,
        help="VI steps between progress updates when --show-vi-progress is enabled",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel worker processes for independent runs",
    )
    parser.add_argument(
        "--show-vi-progress",
        action="store_true",
        help="print per-run VI progress",
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
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_games <= 0:
        raise ValueError("num_games must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if args.max_train_observations < 0:
        raise ValueError("max_train_observations cannot be negative")
    if args.max_test_observations < 0:
        raise ValueError("max_test_observations cannot be negative")
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
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if args.jobs > 1 and args.show_vi_progress:
        raise ValueError("--show-vi-progress is only supported with --jobs 1")


def _iter_run_specs(args: argparse.Namespace) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            feature_set=feature_set,
            profile=profile,
            seed=seed,
            data_source=args.data_source,
            num_games=args.num_games,
            train_fraction=args.train_fraction,
            max_train_observations=args.max_train_observations,
            max_test_observations=args.max_test_observations,
            split_unit=args.split_unit,
            theta_scale=args.theta_scale,
            vi_steps=args.vi_steps,
            learning_rate=args.learning_rate,
            prior_std=args.prior_std,
            elbo_samples=args.elbo_samples,
            progress_interval=args.progress_interval,
        )
        for feature_set in args.feature_sets
        for profile in args.profiles
        for seed in args.seeds
    )


def _format_spec(spec: SimpleNamespace) -> str:
    return (
        f"feature_set={spec.feature_set}, profile={spec.profile}, seed={spec.seed}, "
        f"theta_scale={spec.theta_scale}, split_unit={spec.split_unit}"
    )


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"delta={row['heldout_loglik_delta']:.3f}, "
        f"mean_delta={row['heldout_mean_logp_delta']:.3f}, "
        f"theta_l2={row['theta_l2_error']:.3f}"
    )


def _json_cell(value: Any) -> str:
    return json.dumps(_json_ready(value), separators=(",", ":"))


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


def _limit_observations(
    observations: tuple[Any, ...],
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0 or len(observations) <= limit:
        return observations
    return observations[:limit]


def _run_cases(
    run_specs: tuple[SimpleNamespace, ...],
    args: argparse.Namespace,
    *,
    incremental_runs_csv_path: Path,
) -> list[dict[str, Any]]:
    if args.jobs == 1:
        rows: list[dict[str, Any]] = []
        for index, spec in enumerate(run_specs, start=1):
            _print_progress(f"run {index}/{len(run_specs)}: {_format_spec(spec)}")
            row = _run_validation_case(spec, show_vi_progress=args.show_vi_progress)
            rows.append(row)
            _append_csv_row(incremental_runs_csv_path, row, RUN_FIELDNAMES)
            _print_progress(f"run {index}/{len(run_specs)} done: {_format_row(row)}")
        return rows

    rows_by_index: dict[int, dict[str, Any]] = {}
    max_workers = min(args.jobs, len(run_specs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(_run_validation_case, spec, show_vi_progress=False): (index, spec)
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
                f"run {index}/{len(run_specs)} done "
                f"({completed}/{len(run_specs)} completed): {_format_row(row)}"
            )
    return [rows_by_index[index] for index in range(1, len(run_specs) + 1)]


def _run_validation_case(
    spec: SimpleNamespace,
    *,
    show_vi_progress: bool,
) -> dict[str, Any]:
    feature_names = validation.FEATURE_SETS[spec.feature_set]
    true_theta = validation._scale_theta(
        validation.THETA_PROFILES[spec.feature_set][spec.profile],
        spec.theta_scale,
    )

    observations = validation._collect_validation_observations(
        spec,
        true_theta,
        feature_names,
    )
    train, test = train_test_split(
        observations,
        train_fraction=spec.train_fraction,
        split_unit=spec.split_unit,
    )
    train = _limit_observations(train, spec.max_train_observations)
    test = _limit_observations(test, spec.max_test_observations)
    progress_callback = validation._print_vi_progress if show_vi_progress else None
    posterior = fit_variational_posterior(
        train,
        feature_names=feature_names,
        num_steps=spec.vi_steps,
        learning_rate=spec.learning_rate,
        num_elbo_samples=spec.elbo_samples,
        prior_std=spec.prior_std,
        seed=spec.seed + 4,
        progress_callback=progress_callback,
        progress_interval=spec.progress_interval,
    )
    predictive = heldout_predictive_evaluation(
        test,
        posterior_mean=posterior.mean,
        feature_names=feature_names,
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
        "theta_scale": spec.theta_scale,
        "split_unit": spec.split_unit,
        "prior_std": spec.prior_std,
        "feature_count": len(feature_names),
        "feature_names": _json_cell(feature_names),
        "theta_true": _json_cell(true_theta),
        "theta_posterior_mean": _json_cell(posterior.mean),
        "theta_posterior_std": _json_cell(posterior.std),
        "theta_error": _json_cell(posterior_error),
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
        "initial_elbo": posterior.elbo_history[0],
        "final_elbo": posterior.elbo_history[-1],
        "best_elbo": posterior.best_elbo,
        "best_step": posterior.best_step,
    }


def _write_outputs(
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
            "data_source": args.data_source,
            "num_games": args.num_games,
            "train_fraction": args.train_fraction,
            "max_train_observations": args.max_train_observations,
            "max_test_observations": args.max_test_observations,
            "split_unit": args.split_unit,
            "theta_scale": args.theta_scale,
            "training_likelihood": "sequential",
            "vi_steps": args.vi_steps,
            "learning_rate": args.learning_rate,
            "prior_std": args.prior_std,
            "elbo_samples": args.elbo_samples,
            "jobs": args.jobs,
        },
        "runs": rows,
        "summary": summary,
    }
    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(runs_csv_path, rows, RUN_FIELDNAMES)
    _write_csv(summary_csv_path, summary, _summary_fieldnames())
    return {
        "json": json_path,
        "runs_csv": runs_csv_path,
        "summary_csv": summary_csv_path,
    }


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
    return {field: row.get(field) for field in fieldnames}


def _summary_fieldnames() -> tuple[str, ...]:
    fields = ["feature_set", "profile", "runs", "feature_count"]
    for metric in SUMMARY_METRICS:
        fields.append(f"{metric}_mean")
        fields.append(f"{metric}_std")
    return tuple(fields)


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


def _print_progress(message: str) -> None:
    print(f"[run_comparison] {message}", file=sys.stderr, flush=True)


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
    main()
