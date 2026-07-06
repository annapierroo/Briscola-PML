"""Compare validation runs across feature sets, profiles, and seeds"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    calibration_curve,
    heldout_predictive_evaluation,
    importance_sampling_reference,
    train_test_split,
)
from inference import LikelihoodMode, fit_variational_posterior  # noqa: E402
from scripts import run_validation as validation  # noqa: E402


DEFAULT_FEATURE_SETS = ("core", "compact", "trump_count", "extended")
DEFAULT_PROFILES = ("aggressive", "conservative", "greedy_points")
RUN_FIELDNAMES = (
    "feature_set",
    "profile",
    "seed",
    "theta_scale",
    "split_unit",
    "feature_count",
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
    "calibration_ece",
    "initial_elbo",
    "final_elbo",
)
SUMMARY_METRICS = (
    "theta_l2_error",
    "heldout_loglik_delta",
    "heldout_mean_logp_delta",
    "calibration_ece",
    "final_elbo",
)
REFERENCE_RUN_FIELDNAMES = ("importance_ess",)
REFERENCE_SUMMARY_METRICS = ("importance_ess",)


def main() -> None:
    args = _parse_args()
    run_specs = tuple(_iter_run_specs(args))
    rows: list[dict[str, Any]] = []

    _print_progress(f"starting comparison with {len(run_specs)} runs")
    for index, spec in enumerate(run_specs, start=1):
        _print_progress(
            f"run {index}/{len(run_specs)}: "
            f"feature_set={spec.feature_set}, profile={spec.profile}, seed={spec.seed}, "
            f"theta_scale={spec.theta_scale}, split_unit={spec.split_unit}"
        )
        row = _run_validation_case(spec, show_vi_progress=args.show_vi_progress)
        rows.append(row)
        progress_message = (
            "done: "
            f"delta={row['heldout_loglik_delta']:.3f}, "
            f"theta_l2={row['theta_l2_error']:.3f}, "
            f"ece={row['calibration_ece']:.3f}"
        )
        if "importance_ess" in row:
            progress_message += f", ess={row['importance_ess']:.1f}"
        _print_progress(progress_message)

    summary = summarize_rows(
        rows,
        include_importance_reference=not args.skip_importance_reference,
    )
    output_paths = _write_outputs(
        rows=rows,
        summary=summary,
        args=args,
    )
    _print_summary(summary)
    _print_progress(f"wrote JSON report to {output_paths['json']}")
    _print_progress(f"wrote run CSV to {output_paths['runs_csv']}")
    _print_progress(f"wrote summary CSV to {output_paths['summary_csv']}")


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    include_importance_reference: bool | None = None,
) -> list[dict[str, Any]]:
    if include_importance_reference is None:
        include_importance_reference = all("importance_ess" in row for row in rows)
    metrics = _summary_metrics(include_importance_reference)
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
        for metric in metrics:
            values = tuple(float(row[metric]) for row in group)
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


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
        choices=("simulator", "matched"),
        default="matched",
        help="whether actions come from the simulator hand or the matched local belief",
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
        "--elbo-samples",
        type=int,
        default=2,
        help="Monte Carlo samples from q(theta) per ELBO step",
    )
    parser.add_argument(
        "--importance-samples",
        type=int,
        default=1000,
        help="prior samples for the importance-sampling reference",
    )
    parser.add_argument(
        "--reference-observations",
        type=int,
        default=12,
        help="training observations used by the importance-sampling reference",
    )
    parser.add_argument(
        "--skip-importance-reference",
        action="store_true",
        help="skip the prior-sample importance reference to make comparison runs faster",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=5,
        help="number of bins for the calibration curve",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="VI steps between progress updates when --show-vi-progress is enabled",
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
    if not math.isfinite(args.theta_scale) or args.theta_scale < 0.0:
        raise ValueError("theta_scale must be a finite non-negative value")
    if args.vi_steps <= 0:
        raise ValueError("vi_steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.elbo_samples <= 0:
        raise ValueError("elbo_samples must be positive")
    if not args.skip_importance_reference and args.importance_samples <= 0:
        raise ValueError("importance_samples must be positive")
    if not args.skip_importance_reference and args.reference_observations <= 0:
        raise ValueError("reference_observations must be positive")
    if args.calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if args.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")


def _iter_run_specs(args: argparse.Namespace) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            feature_set=feature_set,
            profile=profile,
            seed=seed,
            data_source=args.data_source,
            num_games=args.num_games,
            train_fraction=args.train_fraction,
            split_unit=args.split_unit,
            theta_scale=args.theta_scale,
            vi_steps=args.vi_steps,
            learning_rate=args.learning_rate,
            elbo_samples=args.elbo_samples,
            importance_samples=args.importance_samples,
            reference_observations=args.reference_observations,
            skip_importance_reference=args.skip_importance_reference,
            calibration_bins=args.calibration_bins,
            progress_interval=args.progress_interval,
        )
        for feature_set in args.feature_sets
        for profile in args.profiles
        for seed in args.seeds
    )


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
    progress_callback = validation._print_vi_progress if show_vi_progress else None
    posterior = fit_variational_posterior(
        train,
        feature_names=feature_names,
        mode=LikelihoodMode.CONDITIONAL,
        num_steps=spec.vi_steps,
        learning_rate=spec.learning_rate,
        num_elbo_samples=spec.elbo_samples,
        seed=spec.seed + 4,
        progress_callback=progress_callback,
        progress_interval=spec.progress_interval,
    )
    predictive = heldout_predictive_evaluation(
        test,
        posterior_mean=posterior.mean,
        feature_names=feature_names,
        mode=LikelihoodMode.CONDITIONAL,
    )
    calibration = calibration_curve(
        test,
        posterior.mean,
        feature_names=feature_names,
        num_bins=spec.calibration_bins,
    )
    posterior_error = tuple(
        estimate - truth
        for estimate, truth in zip(posterior.mean, true_theta)
    )
    row = {
        "feature_set": spec.feature_set,
        "profile": spec.profile,
        "seed": spec.seed,
        "theta_scale": spec.theta_scale,
        "split_unit": spec.split_unit,
        "feature_count": len(feature_names),
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
        "calibration_ece": calibration.expected_calibration_error,
        "initial_elbo": posterior.elbo_history[0],
        "final_elbo": posterior.elbo_history[-1],
    }
    if not spec.skip_importance_reference:
        reference = importance_sampling_reference(
            train[: spec.reference_observations],
            feature_names=feature_names,
            num_samples=spec.importance_samples,
            seed=spec.seed + 5,
            mode=LikelihoodMode.CONDITIONAL,
        )
        row["importance_ess"] = reference.effective_sample_size
    return row


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
            "split_unit": args.split_unit,
            "theta_scale": args.theta_scale,
            "vi_steps": args.vi_steps,
            "learning_rate": args.learning_rate,
            "elbo_samples": args.elbo_samples,
            "importance_samples": args.importance_samples,
            "reference_observations": args.reference_observations,
            "skip_importance_reference": args.skip_importance_reference,
            "calibration_bins": args.calibration_bins,
        },
        "runs": rows,
        "summary": summary,
    }
    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    include_importance_reference = not args.skip_importance_reference
    _write_csv(runs_csv_path, rows, _run_fieldnames(include_importance_reference))
    _write_csv(
        summary_csv_path,
        summary,
        _summary_fieldnames(include_importance_reference),
    )
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
        writer.writerows(rows)


def _run_fieldnames(include_importance_reference: bool) -> tuple[str, ...]:
    if not include_importance_reference:
        return RUN_FIELDNAMES
    insertion_index = RUN_FIELDNAMES.index("initial_elbo")
    return (
        RUN_FIELDNAMES[:insertion_index]
        + REFERENCE_RUN_FIELDNAMES
        + RUN_FIELDNAMES[insertion_index:]
    )


def _summary_metrics(include_importance_reference: bool) -> tuple[str, ...]:
    if not include_importance_reference:
        return SUMMARY_METRICS
    insertion_index = SUMMARY_METRICS.index("final_elbo")
    return (
        SUMMARY_METRICS[:insertion_index]
        + REFERENCE_SUMMARY_METRICS
        + SUMMARY_METRICS[insertion_index:]
    )


def _summary_fieldnames(include_importance_reference: bool) -> tuple[str, ...]:
    fields = ["feature_set", "profile", "runs", "feature_count"]
    for metric in _summary_metrics(include_importance_reference):
        fields.append(f"{metric}_mean")
        fields.append(f"{metric}_std")
    return tuple(fields)


def _print_summary(summary: list[dict[str, Any]]) -> None:
    include_importance_reference = any("importance_ess_mean" in row for row in summary)
    print("Validation comparison")
    header = (
        f"{'feature_set':<12} {'profile':<14} {'runs':>4} "
        f"{'theta_l2':>9} {'loglik_delta':>13} {'mean_logp_delta':>15} "
        f"{'ece':>8}"
    )
    if include_importance_reference:
        header += f" {'ess':>8}"
    print(header)
    for row in summary:
        line = (
            f"{row['feature_set']:<12} "
            f"{row['profile']:<14} "
            f"{row['runs']:>4} "
            f"{row['theta_l2_error_mean']:>9.3f} "
            f"{row['heldout_loglik_delta_mean']:>13.3f} "
            f"{row['heldout_mean_logp_delta_mean']:>15.3f} "
            f"{row['calibration_ece_mean']:>8.3f}"
        )
        if include_importance_reference:
            line += f" {row['importance_ess_mean']:>8.1f}"
        print(line)


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
