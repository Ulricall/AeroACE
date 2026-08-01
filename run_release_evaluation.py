"""Run paper methods through the shared closed-loop evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

PAPER_MODELS = [
    "pid",
    "omac",
    "neural_fly",
    "ood_control",
    "decision_transformer",
    "aeroace",
    "dmrac",
    "neural_bem",
    "pitcn",
    "mann",
    "agile_full",
    "pi_transformer",
    "powerformer",
    "causal_transformer",
    "cluster_causal",
    "pinnsformer",
    "rtnmpc",
    "mlmpc",
]

RETAINED_CONTROLLER_COMPARISONS = [
    "aeroace_fixed_gate",
    "aeroace_vanilla_gru",
]

ALLOWED_MODELS = set(PAPER_MODELS + RETAINED_CONTROLLER_COMPARISONS)

# Keep every learned comparison on its retained release checkpoint.  The
# explicitly listed controller parameters are the release defaults selected
# before this rerun; spelling them out here makes the command record
# self-contained instead of relying on mutable parser defaults in run.py.
MODEL_ARGUMENTS = {
    "dmrac": [
        "--dmrac_load_ckpt", "params/dmrac_feature.pt",
        "--dmrac_train", "0",
        "--dmrac_test", "1",
    ],
    "neural_bem": [
        "--neuralbem_load_ckpt", "params/neuralbem.pt",
        "--neuralbem_train", "0",
        "--neuralbem_test", "1",
    ],
    "pitcn": [
        "--pitcn_load_ckpt", "params/pitcn.pt",
        "--pitcn_train", "0",
        "--pitcn_test", "1",
        "--pitcn_compensation_gain", "0.05",
        "--pitcn_force_bound", "20",
    ],
    "mann": [
        "--mann_load_ckpt", "params/mann_feature.pt",
        "--mann_train", "0",
        "--mann_test", "1",
        "--mann_gamma_adapt", "0.15",
        "--mann_w_bound", "5",
    ],
    "agile_full": [
        "--agile_load_ckpt", "params/agile_full.pt",
        "--agile_train", "0",
        "--agile_test", "1",
    ],
    "pi_transformer": [
        "--pi_transformer_load_ckpt", "params/pi_transformer.pt",
        "--pi_transformer_train", "0",
        "--pi_transformer_test", "1",
        "--pi_compensation_gain", "1",
        "--pi_force_bound", "0.25",
    ],
    "powerformer": [
        "--powerformer_load_ckpt", "params/powerformer.pt",
        "--powerformer_train", "0",
        "--powerformer_test", "1",
        "--powerformer_compensation_gain", "1",
        "--powerformer_force_bound", "0.5",
    ],
    "causal_transformer": [
        "--causal_transformer_load_ckpt", "params/causal_transformer.pt",
        "--causal_transformer_train", "0",
        "--causal_transformer_test", "1",
    ],
    "cluster_causal": [
        "--cluster_causal_load_ckpt", "params/cluster_causal.pt",
        "--cluster_causal_train", "0",
        "--cluster_causal_test", "1",
        "--cc_compensation_gain", "0.1",
        "--cc_force_bound", "200",
    ],
    "pinnsformer": [
        "--pinnsformer_load_ckpt", "params/pinnsformer.pt",
        "--pinnsformer_train", "0",
        "--pinnsformer_test", "1",
        "--pinnsformer_compensation_gain", "0.01",
        "--pinnsformer_force_bound", "200",
    ],
}

CHECKPOINT_LOAD_MARKERS = {
    "dmrac": "Loaded DMRAC feature checkpoint:",
    "neural_bem": "Loaded NeuralBEM checkpoint:",
    "pitcn": "Loaded pitcn checkpoint for control from",
    "mann": "Loaded MANN feature checkpoint:",
    "agile_full": "Loaded agile_full checkpoint from",
    "pi_transformer": "Loaded Pi-Transformer model from",
    "powerformer": "Loaded Powerformer model from",
    "causal_transformer": "Loaded Causal-Transformer model from",
    "cluster_causal": "Loaded cluster-causal model from",
    "pinnsformer": "Loaded PINNsFormer model from",
}


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def portable_command_path(path):
    """Keep command records relative when the target is inside the release."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def run_model(
    model,
    trace,
    wind,
    rounds,
    seed_a,
    seed_b,
    results_dir,
    logs_dir,
    checkpoints_dir,
):
    command = [
        sys.executable,
        "run.py",
        "--model",
        model,
        "--trace",
        trace,
        "--wind",
        wind,
        "--test_rounds",
        str(rounds),
        "--test_seed_a",
        str(seed_a),
        "--test_seed_b",
        str(seed_b),
        "--logs",
        "0",
        "--save_results",
        "1",
        "--results_dir",
        portable_command_path(results_dir),
    ]
    command.extend(MODEL_ARGUMENTS.get(model, []))
    if model.startswith("aeroace"):
        command.extend([
            "--aero_ckpt",
            portable_command_path(checkpoints_dir / "aeroace_trained.pt"),
        ])
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = logs_dir / f"{model}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    marker = CHECKPOINT_LOAD_MARKERS.get(model)
    checkpoint_status = "not_applicable"
    if marker is not None:
        checkpoint_status = (
            "loaded" if marker in completed.stdout else "missing_confirmation"
        )
    return completed.returncode, checkpoint_status, log_path, command


def aggregate_results(results_dir, summary_path):
    rows = []
    for path in sorted(results_dir.rglob("*.csv")):
        if path.resolve() == summary_path.resolve():
            continue
        with path.open(newline="") as file:
            samples = list(csv.DictReader(file))
        if not samples:
            continue
        required = {
            "model",
            "trajectory",
            "wind",
            "seed",
            "position_mae",
            "position_rmse",
            "e_max",
            "failure",
        }
        if not required.issubset(samples[0]):
            continue

        mae = np.asarray([float(item["position_mae"]) for item in samples])
        rmse = np.asarray([float(item["position_rmse"]) for item in samples])
        e_max = np.asarray([float(item["e_max"]) for item in samples])
        failures = np.asarray([int(item["failure"]) for item in samples])
        seeds = [int(item["seed"]) for item in samples]
        rows.append({
            "model": samples[0]["model"],
            "trajectory": samples[0]["trajectory"],
            "wind": samples[0]["wind"],
            "runs": len(samples),
            "position_mae_mean": float(np.mean(mae)),
            "position_mae_std": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
            "position_rmse_mean": float(np.mean(rmse)),
            "position_rmse_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
            "e_max_mean": float(np.mean(e_max)),
            "failure_count": int(np.sum(failures)),
            "failure_rate": float(np.mean(failures)),
            "seeds": ";".join(str(seed) for seed in seeds),
            "source_csv": str(path.relative_to(results_dir)),
        })

    fieldnames = [
        "model",
        "trajectory",
        "wind",
        "runs",
        "position_mae_mean",
        "position_mae_std",
        "position_rmse_mean",
        "position_rmse_std",
        "e_max_mean",
        "failure_count",
        "failure_rate",
        "seeds",
        "source_csv",
    ]
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        default="fig8",
        choices=["hover", "fig8", "spiral", "sin", "zigzag", "terrain"],
    )
    parser.add_argument(
        "--wind",
        default="gale",
        choices=["breeze", "strong_breeze", "gale"],
    )
    parser.add_argument("--rounds", type=positive_int, default=10)
    parser.add_argument(
        "--test-seed-a",
        type=int,
        default=213,
        help="First test seed in seed = A + round * B.",
    )
    parser.add_argument(
        "--test-seed-b",
        type=int,
        default=10,
        help="Test-seed increment in seed = A + round * B.",
    )
    parser.add_argument(
        "--models",
        default=",".join(PAPER_MODELS),
        help="Comma-separated run.py model names.",
    )
    parser.add_argument("--output", default="results/release_evaluation")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate CSV files already present in --output without running models.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed method instead of testing the remaining methods.",
    )
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not models:
        parser.error("--models must contain at least one release model")
    unknown_models = [model for model in models if model not in ALLOWED_MODELS]
    if unknown_models:
        parser.error(
            "unsupported release model(s): " + ", ".join(unknown_models)
        )

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    if (
        not args.aggregate_only
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        parser.error(
            f"output directory is not empty: {output_dir}. "
            "Use a new directory, or use --aggregate-only to inspect existing CSVs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    if not args.aggregate_only:
        logs_dir = output_dir / "command_logs"
        logs_dir.mkdir(exist_ok=True)
        checkpoints_dir = output_dir / "trained_checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        execution_rows = []
        for model in models:
            print(f"[release evaluation] {model}")
            returncode, checkpoint_status, log_path, command = run_model(
                model,
                args.trace,
                args.wind,
                args.rounds,
                args.test_seed_a,
                args.test_seed_b,
                output_dir,
                logs_dir,
                checkpoints_dir,
            )
            run_failed = (
                returncode != 0
                or checkpoint_status == "missing_confirmation"
            )
            execution_rows.append({
                "model": model,
                "trajectory": args.trace,
                "wind": args.wind,
                "rounds": args.rounds,
                "test_seed_a": args.test_seed_a,
                "test_seed_b": args.test_seed_b,
                "returncode": returncode,
                "status": "failed" if run_failed else "completed",
                "checkpoint_status": checkpoint_status,
                "command_log": str(log_path.relative_to(output_dir)),
                "command": shlex.join(command),
            })
            if run_failed:
                failed.append(model)
                detail = (
                    f"exit code {returncode}"
                    if returncode != 0
                    else "no checkpoint-load confirmation"
                )
                print(f"[release evaluation] {model} failed: {detail}; see {log_path}")
                if args.fail_fast:
                    break

        with (output_dir / "execution_status.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "model",
                    "trajectory",
                    "wind",
                    "rounds",
                    "test_seed_a",
                    "test_seed_b",
                    "returncode",
                    "status",
                    "checkpoint_status",
                    "command_log",
                    "command",
                ],
            )
            writer.writeheader()
            writer.writerows(execution_rows)

    summary_path = output_dir / "summary.csv"
    rows = aggregate_results(output_dir, summary_path)
    print(f"Wrote {len(rows)} summaries to {summary_path}")
    if not rows:
        raise SystemExit("No per-seed result CSVs were found.")
    if failed:
        raise SystemExit(
            "One or more methods failed: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
