"""Evaluate the retained AeroACE checkpoint without retraining it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

import controller
import quadsim
from run import evaluate_rollout_metrics
import trajectory


BASE_DIR = Path(__file__).resolve().parent
RETAINED_CHECKPOINT_SHA256 = (
    "12ce374e08f0caf2b64f8668425adf5cd680241eead8cea63132fa547bb06dfa"
)
RESULT_FIELDS = [
    "model",
    "trajectory",
    "wind",
    "wind_mean",
    "wind_std",
    "round",
    "seed",
    "position_mae",
    "position_rmse",
    "e_max",
    "terminal_z_error",
    "J_u",
    "J_delta_u",
    "max_tilt_deg",
    "failure",
]
SUMMARY_FIELDS = [
    "model",
    "trajectory",
    "wind",
    "c_refer_threshold_coefficient",
    "runs",
    "position_mae_mean",
    "position_mae_std",
    "position_rmse_mean",
    "position_rmse_std",
    "e_max_mean",
    "failure_count",
    "failure_rate",
    "direct_retrieval_fraction",
    "seeds",
    "source_csv",
]
RETRIEVAL_DIAGNOSTIC_FIELDS = [
    "c_refer_threshold_coefficient",
    "round",
    "seed",
    "retrieval_steps",
    "direct_retrieval_steps",
    "reference_steps",
    "direct_retrieval_fraction",
    "ratio_median",
    "ratio_p95",
    "ratio_max",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def safe_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def build_controller(args):
    controller.setup_seed(args.init_seed)
    return controller.AeroACE(
        given_pid=True,
        p=args.p,
        i=args.i,
        d=args.d,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        expert_dim=3 * args.hidden_dim,
        dict_max_entries=args.dict_max_entries,
        dict_normalize_keys=bool(args.dict_normalize_keys),
        dict_temperature=args.dict_temperature,
        dict_min_cosine_distance=args.dict_min_cosine_distance,
        dict_max_entries_per_bucket=args.dict_max_entries_per_bucket,
        dict_ema_alpha=args.dict_ema_alpha,
        c_refer_threshold_coefficient=args.c_refer_coefficient,
        online_update=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="params/aeroace_trained.pt")
    parser.add_argument(
        "--output",
        default="results/aeroace_checkpoint_evaluation",
        help="New or empty output directory.",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--test-seed-a", type=int, default=213)
    parser.add_argument("--test-seed-b", type=int, default=10)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--wind-mean", type=float, default=0.5)
    parser.add_argument("--wind-std", type=float, default=30.0)
    parser.add_argument("--fail-pos-err", type=float, default=2.0)
    parser.add_argument("--fail-tilt-deg", type=float, default=75.0)
    parser.add_argument("--p", type=float, default=2.503)
    parser.add_argument("--i", type=float, default=1.58)
    parser.add_argument("--d", type=float, default=7.647)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=33)
    parser.add_argument("--dict-max-entries", type=int, default=10000)
    parser.add_argument("--dict-normalize-keys", type=int, default=0)
    parser.add_argument("--dict-temperature", type=float, default=1.0)
    parser.add_argument("--dict-min-cosine-distance", type=float, default=0.01)
    parser.add_argument("--dict-max-entries-per-bucket", type=int, default=1024)
    parser.add_argument("--dict-ema-alpha", type=float, default=0.9)
    parser.add_argument("--c-refer-coefficient", type=float, default=0.1)
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.wind_std < 0.0:
        parser.error("--wind-std must be non-negative")
    if args.c_refer_coefficient < 0.0:
        parser.error("--c-refer-coefficient must be non-negative")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = BASE_DIR / checkpoint
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != RETAINED_CHECKPOINT_SHA256:
        parser.error(
            "checkpoint SHA-256 does not match the retained release artifact: "
            f"{checkpoint_hash}"
        )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    retrieval_config_fields = (
        "expert_normalize_keys",
        "expert_temperature",
        "expert_min_cosine_distance",
        "expert_max_entries_per_bucket",
        "expert_ema_alpha",
    )

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    aeroace = build_controller(args)
    aeroace.load(checkpoint, map_location="cpu")
    aeroace.state = "test"
    dictionary_size_before = int(aeroace.expert_dict.current_size)
    if dictionary_size_before <= 0:
        raise RuntimeError("retained checkpoint loaded an empty Expert Dictionary")

    retrieval_ratios = []
    original_get_c_refer = aeroace.get_c_refer

    def monitored_get_c_refer(h_t, c_last):
        c_refer = original_get_c_refer(h_t, c_last)
        retrieval_ratios.append(
            float(
                np.linalg.norm(np.asarray(c_last) - np.asarray(c_refer))
                / np.sqrt(aeroace.hidden_dim)
            )
        )
        return c_refer

    aeroace.get_c_refer = monitored_get_c_refer

    quadrotor = quadsim.Quadrotor()
    quadrotor.params["disable_progress"] = True
    quadrotor_state = quadrotor.state
    trace = trajectory.fig8()
    aeroace.reset_controller()

    rows = []
    retrieval_rows = []
    for round_index in range(args.rounds):
        seed = args.test_seed_a + round_index * args.test_seed_b
        controller.setup_seed(seed)
        retrieval_ratios.clear()
        wind_velocity = np.random.normal(
            loc=args.wind_mean,
            scale=args.wind_std,
            size=(30000, 3),
        )
        log = quadrotor.run(
            trajectory=trace,
            controller=aeroace,
            wind_velocity_list=wind_velocity,
            reset_control=True,
            Name="AeroACE",
        )
        metrics = evaluate_rollout_metrics(
            log,
            fail_pos_err=args.fail_pos_err,
            fail_tilt_rad=np.deg2rad(args.fail_tilt_deg),
        )
        row = {
            "model": "AeroACE",
            "trajectory": "fig8",
            "wind": "gale",
            "wind_mean": float(args.wind_mean),
            "wind_std": float(args.wind_std),
            "round": int(round_index),
            "seed": int(seed),
            "position_mae": float(metrics["ace_error"]),
            "position_rmse": float(metrics["rmse_error"]),
            "e_max": float(metrics["e_max"]),
            "terminal_z_error": float(metrics["terminal_z_error"]),
            "J_u": float(metrics["J_u"]),
            "J_delta_u": float(metrics["J_delta_u"]),
            "max_tilt_deg": float(np.rad2deg(metrics["tilt_max"])),
            "failure": int(metrics["failure"]),
        }
        rows.append(row)
        ratios = np.asarray(retrieval_ratios, dtype=float)
        if ratios.size == 0:
            raise RuntimeError(
                f"no AeroACE retrieval decisions recorded for seed {seed}"
            )
        direct_steps = int(
            np.sum(ratios <= args.c_refer_coefficient)
        )
        retrieval_rows.append(
            {
                "c_refer_threshold_coefficient": float(
                    args.c_refer_coefficient
                ),
                "round": int(round_index),
                "seed": int(seed),
                "retrieval_steps": int(ratios.size),
                "direct_retrieval_steps": direct_steps,
                "reference_steps": int(ratios.size - direct_steps),
                "direct_retrieval_fraction": float(
                    direct_steps / ratios.size
                ),
                "ratio_median": float(np.median(ratios)),
                "ratio_p95": float(np.percentile(ratios, 95)),
                "ratio_max": float(np.max(ratios)),
            }
        )
        print(
            "round %d seed %d | MAE %.6f | RMSE %.6f | e_max %.6f | "
            "tilt %.3f deg | failure %d"
            % (
                round_index,
                seed,
                row["position_mae"],
                row["position_rmse"],
                row["e_max"],
                row["max_tilt_deg"],
                row["failure"],
            )
        )

    raw_path = output_dir / "fig8_gale_AeroACE.csv"
    with raw_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "retrieval_diagnostics.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=RETRIEVAL_DIAGNOSTIC_FIELDS
        )
        writer.writeheader()
        writer.writerows(retrieval_rows)

    mae = np.asarray([row["position_mae"] for row in rows])
    rmse = np.asarray([row["position_rmse"] for row in rows])
    e_max = np.asarray([row["e_max"] for row in rows])
    failures = np.asarray([row["failure"] for row in rows], dtype=int)
    retrieval_steps = int(
        sum(row["retrieval_steps"] for row in retrieval_rows)
    )
    direct_retrieval_steps = int(
        sum(row["direct_retrieval_steps"] for row in retrieval_rows)
    )
    summary = {
        "model": "AeroACE",
        "trajectory": "fig8",
        "wind": "gale",
        "c_refer_threshold_coefficient": float(
            args.c_refer_coefficient
        ),
        "runs": len(rows),
        "position_mae_mean": float(np.mean(mae)),
        "position_mae_std": safe_std(mae),
        "position_rmse_mean": float(np.mean(rmse)),
        "position_rmse_std": safe_std(rmse),
        "e_max_mean": float(np.mean(e_max)),
        "e_max_max": float(np.max(e_max)),
        "max_tilt_deg_max": float(
            np.max([row["max_tilt_deg"] for row in rows])
        ),
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
        "retrieval_steps": retrieval_steps,
        "direct_retrieval_steps": direct_retrieval_steps,
        "reference_steps": retrieval_steps - direct_retrieval_steps,
        "direct_retrieval_fraction": float(
            direct_retrieval_steps / retrieval_steps
        ),
        "failure_seeds": [
            row["seed"] for row in rows if int(row["failure"]) == 1
        ],
        "seeds": [row["seed"] for row in rows],
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    summary_csv_row = {
        key: summary[key]
        for key in SUMMARY_FIELDS
        if key not in ("seeds", "source_csv")
    }
    summary_csv_row["seeds"] = ";".join(
        str(seed) for seed in summary["seeds"]
    )
    summary_csv_row["source_csv"] = raw_path.name
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary_csv_row)

    dictionary_size_after = int(aeroace.expert_dict.current_size)
    metadata = {
        "checkpoint": release_relative(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_confirmed": True,
        "dictionary_size_before": dictionary_size_before,
        "dictionary_size_after": dictionary_size_after,
        "online_update_enabled": bool(aeroace.online_update_enabled),
        "c_refer_threshold_coefficient": float(
            aeroace.c_refer_threshold_coefficient
        ),
        "checkpoint_retrieval_config_present": all(
            field in checkpoint_payload for field in retrieval_config_fields
        ),
        "effective_expert_dictionary": {
            "key_dim": int(aeroace.expert_dict.key_dim),
            "value_dim": int(aeroace.expert_dict.value_dim),
            "max_entries": int(aeroace.expert_dict.max_entries),
            "normalize_keys": bool(aeroace.expert_dict.normalize_keys),
            "temperature": float(aeroace.expert_dict.temperature),
            "min_cosine_distance": float(
                aeroace.expert_dict.min_cosine_distance
            ),
            "max_entries_per_bucket": int(
                aeroace.expert_dict.max_entries_per_bucket
            ),
            "ema_alpha": float(aeroace.expert_dict.ema_alpha),
        },
        "controller_init_seed": args.init_seed,
        "quadrotor_state": quadrotor_state,
        "train_t_stop": float(quadrotor.params["train_t_stop"]),
        "test_t_stop": float(quadrotor.params["test_t_stop"]),
        "effective_pid": {
            "K_p": np.asarray(aeroace.params["K_p"]).tolist(),
            "K_i": np.asarray(aeroace.params["K_i"]).tolist(),
            "K_d": np.asarray(aeroace.params["K_d"]).tolist(),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_sha256": {
            name: sha256(BASE_DIR / name)
            for name in (
                "controller.py",
                "quadsim.py",
                "run.py",
                "trajectory.py",
            )
        },
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    if dictionary_size_after != dictionary_size_before:
        raise RuntimeError(
            "Expert Dictionary changed during test-only evaluation: "
            f"{dictionary_size_before} -> {dictionary_size_after}"
        )
    print(
        "AeroACE | position MAE %.6f ± %.6f | failures %d/%d"
        % (
            summary["position_mae_mean"],
            summary["position_mae_std"],
            summary["failure_count"],
            summary["runs"],
        )
    )
    print(f"Wrote {raw_path}")


if __name__ == "__main__":
    main()
