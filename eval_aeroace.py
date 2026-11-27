"""Evaluate the retained AeroACE checkpoint without retraining it."""

from __future__ import annotations

import argparse
import csv
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
    "terminal_z_error",
    "J_u",
    "J_delta_u",
    "failure",
]
SUMMARY_FIELDS = [
    "model",
    "trajectory",
    "wind",
    "runs",
    "position_mae_mean",
    "position_mae_std",
    "position_rmse_mean",
    "position_rmse_std",
    "failure_count",
    "failure_rate",
    "seeds",
    "source_csv",
]


def release_relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def safe_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def build_trajectory(trace_name):
    if trace_name == "hover":
        return trajectory.hover()
    if trace_name == "fig8":
        return trajectory.fig8()
    if trace_name == "spiral":
        return trajectory.spiral_up()
    if trace_name == "sin":
        return trajectory.sin_forward()
    if trace_name == "zigzag":
        return trajectory.ZigZag()
    if trace_name == "terrain":
        return trajectory.TerrainFollowingPath(
            base_path_planner=trajectory.LinePathXY(),
            terrain_model=trajectory.RandomGaussianTerrain(),
        )
    raise ValueError(f"Unknown trajectory: {trace_name}")


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
        online_update=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="params/aeroace_trained.pt")
    parser.add_argument(
        "--trace",
        type=str,
        default="fig8",
        choices=["hover", "fig8", "spiral", "sin", "zigzag", "terrain"],
    )
    parser.add_argument(
        "--output",
        default="results/eval_aeroace",
        help="Output directory (existing files will be overwritten).",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--test-seed-a",
        type=int,
        default=0,
        help="First evaluation seed; round r uses this seed + r.",
    )
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
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.wind_std < 0.0:
        parser.error("--wind-std must be non-negative")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = BASE_DIR / checkpoint
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
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
    output_dir.mkdir(parents=True, exist_ok=True)

    aeroace = build_controller(args)
    aeroace.load(checkpoint, map_location="cpu")
    aeroace.state = "test"
    dictionary_size_before = int(aeroace.expert_dict.current_size)
    if dictionary_size_before <= 0:
        raise RuntimeError("retained checkpoint loaded an empty Expert Dictionary")

    quadrotor = quadsim.Quadrotor()
    quadrotor.params["disable_progress"] = True
    quadrotor_state = quadrotor.state
    trace = build_trajectory(args.trace)
    aeroace.reset_controller()

    rows = []
    for round_index in range(args.rounds):
        seed = args.test_seed_a + round_index
        controller.setup_seed(seed)
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
            "trajectory": args.trace,
            "wind": "gale",
            "wind_mean": float(args.wind_mean),
            "wind_std": float(args.wind_std),
            "round": int(round_index),
            "seed": int(seed),
            "position_mae": float(metrics["ace_error"]),
            "position_rmse": float(metrics["rmse_error"]),
            "terminal_z_error": float(metrics["terminal_z_error"]),
            "J_u": float(metrics["J_u"]),
            "J_delta_u": float(metrics["J_delta_u"]),
            "failure": int(metrics["failure"]),
        }
        rows.append(row)
        print(
            "round %d seed %d | MAE %.3f"
            % (
                round_index,
                seed,
                row["position_mae"],
            )
        )

    raw_path = output_dir / f"{args.trace}_gale_AeroACE.csv"
    with raw_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mae = np.asarray([row["position_mae"] for row in rows])
    rmse = np.asarray([row["position_rmse"] for row in rows])
    failures = np.asarray([row["failure"] for row in rows], dtype=int)
    summary = {
        "model": "AeroACE",
        "trajectory": args.trace,
        "wind": "gale",
        "runs": len(rows),
        "position_mae_mean": float(np.mean(mae)),
        "position_mae_std": safe_std(mae),
        "position_rmse_mean": float(np.mean(rmse)),
        "position_rmse_std": safe_std(rmse),
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
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
        "checkpoint_load_confirmed": True,
        "dictionary_size_before": dictionary_size_before,
        "dictionary_size_after": dictionary_size_after,
        "online_update_enabled": bool(aeroace.online_update_enabled),
        "checkpoint_retrieval_config_present": all(
            field in checkpoint_payload for field in retrieval_config_fields
        ),
        "effective_expert_dictionary": {
            "key_dim": int(aeroace.expert_dict.key_dim),
            "value_dim": int(aeroace.expert_dict.value_dim),
            "max_entries": int(aeroace.expert_dict.max_entries),
            "temperature": float(aeroace.expert_dict.temperature),
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
        "AeroACE | position MAE %.3f ± %.3f"
        % (
            summary["position_mae_mean"],
            summary["position_mae_std"],
        )
    )
    print(f"Wrote {raw_path}")


if __name__ == "__main__":
    main()
