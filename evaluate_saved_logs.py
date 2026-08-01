import argparse
import glob
import os
import sys
import numpy as np

from run import evaluate_rollout_metrics


def parse_name_and_round(path):
    base = os.path.basename(path)
    if not base.endswith("_full.npz"):
        return None, -1
    stem = base[:-len("_full.npz")]
    if "_" not in stem:
        return stem, -1
    name, round_str = stem.rsplit("_", 1)
    try:
        round_idx = int(round_str)
    except ValueError:
        round_idx = -1
    return name, round_idx


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_metrics(name, rows):
    rows = sorted(rows, key=lambda x: x[0])
    ace = np.array([r[1]["ace_error"] for r in rows], dtype=float)
    e_max = np.array([r[1]["e_max"] for r in rows], dtype=float)
    tilt = np.array([r[1]["tilt_max"] for r in rows], dtype=float)
    fail = np.array([int(r[1]["failure"]) for r in rows], dtype=float)
    ju = np.array([r[1]["J_u"] for r in rows], dtype=float)
    jdu = np.array([r[1]["J_delta_u"] for r in rows], dtype=float)
    wind_axis = np.array([r[1]["mean_abs_wind_effect_axis"] for r in rows], dtype=float)

    print(f"******* {name} *******")
    print("ACE Error: %.3f(%.3f)" % (np.mean(ace), safe_std(ace)))
    print("Worst-case deviation e_max: %.3f(%.3f)" % (np.mean(e_max), safe_std(e_max)))
    print("Failure Rate: %.2f%% (%d/%d)" % (100.0 * np.mean(fail), int(np.sum(fail)), len(fail)))
    print("Control Effort J_u: %.3f(%.3f)" % (np.mean(ju), safe_std(ju)))
    print("Control Smoothness J_delta_u: %.3f(%.3f)" % (np.mean(jdu), safe_std(jdu)))
    if np.isfinite(wind_axis).any():
        wind_mean = np.nanmean(wind_axis, axis=0)
        print("Mean |wind effect| x/y/z (N): [%.3f, %.3f, %.3f]" % (
            wind_mean[0], wind_mean[1], wind_mean[2]
        ))
    tilt_deg = np.rad2deg(tilt)
    print("Max tilt angle(deg): %.2f(%.2f)" % (np.mean(tilt_deg), safe_std(tilt_deg)))
    print("")


def load_full_rollout(path):
    with np.load(path) as data:
        required = ("X", "pd", "u")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{path} missing keys: {missing}")
        log = {"X": data["X"], "pd": data["pd"], "u": data["u"]}
        if "wind_force" in data:
            log["wind_force"] = data["wind_force"]
        return log


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved rollout logs using run.evaluate_rollout_metrics")
    parser.add_argument("--log_dir", type=str, required=True, help="Path like logs/hover")
    parser.add_argument("--name", type=str, default=None, help="Controller name prefix, e.g. AeroACE")
    parser.add_argument("--fail_pos_err", type=float, default=2.0)
    parser.add_argument("--fail_tilt_deg", type=float, default=75.0)
    args = parser.parse_args()

    pattern = "*_full.npz" if args.name is None else f"{args.name}_*_full.npz"
    files = sorted(glob.glob(os.path.join(args.log_dir, pattern)))
    if not files:
        legacy_npy = glob.glob(os.path.join(args.log_dir, "*.npy"))
        if legacy_npy:
            print("No *_full.npz found. Existing *.npy only contain position logs and cannot compute J_u/J_delta_u.")
            print("Please rerun run.py once (with --logs 1) to generate *_full.npz files.")
            return 1
        print(f"No files matched: {os.path.join(args.log_dir, pattern)}")
        return 1

    fail_tilt_rad = np.deg2rad(args.fail_tilt_deg)
    groups = {}
    for path in files:
        name, round_idx = parse_name_and_round(path)
        if name is None:
            continue
        log = load_full_rollout(path)
        metrics = evaluate_rollout_metrics(log, fail_pos_err=args.fail_pos_err, fail_tilt_rad=fail_tilt_rad)
        groups.setdefault(name, []).append((round_idx, metrics, path))

    if not groups:
        print("No valid *_full.npz rollout files were parsed.")
        return 1

    for name in sorted(groups.keys()):
        rows = sorted(groups[name], key=lambda x: x[0])
        for round_idx, m, _ in rows:
            print(
                "round %d | ACE: %.3f | e_max: %.3f | J_u: %.2f | J_delta_u: %.2f | tilt_max(deg): %.2f | fail: %d"
                % (
                    round_idx,
                    m["ace_error"],
                    m["e_max"],
                    m["J_u"],
                    m["J_delta_u"],
                    np.rad2deg(m["tilt_max"]),
                    int(m["failure"]),
                )
            )
        summarize_metrics(name, rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
