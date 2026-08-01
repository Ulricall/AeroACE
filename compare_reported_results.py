"""Compare aggregated release results with values printed in the final manuscript."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

METHOD_MAP = {
    "OMAC(deep)": "OMAC",
    "Decision Transformer": "Transformer",
    "dmrac": "DMRAC",
    "NeuralBEM": "NeuroBEM",
    "pitcn": "PI-TCN",
    "mann": "MANN",
    "agile_full": "Agile",
    "Pi-Transformer": "Pi-Transformer",
    "Powerformer": "PowerFormer",
    "Causal-Transformer": "Causal Transformer",
    "Cluster-Causal": "Causal Attn. Masking",
    "PINNsFormer": "Pinnsformer",
    "AeroACE + vanilla GRU": "GRU",
}

TRAJECTORY_MAP = {
    "hover": "Hover",
    "fig8": "Figure-8",
    "spiral": "Spiral",
    "sin": "Sin-forward",
    "terrain": "Terrain",
}

WIND_MAP = {
    "breeze": "Wind I",
    "strong_breeze": "Wind II",
    "gale": "Wind III",
}


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--reported",
        default=str(BASE_DIR / "docs" / "reported_results.csv"),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    reported_path = Path(args.reported)
    output_path = (
        Path(args.output)
        if args.output
        else summary_path.with_name("comparison_to_reported.csv")
    )

    reported_rows = read_csv(reported_path)
    index = {}
    for row in reported_rows:
        key = (row["method"], row["split"], row["trajectory"], row["wind"])
        index[key] = row

    comparisons = []
    for row in read_csv(summary_path):
        method = METHOD_MAP.get(row["model"], row["model"])
        trajectory = TRAJECTORY_MAP.get(row["trajectory"], row["trajectory"])
        wind = WIND_MAP.get(row["wind"], row["wind"])
        reference = index.get((method, "test", trajectory, wind))
        setting_match = "exact"
        if reference is None:
            # The Transformer-variant table labels the domain only as "test set"
            # and does not identify Wind I/II/III.  Keep the printed number
            # available for inspection, but do not claim an exact setting match.
            reference = index.get((method, "test", trajectory, ""))
            setting_match = "reported_wind_unspecified"

        item = {
            "model": row["model"],
            "trajectory": row["trajectory"],
            "wind": row["wind"],
            "runs": row["runs"],
            "measured_mean": row["position_mae_mean"],
            "measured_std": row["position_mae_std"],
            "reported_mean": "",
            "reported_std": "",
            "mean_difference": "",
            "std_difference": "",
            "status": "no_matching_reported_setting",
            "setting_match": "",
            "source_line": "",
        }
        if reference is not None:
            measured_mean = float(row["position_mae_mean"])
            measured_std = float(row["position_mae_std"])
            reported_mean = float(reference["mean"])
            reported_std = float(reference["std"])
            item.update({
                "reported_mean": reported_mean,
                "reported_std": reported_std,
                "mean_difference": measured_mean - reported_mean,
                "std_difference": measured_std - reported_std,
                "setting_match": setting_match,
                "source_line": reference["source_line"],
            })
            if setting_match != "exact":
                item["status"] = "reported_setting_unspecified"
            elif int(row["runs"]) != 10:
                item["status"] = "insufficient_runs"
            elif (
                round(measured_mean, 3) == round(reported_mean, 3)
                and round(measured_std, 3) == round(reported_std, 3)
            ):
                item["status"] = "matches_reported_rounding"
            else:
                item["status"] = "does_not_match_reported_rounding"

        comparisons.append(item)

    fieldnames = list(comparisons[0].keys()) if comparisons else [
        "model", "status"
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"Wrote {len(comparisons)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
