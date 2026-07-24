from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = ["healthy", "systemic_sclerosis"]
COLORS = {"healthy": "#2b6cb0", "systemic_sclerosis": "#c53030"}


def normalize_label(value: object) -> str:
    text = str(value).strip().lower().replace("/", "_").replace(" ", "_")
    if text in {"healthy", "hc"}:
        return "healthy"
    if text in {"systemic_sclerosis", "systemic_sclerosis_", "ssc"}:
        return "systemic_sclerosis"
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count predicted labels per patient from FOV-level outer-CV predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="fold_predictions.csv")
    parser.add_argument("--metadata", type=Path, required=True, help="fov_metadata.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def render_patient_plot(row: pd.Series, output_path: Path) -> None:
    values = [int(row["predicted_healthy_fovs"]), int(row["predicted_systemic_sclerosis_fovs"])]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    left = 0
    for label, value in zip(LABELS, values):
        ax.barh([0], [value], left=left, color=COLORS[label], label=label.replace("_", " ").title())
        if value:
            ax.text(left + value / 2, 0, str(value), ha="center", va="center", color="white", weight="bold")
        left += value

    ground_truth = str(row["ground_truth"]).replace("_", " ").title()
    accuracy = float(row["accuracy"])
    ax.set_yticks([])
    ax.set_xlabel("Number of 750 um FOVs")
    ax.set_xlim(0, max(1, int(row["total_fovs"])))
    ax.set_title(f"{row['patient']} | Ground truth: {ground_truth}\nFOV prediction counts", weight="bold")
    ax.text(
        0.5,
        -0.28,
        f"Total FOVs: {int(row['total_fovs'])}    Correct: {int(row['correct_fovs'])}    FOV accuracy: {accuracy:.3f}",
        transform=ax.transAxes,
        ha="center",
        color="#4a5568",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_overview(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.sort_values(["ground_truth", "patient"]).reset_index(drop=True)
    fig, axes = plt.subplots(len(plot_data), 1, figsize=(10, max(5, 0.55 * len(plot_data))))
    if len(plot_data) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, plot_data.iterrows()):
        left = 0
        for label in LABELS:
            value = int(row[f"predicted_{label}_fovs"])
            ax.barh([0], [value], left=left, color=COLORS[label])
            if value:
                ax.text(left + value / 2, 0, str(value), ha="center", va="center", color="white", fontsize=8, weight="bold")
            left += value
        ax.set_yticks([0], [str(row["patient"])])
        ax.set_xlim(0, max(1, int(row["total_fovs"])))
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Number of 750 um FOVs")
    fig.suptitle("Patient-level FOV predictions with ground truth", weight="bold", y=0.995)
    fig.text(0.5, 0.01, "Blue = predicted healthy; red = predicted systemic sclerosis", ha="center", color="#4a5568")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    required_predictions = {"item_id", "predicted_label", "true_label"}
    required_metadata = {"fov_key", "patient", "Disease_State"}
    missing_predictions = required_predictions - set(predictions.columns)
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_predictions:
        raise ValueError(f"Predictions missing columns: {sorted(missing_predictions)}")
    if missing_metadata:
        raise ValueError(f"Metadata missing columns: {sorted(missing_metadata)}")

    predictions["item_id"] = predictions["item_id"].astype(str)
    metadata["fov_key"] = metadata["fov_key"].astype(str)
    if predictions["item_id"].duplicated().any():
        raise ValueError("Each FOV must appear once in the outer-fold prediction file.")

    merged = predictions.merge(
        metadata[["fov_key", "patient", "Disease_State"]],
        left_on="item_id",
        right_on="fov_key",
        how="left",
        validate="one_to_one",
    )
    if merged["patient"].isna().any():
        missing = merged.loc[merged["patient"].isna(), "item_id"].head(10).tolist()
        raise ValueError(f"Could not map prediction FOVs to metadata. Examples: {missing}")

    merged["ground_truth"] = merged["Disease_State"].map(normalize_label)
    merged["predicted_label"] = merged["predicted_label"].map(normalize_label)
    merged["model_true_label"] = merged["true_label"].map(normalize_label)
    mismatch = merged["ground_truth"] != merged["model_true_label"]
    if mismatch.any():
        raise ValueError(f"Metadata and prediction ground truth disagree for {int(mismatch.sum())} FOVs.")

    summary = (
        merged.assign(correct=merged["ground_truth"] == merged["predicted_label"])
        .groupby(["patient", "ground_truth", "predicted_label"], as_index=False)
        .size()
        .pivot_table(index=["patient", "ground_truth"], columns="predicted_label", values="size", fill_value=0)
        .reset_index()
    )
    for label in LABELS:
        if label not in summary.columns:
            summary[label] = 0
    summary = summary.rename(
        columns={label: f"predicted_{label}_fovs" for label in LABELS}
    )
    summary["total_fovs"] = summary[[f"predicted_{label}_fovs" for label in LABELS]].sum(axis=1)
    summary["correct_fovs"] = summary.apply(
        lambda row: row[f"predicted_{row['ground_truth']}_fovs"], axis=1
    )
    summary["incorrect_fovs"] = summary["total_fovs"] - summary["correct_fovs"]
    summary["accuracy"] = summary["correct_fovs"] / summary["total_fovs"]
    summary = summary[
        [
            "patient",
            "ground_truth",
            "total_fovs",
            "predicted_healthy_fovs",
            "predicted_systemic_sclerosis_fovs",
            "correct_fovs",
            "incorrect_fovs",
            "accuracy",
        ]
    ].sort_values(["ground_truth", "patient"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "fov_predictions_with_ground_truth.csv", index=False)
    summary.to_csv(args.output_dir / "patient_fov_prediction_counts.csv", index=False)
    render_overview(summary, args.output_dir / "patient_fov_prediction_counts.png")
    slide_dir = args.output_dir / "patient_slides"
    slide_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        render_patient_plot(row, slide_dir / f"{row['patient']}_fov_predictions.png")

    print(summary.to_string(index=False))
    print(f"Wrote patient summary: {args.output_dir / 'patient_fov_prediction_counts.csv'}")


if __name__ == "__main__":
    main()
