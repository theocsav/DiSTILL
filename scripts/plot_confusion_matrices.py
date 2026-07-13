from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one or more confusion_matrix.csv files as a figure."
    )
    parser.add_argument(
        "--matrix",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Add a confusion matrix with display label and CSV path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Confusion Matrices",
        help="Figure title.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Confusion matrix must be square: {path}")
    return frame


def main() -> None:
    args = parse_args()
    matrices = [(label, load_matrix(Path(csv_path))) for label, csv_path in args.matrix]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.8))
    if len(matrices) == 1:
        axes = [axes]

    vmax = max(int(frame.to_numpy().max()) for _, frame in matrices)

    for ax, (label, frame) in zip(axes, matrices):
        sns.heatmap(
            frame,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(label, fontsize=12, pad=10)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(args.title, fontsize=14, y=0.98)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
