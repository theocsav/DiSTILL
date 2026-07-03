#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(description="Scan thresholds over exported fold prediction probabilities.")
    parser.add_argument("--predictions", required=True, help="Path to fold_predictions.csv")
    parser.add_argument("--positive-class", default="systemic_sclerosis")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "positive_class_probability" not in df.columns:
        raise ValueError("fold_predictions.csv is missing positive_class_probability; rerun evaluation with current pipeline code.")

    y_true = df["true_label"].astype(str)
    positive_class = args.positive_class
    labels = sorted(y_true.unique().tolist())
    if positive_class not in labels:
        raise ValueError(f"Positive class {positive_class!r} not present in predictions labels: {labels}")
    negative_candidates = [label for label in labels if label != positive_class]
    if len(negative_candidates) != 1:
        raise ValueError(f"Expected binary labels, found: {labels}")
    negative_class = negative_candidates[0]

    rows = []
    threshold = args.threshold_start
    while threshold <= args.threshold_stop + 1e-9:
        y_pred = df["positive_class_probability"].apply(lambda p: positive_class if float(p) >= threshold else negative_class)
        cm = confusion_matrix(y_true, y_pred, labels=[negative_class, positive_class])
        rows.append(
            {
                "threshold": round(threshold, 4),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
                "healthy_precision": precision_score(y_true, y_pred, labels=[negative_class], average="macro", zero_division=0),
                "healthy_recall": recall_score(y_true, y_pred, labels=[negative_class], average="macro", zero_division=0),
                "healthy_f1": f1_score(y_true, y_pred, labels=[negative_class], average="macro", zero_division=0),
                "positive_precision": precision_score(y_true, y_pred, labels=[positive_class], average="macro", zero_division=0),
                "positive_recall": recall_score(y_true, y_pred, labels=[positive_class], average="macro", zero_division=0),
                "positive_f1": f1_score(y_true, y_pred, labels=[positive_class], average="macro", zero_division=0),
                f"{negative_class}_correct": int(cm[0, 0]),
                f"{negative_class}_called_{positive_class}": int(cm[0, 1]),
                f"{positive_class}_called_{negative_class}": int(cm[1, 0]),
                f"{positive_class}_correct": int(cm[1, 1]),
            }
        )
        threshold += args.threshold_step

    out_df = pd.DataFrame(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False)
        print(f"Wrote threshold scan to {args.output}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
