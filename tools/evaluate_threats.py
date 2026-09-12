"""Evaluate alert thresholds against manually labelled event CSV data.

CSV columns required: ``threat_score`` and ``is_threat`` (0/1).
Example:
    python tools/evaluate_threats.py labels.csv --threshold 60
"""

from __future__ import annotations

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IBVAP threat alerts")
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=60)
    args = parser.parse_args()

    tp = fp = fn = tn = 0
    with open(args.csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            predicted = float(row["threat_score"]) >= args.threshold
            actual = str(row["is_threat"]).strip().lower() in {"1", "true", "yes"}
            if predicted and actual:
                tp += 1
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    print(f"threshold={args.threshold}")
    print(f"confusion_matrix=TP:{tp} FP:{fp} FN:{fn} TN:{tn}")
    print(f"precision={precision:.4f}")
    print(f"recall={recall:.4f}")
    print(f"f1={f1:.4f}")
    print(f"false_positive_rate={false_positive_rate:.4f}")


if __name__ == "__main__":
    main()
