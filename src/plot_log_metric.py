#!/usr/bin/env python3
"""Plot metrics (test metrics or train loss) by epoch for each hyperparameter combination."""
import argparse
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

PARAM_PATTERN = re.compile(r"Parameters:\s*\[([^\]]+)\]=\(([^)]+)\)")
EPOCH_PATTERN = re.compile(r"epoch\s+(\d+)\s+evaluating", re.IGNORECASE)
TRAIN_LOSS_PATTERN = re.compile(
    r"epoch\s+(\d+)\s+training.*?train loss:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
METRIC_PATTERN = re.compile(r"([A-Za-z0-9@_]+):\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)")


def parse_log(log_path):
    """Parse the log file and return structured metric data.

    Returns
    -------
    dict
        Mapping from combo key string to list of (epoch, metrics_dict).
    dict
        Mapping from combo key string to human readable label.
    """
    combo_metrics = defaultdict(dict)
    combo_labels = {}
    parse_order = []

    current_combo = None
    current_epoch = None
    waiting_for_test_metrics = False

    with open(log_path, "r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip("\n")

            param_match = PARAM_PATTERN.search(line)
            if param_match:
                names = [piece.strip().strip("'\"") for piece in param_match.group(1).split(',')]
                values = [piece.strip() for piece in param_match.group(2).split(',')]
                label_parts = []
                for idx, name in enumerate(names):
                    if not name:
                        continue
                    value = values[idx] if idx < len(values) else ""
                    label_parts.append(f"{name}={value}")
                combo_key = " | ".join(label_parts) if label_parts else param_match.group(0)
                if combo_key not in combo_labels:
                    combo_labels[combo_key] = ", ".join(label_parts) if label_parts else combo_key
                    parse_order.append(combo_key)
                current_combo = combo_key
                current_epoch = None
                waiting_for_test_metrics = False
                continue

            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                waiting_for_test_metrics = False
                continue

            train_loss_match = TRAIN_LOSS_PATTERN.search(line)
            if train_loss_match and current_combo is not None:
                epoch = int(train_loss_match.group(1))
                try:
                    train_loss_value = float(train_loss_match.group(2))
                except ValueError:
                    continue
                epoch_metrics = combo_metrics[current_combo].setdefault(epoch, {})
                epoch_metrics["train_loss"] = train_loss_value
                current_epoch = epoch
                waiting_for_test_metrics = False
                continue

            if "test result" in line.lower():
                waiting_for_test_metrics = True
                continue

            if waiting_for_test_metrics:
                metric_pairs = METRIC_PATTERN.findall(line)
                if not metric_pairs:
                    # Skip lines without metrics while waiting.
                    continue

                metrics_dict = {}
                for name, value in metric_pairs:
                    try:
                        metrics_dict[name] = float(value)
                    except ValueError:
                        # Ignore unparsable values (e.g. nan strings).
                        continue

                if metrics_dict and current_combo is not None and current_epoch is not None:
                    epoch_metrics = combo_metrics[current_combo].setdefault(current_epoch, {})
                    epoch_metrics.update(metrics_dict)
                waiting_for_test_metrics = False

    # Ensure order of combinations follows appearance in file.
    ordered_metrics = {
        combo: sorted(combo_metrics.get(combo, {}).items(), key=lambda item: item[0])
        for combo in parse_order
    }
    ordered_labels = {combo: combo_labels[combo] for combo in parse_order}
    return ordered_metrics, ordered_labels


def resolve_metric_name(target_metric, available_metrics):
    for candidate in available_metrics:
        if candidate.lower() == target_metric.lower():
            return candidate
    return None


def ensure_output_dir(base_dir, log_path):
    if base_dir:
        output_dir = base_dir
    else:
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        output_dir = os.path.join(os.getcwd(), "plots", log_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sanitize_filename(text):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return safe.strip("_") or "plot"


def plot_metrics(log_path, metric_name, output_dir=None, dpi=150):
    combo_data, combo_labels = parse_log(log_path)
    output_dir = ensure_output_dir(output_dir, log_path)

    created = []
    for combo_key, entries in combo_data.items():
        if not entries:
            continue
        sorted_entries = sorted(entries, key=lambda item: item[0])
        available_metrics = set().union(*(metrics.keys() for _, metrics in sorted_entries))
        resolved_metric = resolve_metric_name(metric_name, available_metrics)
        if not resolved_metric:
            continue

        epochs = [epoch for epoch, _ in sorted_entries]
        values = [metrics.get(resolved_metric) for _, metrics in sorted_entries]
        plottable_values = [v if v is not None else math.nan for v in values]
        if all(math.isnan(v) for v in plottable_values):
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, plottable_values, marker="o", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(resolved_metric)
        plt.title(f"{os.path.basename(log_path)}\n{combo_labels.get(combo_key, combo_key)}")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()

        filename = f"{sanitize_filename(combo_labels.get(combo_key, combo_key))}_{sanitize_filename(resolved_metric)}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        created.append(save_path)

    if not created:
        raise ValueError(
            f"No plots were created. Metric '{metric_name}' was not found in the log or no test metrics were logged."
        )
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics over epochs (including train loss) for each hyperparameter combination in a log file."
    )
    parser.add_argument("log_path", help="Path to the training log file.")
    parser.add_argument(
        "--metric",
        default="recall@10",
        help="Metric to plot (e.g., recall@10, ndcg@20, train_loss). Defaults to recall@10.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to store generated plots. Defaults to ./plots/<log_name>.")
    parser.add_argument("--dpi", type=int, default=150, help="Resolution (DPI) for the saved figures. Defaults to 150.")

    args = parser.parse_args()

    created_paths = plot_metrics(args.log_path, args.metric, args.output_dir, args.dpi)
    print("Saved plots:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
