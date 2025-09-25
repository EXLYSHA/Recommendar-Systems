#!/usr/bin/env python3
"""Visualize SMORE diagnostics evolution across epochs.

Given a set of exported `.npz` diagnostics (one per epoch), plot heatmaps
and summary curves to show how spectral energy shifts through training.
"""

import argparse
import pathlib
import re
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_npz(path: pathlib.Path):
    data = np.load(path)
    return {
        key: np.asarray(data[key]) if key in data else None
        for key in ("img_energy", "txt_energy")
    }


def extract_epoch(path: pathlib.Path) -> Optional[int]:
    match = re.search(r"epoch(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def sort_paths(paths: Sequence[pathlib.Path]) -> List[pathlib.Path]:
    def _key(p: pathlib.Path):
        epoch = extract_epoch(p)
        return (epoch if epoch is not None else float("inf"), p.name)

    return sorted(paths, key=_key)


def plot_sequence(paths: Sequence[pathlib.Path], show: bool, save: bool, out_dir: Optional[pathlib.Path]):
    ordered = sort_paths(paths)
    diagnostics = [load_npz(p) for p in ordered]

    img_matrix = np.stack([d["img_energy"] for d in diagnostics])
    txt_matrix = np.stack([d["txt_energy"] for d in diagnostics])

    epochs = [extract_epoch(p) for p in ordered]
    epoch_labels = [e if e is not None else idx + 1 for idx, e in enumerate(epochs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    im0 = axes[0, 0].imshow(img_matrix, aspect="auto", origin="lower", cmap="viridis")
    axes[0, 0].set_ylabel("Epoch")
    axes[0, 0].set_title("Image energy per frequency")
    axes[0, 0].set_yticks(range(len(epoch_labels)))
    axes[0, 0].set_yticklabels(epoch_labels)
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(txt_matrix, aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_title("Text energy per frequency")
    axes[0, 1].set_yticks(range(len(epoch_labels)))
    axes[0, 1].set_yticklabels(epoch_labels)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    axes[1, 0].plot(epoch_labels, img_matrix.mean(axis=1), label="Mean")
    axes[1, 0].plot(epoch_labels, img_matrix.max(axis=1), label="Max", linestyle="--")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Energy")
    axes[1, 0].set_title("Image energy summary")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epoch_labels, txt_matrix.mean(axis=1), label="Mean")
    axes[1, 1].plot(epoch_labels, txt_matrix.max(axis=1), label="Max", linestyle="--")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Energy")
    axes[1, 1].set_title("Text energy summary")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("SMORE diagnostics across epochs")
    fig.tight_layout()

    if save:
        out_dir = out_dir or ordered[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "smore_diagnostics_over_epochs.png"
        fig.savefig(png_path, dpi=150)
        print(f"saved plot -> {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot SMORE diagnostic npz files across epochs")
    parser.add_argument("paths", nargs="+", help="npz files or directories to aggregate (one per epoch)")
    parser.add_argument("--save", action="store_true", help="save the figure to disk")
    parser.add_argument("--no-show", dest="show", action="store_false", help="skip interactive display")
    parser.add_argument("--out-dir", type=pathlib.Path, help="optional output directory for the figure")
    args = parser.parse_args()

    paths: List[pathlib.Path] = []
    for entry in args.paths:
        p = pathlib.Path(entry)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz" and p.exists():
            paths.append(p)
        else:
            print(f"skip -> {p} (not found or not an npz file)")

    if not paths:
        raise SystemExit("No npz files found")

    plot_sequence(paths, show=args.show, save=args.save, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
