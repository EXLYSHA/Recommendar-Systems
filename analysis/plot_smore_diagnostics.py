#!/usr/bin/env python3
"""Quick plotter for SMORE diagnostics exports.

Load one or more `*.npz` files produced by `SMOREMG.export_diagnostics()`
and plot image/text per-band energies.
"""

import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_npz(path: pathlib.Path):
    data = np.load(path)
    return {
        key: np.asarray(data[key]) if key in data else None
        for key in ("img_energy", "txt_energy")
    }


def plot_file(path: pathlib.Path, show: bool = True, save: bool = False, out_dir: Optional[pathlib.Path] = None):
    diag = load_npz(path)
    x = np.arange(diag["img_energy"].shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, diag["img_energy"], label="Image energy")
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, diag["txt_energy"], color="tab:orange", label="Text energy")
    axes[1].set_xlabel("Frequency bin")
    axes[1].set_ylabel("Energy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(path.name)
    fig.tight_layout()

    if save:
        out_dir = out_dir or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{path.stem}.png"
        fig.savefig(png_path, dpi=150)
        print(f"saved plot -> {png_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot SMORE diagnostic npz files")
    parser.add_argument("paths", nargs="+", help="npz files or directories to plot")
    parser.add_argument("--save", action="store_true", help="save figures alongside the npz files")
    parser.add_argument("--no-show", dest="show", action="store_false", help="skip interactive display")
    parser.add_argument("--out-dir", type=pathlib.Path, help="optional output directory for images")
    args = parser.parse_args()

    paths: list[pathlib.Path] = []
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

    for path in paths:
        plot_file(path, show=args.show, save=args.save, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
