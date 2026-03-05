# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Post-training metrics plotting utilities.

Reads the ``metrics.csv`` written by PTL's ``CSVLogger`` (always present
after a ``build_trainer``-based run) and saves a 2×2 matplotlib figure.

Usage::

    from rfdetr.lit.plot import plot_metrics
    plot_metrics("output/rfdetr_base/metrics.csv", "output/rfdetr_base/metrics_plot.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def plot_metrics(
    metrics_csv: str,
    output_path: Optional[str] = None,
) -> str:
    """Read a PTL ``CSVLogger`` metrics file and save a 2×2 training plot.

    The figure contains four subplots:

    * **(0, 0)** Training and Validation Loss
    * **(0, 1)** Average Precision @0.50 (base vs EMA)
    * **(1, 0)** Average Precision @0.50:0.95 (base vs EMA)
    * **(1, 1)** Average Recall @0.50:0.95 (base vs EMA)

    Args:
        metrics_csv: Path to the ``metrics.csv`` file produced by
            ``CSVLogger``.
        output_path: Destination for the PNG file.  Defaults to
            ``metrics_plot.png`` next to ``metrics_csv``.

    Returns:
        The absolute path where the figure was saved.

    Raises:
        ImportError: If ``matplotlib`` or ``pandas`` are not installed.
        FileNotFoundError: If ``metrics_csv`` does not exist.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_metrics(). Install it with: pip install matplotlib") from exc

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for plot_metrics(). Install it with: pip install pandas") from exc

    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    if output_path is None:
        output_path = str(csv_path.parent / "metrics_plot.png")

    df = pd.read_csv(csv_path)
    # CSVLogger writes one row per step with NaN for metrics not logged that step.
    # Aggregate by epoch (mean of non-NaN values per column per epoch).
    if "epoch" not in df.columns:
        raise ValueError("metrics.csv does not contain an 'epoch' column.")
    df = df.groupby("epoch").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RF-DETR Training Metrics", fontsize=14)

    def _plot(ax: "plt.Axes", title: str, cols: list[tuple[str, str]]) -> None:
        """Plot one or more columns onto an axis if they exist in df."""
        any_plotted = False
        for col, label in cols:
            if col in df.columns and df[col].notna().any():
                ax.plot(df["epoch"], df[col], label=label)
                any_plotted = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if any_plotted:
            ax.legend()

    _plot(axes[0, 0], "Loss", [("train/loss", "train"), ("val/loss", "val")])
    _plot(
        axes[0, 1],
        "AP@0.50",
        [("val/mAP_50", "base"), ("val/ema_mAP_50", "EMA")],
    )
    _plot(
        axes[1, 0],
        "AP@0.50:0.95",
        [("val/mAP_50_95", "base"), ("val/ema_mAP_50_95", "EMA")],
    )
    _plot(
        axes[1, 1],
        "AR@0.50:0.95",
        [("val/mAR", "base"), ("val/ema_mAR", "EMA")],
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(Path(output_path).resolve())
