"""Core visualization module to standardize matplotlib plots across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Set the global styling for matplotlib and seaborn."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.family": "sans-serif",
            "lines.linewidth": 1.5,
        }
    )


def plot_equity_curve(
    dates: pd.Series | list | np.ndarray,
    portfolio_values: pd.Series | list | np.ndarray,
    benchmark_values: pd.Series | list | np.ndarray | None = None,
    title: str = "Portfolio Equity Curve",
    output_path: str | Path | None = None,
    show: bool = False,
    ax: plt.Axes | None = None,
) -> None:
    """Plot the portfolio equity curve, optionally comparing it with a benchmark."""
    close_fig = False
    if ax is None:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        close_fig = True

    ax.plot(dates, portfolio_values, label="Strategy", color="blue", linewidth=2)
    
    if benchmark_values is not None:
        ax.plot(
            dates,
            benchmark_values,
            label="Benchmark",
            color="orange",
            linestyle="--",
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if output_path and close_fig:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        
    if show and close_fig:
        plt.show()
        
    if close_fig:
        plt.close()


def plot_allocation_heatmap(
    weights_df: pd.DataFrame,
    title: str = "Asset Allocation Heatmap",
    output_path: str | Path | None = None,
    show: bool = False,
    ax: plt.Axes | None = None,
) -> None:
    """Plot a heatmap of asset weights over time."""
    close_fig = False
    if ax is None:
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
        close_fig = True

    im = ax.imshow(
        weights_df.T.values,
        aspect="auto",
        cmap="viridis",
        interpolation="none",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title)
    ax.set_ylabel("Asset Rank")
    ax.set_xlabel("Time Step")
    
    # Adding colorbar requires figure context, best to let caller handle if ax is provided,
    # but we can do it locally if close_fig is True.
    if close_fig:
        plt.colorbar(im, ax=ax, label="Weight")

    if output_path and close_fig:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        
    if show and close_fig:
        plt.show()
        
    if close_fig:
        plt.close()


def plot_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path | None = None,
    horizontal: bool = False,
    color: str = "skyblue",
    show: bool = False,
) -> None:
    """Plot a standard bar chart."""
    plt.figure(figsize=(10, 6))
    
    if horizontal:
        plt.barh(labels, values, color=color)
        plt.xlabel(xlabel)
    else:
        plt.bar(labels, values, color=color)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        
    if show:
        plt.show()
        
    plt.close()


def plot_histogram(
    data: Sequence[float] | pd.Series | np.ndarray,
    bins: int = 50,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    color: str = "coral",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot a standard histogram."""
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        
    if show:
        plt.show()
        
    plt.close()


def plot_multi_line_chart(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    output_path: str | Path | None = None,
    hline: float | None = None,
    show: bool = False,
) -> None:
    """Plot multiple lines from a DataFrame's columns."""
    plt.figure(figsize=(12, 6))
    
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, alpha=0.8)
        
    if hline is not None:
        plt.axhline(hline, color="black", linewidth=1, linestyle="--")
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date / Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        
    if show:
        plt.show()
        
    plt.close()
