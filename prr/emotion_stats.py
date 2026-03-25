#!/usr/bin/env python3
"""
Analyze emotion probability statistics from a CSV file.

This script reads a CSV of frame-level emotion predictions and computes
statistics for each emotion column, including:
  - Mean, median, std dev, min, max
  - Percentiles (25th, 75th, 95th, 99th)
  - Entropy (measure of prediction confidence)
  - Distribution across frames

Usage:
    python emotion_stats.py --csv orchestra_analysis/cyclesix_owl_emotions.csv
    python emotion_stats.py --csv data.csv --by-participant
    python emotion_stats.py --csv data.csv --plot stats_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# columns considered metadata / not emotion scores
NON_EMOTION_COLS = {
    "video_id",
    "filepath",
    "roi",
    "readable_date",
    "readable_time",
    "duration_s",
    "frame_index",
    "timestamp_ms",
    "dominant_emotion",
}

# Canonical preferred order for detected emotions
PREFERRED_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Default color mapping
DEFAULT_COLOR_MAP = {
    "neutral": "gray",
    "angry": "#8B0000",  # dark red
    "disgust": "violet",
    "sad": "#00008B",  # dark blue
    "happy": "orange",
    "surprise": "yellow",
    "fear": "green",
}


def infer_emotion_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer which columns in the dataframe are emotion score columns.

    Heuristic:
      - Numeric dtype
      - Not in NON_EMOTION_COLS
      - Prefer a canonical order if present
    """
    candidate_cols = [
        c
        for c in df.columns
        if c not in NON_EMOTION_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    emotions = [c for c in PREFERRED_ORDER if c in candidate_cols]
    emotions += [c for c in candidate_cols if c not in emotions]
    return emotions


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate minimal required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    # allow comment lines starting with #
    df = pd.read_csv(csv_path, comment="#")
    if "timestamp_ms" not in df.columns:
        raise KeyError("'timestamp_ms' column is required in the CSV.")
    return df


def calculate_entropy(scores: np.ndarray) -> float:
    """
    Calculate Shannon entropy of emotion scores (0-1 scale).
    
    Entropy measures the "spread" of the probability distribution:
      - High entropy (near 1.0): scores are evenly distributed (low confidence)
      - Low entropy (near 0.0): one emotion dominates (high confidence)
    
    Args:
        scores: Array of 7 emotion scores that sum to ~100
    
    Returns:
        Entropy value (0-1 normalized scale)
    """
    # Normalize to 0-1 range
    normalized = scores / 100.0
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    normalized = np.clip(normalized, eps, 1.0)
    # Shannon entropy: -sum(p * log(p))
    entropy = -np.sum(normalized * np.log(normalized))
    # Normalize to 0-1 by dividing by max entropy for 7 emotions
    max_entropy = np.log(7)  # all emotions equally likely
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def compute_emotion_stats(
    df: pd.DataFrame,
    emotions: List[str],
    by_participant: bool = False,
) -> Dict[str, Dict]:
    """
    Compute comprehensive statistics for each emotion.
    
    Args:
        df: DataFrame with emotion columns
        emotions: List of emotion column names
        by_participant: If True, compute stats separately per video_id
    
    Returns:
        Dictionary mapping emotion -> stats dict
    """
    stats = {}
    
    if by_participant:
        # Group by participant
        grouped = df.groupby("video_id") if "video_id" in df.columns else {None: df}
        
        for emotion in emotions:
            all_stats = []
            for participant_id, group_df in grouped:
                emotion_values = group_df[emotion].dropna()
                if len(emotion_values) > 0:
                    all_stats.append(emotion_values)
            
            if all_stats:
                combined_values = pd.concat(all_stats)
                stats[emotion] = _compute_single_emotion_stats(combined_values, emotion)
                stats[emotion]["num_participants"] = len(all_stats)
            else:
                stats[emotion] = {}
    else:
        # Global stats across all frames
        for emotion in emotions:
            emotion_values = df[emotion].dropna()
            if len(emotion_values) > 0:
                stats[emotion] = _compute_single_emotion_stats(emotion_values, emotion)
            else:
                stats[emotion] = {}
    
    return stats


def _compute_single_emotion_stats(values: pd.Series, emotion: str) -> Dict:
    """Compute all statistics for a single emotion series."""
    stats_dict = {
        "count": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "25%": values.quantile(0.25),
        "75%": values.quantile(0.75),
        "95%": values.quantile(0.95),
        "99%": values.quantile(0.99),
    }
    
    return stats_dict


def compute_entropy_stats(df: pd.DataFrame, emotions: List[str]) -> Dict[str, float]:
    """
    Compute entropy statistics across all frames.
    
    Entropy indicates model confidence:
      - Low entropy: one emotion dominates (high confidence in prediction)
      - High entropy: emotions spread evenly (low confidence, ambiguous)
    
    Returns:
        Dictionary with entropy metrics
    """
    # Build score matrix: rows=frames, cols=emotions
    score_matrix = df[emotions].values  # shape: (n_frames, 7)
    
    # Calculate entropy per frame
    entropies = []
    for row_scores in score_matrix:
        # Skip rows with all NaN
        if not np.all(np.isnan(row_scores)):
            row_scores = np.nan_to_num(row_scores, nan=0.0)
            entropy = calculate_entropy(row_scores)
            entropies.append(entropy)
    
    if not entropies:
        return {}
    
    entropies = np.array(entropies)
    
    return {
        "mean_entropy": entropies.mean(),
        "median_entropy": np.median(entropies),
        "std_entropy": entropies.std(),
        "min_entropy": entropies.min(),
        "max_entropy": entropies.max(),
        "entropy_95th_percentile": np.percentile(entropies, 95),
    }


def check_normalization(df: pd.DataFrame, emotions: List[str]) -> Dict[str, float]:
    """
    Verify that emotion scores sum to ~100 per frame (probability normalization).
    
    Returns:
        Statistics on the sums
    """
    sums = df[emotions].sum(axis=1).dropna()
    
    if len(sums) == 0:
        return {}
    
    return {
        "mean_sum": sums.mean(),
        "median_sum": sums.median(),
        "std_sum": sums.std(),
        "min_sum": sums.min(),
        "max_sum": sums.max(),
        "percent_near_100": (np.abs(sums - 100) < 0.1).sum() / len(sums) * 100,
    }


def plot_emotion_distributions(
    df: pd.DataFrame,
    emotions: List[str],
    out_path: Path,
    figsize: Optional[tuple] = None,
):
    """
    Create box plots and histograms for each emotion distribution.
    """
    if plt is None or sns is None:
        print("Warning: matplotlib or seaborn not available; skipping plot generation")
        return
    
    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})
    
    if figsize is None:
        figsize = (14, 10)
    
    n = len(emotions)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, emotion in enumerate(emotions):
        ax = axes[idx]
        
        values = df[emotion].dropna()
        color = DEFAULT_COLOR_MAP.get(emotion, "steelblue")
        
        # Histogram with box plot overlay
        ax.hist(values, bins=50, alpha=0.7, color=color, edgecolor="black")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{emotion.capitalize()} Distribution", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f"μ={values.mean():.1f}\n"
            f"σ={values.std():.1f}\n"
            f"n={len(values)}"
        )
        ax.text(
            0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    
    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Emotion Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved distribution plots to: {out_path}")


def print_stats_table(stats: Dict[str, Dict], emotions: List[str]):
    """Print statistics in a nicely formatted table."""
    print("\n" + "="*100)
    print("EMOTION PROBABILITY STATISTICS")
    print("="*100)
    
    for emotion in emotions:
        if emotion not in stats or not stats[emotion]:
            print(f"\n{emotion.upper()}: No data")
            continue
        
        s = stats[emotion]
        print(f"\n{emotion.upper()}:")
        print(f"  Count:          {s.get('count', 'N/A')}")
        print(f"  Mean:           {s.get('mean', 'N/A'):.2f}")
        print(f"  Median:         {s.get('median', 'N/A'):.2f}")
        print(f"  Std Dev:        {s.get('std', 'N/A'):.2f}")
        print(f"  Range:          [{s.get('min', 'N/A'):.2f}, {s.get('max', 'N/A'):.2f}]")
        print(f"  25th percentile: {s.get('25%', 'N/A'):.2f}")
        print(f"  75th percentile: {s.get('75%', 'N/A'):.2f}")
        print(f"  95th percentile: {s.get('95%', 'N/A'):.2f}")
        print(f"  99th percentile: {s.get('99%', 'N/A'):.2f}")
        
        if "num_participants" in s:
            print(f"  Participants:   {s['num_participants']}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analyze emotion probability statistics from CSV."
    )
    p.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to input CSV file with emotion columns",
    )
    p.add_argument(
        "--by-participant",
        action="store_true",
        help="Compute statistics separately per participant (video_id)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional output path for distribution plots (PNG)",
    )
    p.add_argument(
        "--emotions",
        "-e",
        nargs="*",
        default=None,
        help="Explicit list of emotion columns (default: auto-detect)",
    )
    
    args = p.parse_args(argv)
    
    csv_path = Path(args.csv)
    
    print(f"Loading {csv_path}...")
    df = load_csv(csv_path)
    
    if args.emotions and len(args.emotions) > 0:
        emotions = args.emotions
    else:
        emotions = infer_emotion_columns(df)
    
    if not emotions:
        print("No emotion columns detected. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df):,} rows; {len(emotions)} emotions detected: {', '.join(emotions)}")
    
    # Check normalization
    print("\n" + "="*100)
    print("NORMALIZATION CHECK (should sum to ~100)")
    print("="*100)
    norm_stats = check_normalization(df, emotions)
    for key, value in norm_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Compute emotion statistics
    print("\nComputing emotion statistics...")
    emotion_stats = compute_emotion_stats(df, emotions, by_participant=args.by_participant)
    print_stats_table(emotion_stats, emotions)
    
    # Compute entropy statistics
    print("\n" + "="*100)
    print("ENTROPY STATISTICS (Model Confidence)")
    print("="*100)
    print("Low entropy = high confidence (one emotion dominates)")
    print("High entropy = low confidence (emotions spread evenly)")
    print("-"*100)
    
    entropy_stats = compute_entropy_stats(df, emotions)
    for key, value in entropy_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating distribution plots...")
        plot_emotion_distributions(df, emotions, Path(args.plot))
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)


if __name__ == "__main__":
    main()