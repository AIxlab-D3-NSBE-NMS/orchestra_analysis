#!/usr/bin/env python3
"""
Generate two histograms of emotions:
1. Time spent in each emotion (sum of frame durations per emotion)
2. Unique participant count (number of unique video_id per emotion)

This script reads a CSV of frame-level emotion predictions and produces
histograms showing the distribution of time and participants across emotions.

Usage:
    python orchestra_analysis/prr/emotion_histograms.py \
        --csv orchestra_analysis/cyclesix_owl_emotions.csv \
        --out-time emotion_histogram_time.png \
        --out-participants emotion_histogram_participants.png

Dependencies:
    pandas, numpy, matplotlib, seaborn
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
except Exception:
    print(
        "Missing plotting libraries: ensure matplotlib and seaborn are installed.",
        file=sys.stderr,
    )
    raise

# Defaults
DEFAULT_CSV = "orchestra_analysis/cyclesix_owl_emotions.csv"
DEFAULT_OUT_TIME = "emotion_histogram_time.png"
DEFAULT_OUT_PARTICIPANTS = "emotion_histogram_participants.png"

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
    if "video_id" not in df.columns:
        raise KeyError("'video_id' column is required in the CSV.")
    if "timestamp_ms" not in df.columns:
        raise KeyError("'timestamp_ms' column is required in the CSV.")
    return df


def prepare_dataframe(df: pd.DataFrame, emotions: List[str]) -> pd.DataFrame:
    """
    Prepare and validate dataframe:
      - convert timestamp_ms -> timestamp_s (seconds)
      - coerce emotion columns to numeric
      - drop rows without timestamp or without any emotion values
    """
    df = df.copy()
    df["timestamp_s"] = pd.to_numeric(df["timestamp_ms"], errors="coerce") / 1000.0
    df = df.dropna(subset=["timestamp_s"])
    if df.empty:
        raise ValueError("No valid timestamp data after conversion to seconds.")
    for e in emotions:
        df[e] = pd.to_numeric(df[e], errors="coerce")
    df = df.dropna(subset=emotions, how="all")
    if df.empty:
        raise ValueError("No emotion data available after dropping NaN rows.")
    return df


def calculate_time_per_emotion(df: pd.DataFrame, emotions: List[str]) -> Dict[str, float]:
    """
    Calculate total time spent in each emotion.
    
    Assumes each row represents a frame, and computes the time per frame
    by looking at consecutive timestamp differences. Uses the emotion
    with the highest score as the dominant emotion for that frame.
    
    Returns:
        Dictionary mapping emotion -> total time in seconds
    """
    # Find dominant emotion per frame (highest score)
    df_copy = df.copy()
    df_copy["dominant"] = df_copy[emotions].idxmax(axis=1)
    
    # Calculate frame duration (time to next frame)
    df_copy = df_copy.sort_values(by=["video_id", "timestamp_s"])
    df_copy["next_timestamp"] = df_copy.groupby("video_id")["timestamp_s"].shift(-1)
    df_copy["frame_duration"] = df_copy["next_timestamp"] - df_copy["timestamp_s"]
    
    # Fill NaN durations (last frame of each video) with 0
    df_copy["frame_duration"] = df_copy["frame_duration"].fillna(0)
    
    # Sum frame durations by dominant emotion
    time_per_emotion = df_copy.groupby("dominant")["frame_duration"].sum().to_dict()
    
    # Ensure all emotions are present (even if 0)
    for emotion in emotions:
        if emotion not in time_per_emotion:
            time_per_emotion[emotion] = 0.0
    
    return time_per_emotion


def calculate_participants_per_emotion(df: pd.DataFrame, emotions: List[str]) -> Dict[str, int]:
    """
    Calculate number of unique participants (video_id) per emotion.
    
    A participant is counted for an emotion if there's at least one frame
    where that emotion has a non-zero score.
    
    Returns:
        Dictionary mapping emotion -> number of unique participants
    """
    participants_per_emotion = {}
    
    for emotion in emotions:
        # Get rows where this emotion has a non-zero/non-NaN score
        mask = (df[emotion].notna()) & (df[emotion] > 0)
        unique_vids = df[mask]["video_id"].nunique()
        participants_per_emotion[emotion] = unique_vids
    
    return participants_per_emotion


def plot_emotion_histogram_time(
    time_per_emotion: Dict[str, float],
    emotions: List[str],
    out_path: Path,
    figsize: Optional[tuple] = None,
):
    """
    Plot a histogram of total time spent in each emotion.
    
    Args:
        time_per_emotion: Dictionary mapping emotion -> total time (seconds)
        emotions: List of emotion names in preferred order
        out_path: Path to save the figure
        figsize: Optional figure size tuple
    """
    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})
    
    if figsize is None:
        figsize = (10, 6)
    
    # Sort by emotion order
    emotions_sorted = [e for e in emotions if e in time_per_emotion]
    times = [time_per_emotion[e] for e in emotions_sorted]
    
    # Get colors
    colors = [DEFAULT_COLOR_MAP.get(e, "steelblue") for e in emotions_sorted]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(emotions_sorted, times, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    ax.set_xlabel("Emotion", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Total Time Spent in Each Emotion", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.1 if times else 100)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved time histogram to: {out_path}")


def plot_emotion_histogram_participants(
    participants_per_emotion: Dict[str, int],
    emotions: List[str],
    out_path: Path,
    figsize: Optional[tuple] = None,
):
    """
    Plot a histogram of unique participant count per emotion.
    
    Args:
        participants_per_emotion: Dictionary mapping emotion -> participant count
        emotions: List of emotion names in preferred order
        out_path: Path to save the figure
        figsize: Optional figure size tuple
    """
    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})
    
    if figsize is None:
        figsize = (10, 6)
    
    # Sort by emotion order
    emotions_sorted = [e for e in emotions if e in participants_per_emotion]
    counts = [participants_per_emotion[e] for e in emotions_sorted]
    
    # Get colors
    colors = [DEFAULT_COLOR_MAP.get(e, "steelblue") for e in emotions_sorted]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(emotions_sorted, counts, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    ax.set_xlabel("Emotion", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Participants", fontsize=12, fontweight="bold")
    ax.set_title("Number of Unique Participants per Emotion", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.1 if counts else 100)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved participants histogram to: {out_path}")


def main(argv=None):
    """
    Main entry point for the histogram generation script.
    
    Can be called from command line or directly from Python:
    
    From command line:
        python emotion_histograms.py --csv data.csv --out-time time.png --out-participants participants.png
    
    From Python:
        from emotion_histograms import main
        main([
            '--csv', 'orchestra_analysis/cyclesix_owl_emotions.csv',
            '--out-time', 'emotion_histogram_time.png',
            '--out-participants', 'emotion_histogram_participants.png'
        ])
    """
    p = argparse.ArgumentParser(
        description="Create histograms of emotions by time spent and participant count."
    )
    p.add_argument("--csv", "-c", default=DEFAULT_CSV, help="Path to input CSV")
    p.add_argument(
        "--out-time",
        default=DEFAULT_OUT_TIME,
        help="Output image path for time histogram (PNG). Set empty to skip.",
    )
    p.add_argument(
        "--out-participants",
        default=DEFAULT_OUT_PARTICIPANTS,
        help="Output image path for participants histogram (PNG). Set empty to skip.",
    )
    p.add_argument(
        "--emotions",
        "-e",
        nargs="*",
        default=None,
        help="Explicit list of emotion columns to plot (default: auto-detect).",
    )
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    out_time = Path(args.out_time) if args.out_time else None
    out_participants = Path(args.out_participants) if args.out_participants else None

    df = load_csv(csv_path)

    if args.emotions and len(args.emotions) > 0:
        emotions = args.emotions
    else:
        emotions = infer_emotion_columns(df)

    if not emotions:
        print("No emotion columns detected in the CSV. Exiting.", file=sys.stderr)
        sys.exit(2)

    # quick check: ensure at least one non-empty emotion row
    df_check = df.dropna(subset=emotions, how="all")
    if df_check.empty:
        print(
            "No emotion data available after dropping NaNs. Exiting.", file=sys.stderr
        )
        sys.exit(3)

    print(
        f"Loaded {len(df):,} rows from {csv_path}; {df['video_id'].nunique():,} unique participants; analyzing {len(emotions)} emotions."
    )

    try:
        prepared = prepare_dataframe(df, emotions)
    except Exception as exc:
        print(f"Error preparing dataframe: {exc}", file=sys.stderr)
        sys.exit(4)

    # Calculate metrics
    time_per_emotion = calculate_time_per_emotion(prepared, emotions)
    participants_per_emotion = calculate_participants_per_emotion(prepared, emotions)

    print("\nTime per emotion (seconds):")
    for emotion in emotions:
        print(f"  {emotion}: {time_per_emotion.get(emotion, 0):.2f}s")

    print("\nParticipants per emotion:")
    for emotion in emotions:
        print(f"  {emotion}: {participants_per_emotion.get(emotion, 0)} unique video_ids")

    if out_time:
        plot_emotion_histogram_time(time_per_emotion, emotions, out_time)

    if out_participants:
        plot_emotion_histogram_participants(participants_per_emotion, emotions, out_participants)

    if not out_time and not out_participants:
        print(
            "No output paths specified (all skipped). Nothing to do.", file=sys.stderr
        )
        sys.exit(0)


if __name__ == "__main__":
    main()