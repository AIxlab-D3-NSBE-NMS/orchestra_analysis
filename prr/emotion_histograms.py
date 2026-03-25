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
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
DEFAULT_WINDOW_LENGTH_S = 0.5  # seconds
DEFAULT_WINDOW_OVERLAP = 0.5  # 50% overlap
DEFAULT_CONFIDENCE_THRESHOLD = 95.0  # percent (0-100)

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


def apply_sliding_window(
    df: pd.DataFrame,
    emotions: List[str],
    window_length_s: float = DEFAULT_WINDOW_LENGTH_S,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply a sliding time window to aggregate emotions over time windows.
    
    For each participant (video_id), slides a window of specified length
    with specified overlap percentage. Within each window, finds the emotion
    with the highest mean score and records that as the dominant emotion.
    
    Duration per window is calculated as the actual time span from first to last
    frame in the window (actual_end_time - actual_start_time), not the nominal
    window_length_s. This accounts for actual frame sampling rates.
    
    Windows with average emotion scores below the confidence_threshold are filtered out.
    
    Args:
        df: Prepared dataframe with timestamp_s and emotion columns
        emotions: List of emotion column names
        window_length_s: Length of sliding window in seconds (default: 0.5)
        window_overlap: Overlap as fraction 0-1 (default: 0.5 for 50%)
        confidence_threshold: Minimum average emotion score (0-100) for window inclusion (default: 95.0)
    
    Returns:
        DataFrame with windowed data: timestamp_s, video_id, dominant emotion, window_duration_s, confidence
    """
    # Ensure data is sorted by video_id and timestamp
    df = df.sort_values(by=["video_id", "timestamp_s"]).reset_index(drop=True)
    
    windowed_rows = []
    unique_videos = df["video_id"].unique()
    
    # Process each participant separately with progress bar
    for video_id in tqdm(unique_videos, desc="Applying sliding window", unit="participant"):
        video_df = df[df["video_id"] == video_id].reset_index(drop=True)
        
        if len(video_df) == 0:
            continue
        
        # Calculate window parameters
        start_time = video_df["timestamp_s"].min()
        end_time = video_df["timestamp_s"].max()
        step_size = window_length_s * (1 - window_overlap)
        
        # Create windows
        window_starts = []
        current_start = start_time
        while current_start <= end_time:
            window_starts.append(current_start)
            current_start += step_size
        
        # Process each window
        for window_start in window_starts:
            window_end = window_start + window_length_s
            
            # Get frames within this window
            window_frames = video_df[
                (video_df["timestamp_s"] >= window_start)
                & (video_df["timestamp_s"] < window_end)
            ]
            
            # Skip if no frames in window or only NaN values
            if len(window_frames) == 0:
                continue
            
            # Calculate mean emotion scores in this window
            emotion_means = {}
            for emotion in emotions:
                valid_scores = window_frames[emotion].dropna()
                if len(valid_scores) > 0:
                    emotion_means[emotion] = valid_scores.mean()
                else:
                    emotion_means[emotion] = 0.0
            
            # Find dominant emotion (highest mean score)
            if max(emotion_means.values()) > 0:
                dominant_emotion = max(emotion_means, key=emotion_means.get)
                dominant_score = emotion_means[dominant_emotion]
                
                # Apply confidence threshold: skip windows with low average emotion scores
                if dominant_score < confidence_threshold:
                    continue
                
                # Calculate actual duration: time span from first to last frame in window
                actual_start_time = window_frames["timestamp_s"].min()
                actual_end_time = window_frames["timestamp_s"].max()
                # If only one frame, use a reasonable estimate (e.g., 1/30 fps ≈ 0.033s)
                # Otherwise use actual span
                if actual_start_time == actual_end_time:
                    window_duration = 1.0 / 30.0  # Assume 30 fps frame duration
                else:
                    window_duration = actual_end_time - actual_start_time
                
                # Use window center as timestamp
                window_center = (window_start + window_end) / 2.0
                windowed_rows.append({
                    "video_id": video_id,
                    "timestamp_s": window_center,
                    "dominant": dominant_emotion,
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_duration": window_duration,
                    "confidence": dominant_score,
                })
    
    windowed_df = pd.DataFrame(windowed_rows)
    
    if windowed_df.empty:
        raise ValueError("No windows generated from the data. Check window_length_s and overlap parameters.")
    
    return windowed_df


def calculate_time_per_emotion(
    df: pd.DataFrame,
    emotions: List[str],
    window_length_s: Optional[float] = None,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate total time spent in each emotion.
    
    If window_length_s is provided, applies sliding window aggregation first.
    Otherwise, uses frame-by-frame dominant emotion (noisy).
    
    Args:
        df: Prepared dataframe with timestamp_s and emotion columns
        emotions: List of emotion column names
        window_length_s: Optional sliding window length in seconds. If None, uses frame-by-frame.
        window_overlap: Overlap as fraction 0-1 (default: 0.5 for 50%)
        confidence_threshold: Minimum emotion score (0-100) for window inclusion (default: 95.0)
    
    Returns:
        Dictionary mapping emotion -> total time in seconds
    """
    df_copy = df.copy()
    
    # Apply sliding window if specified
    if window_length_s is not None and window_length_s > 0:
        df_copy = apply_sliding_window(df_copy, emotions, window_length_s, window_overlap, confidence_threshold)
        
        # Use actual window duration calculated from frame spans
        df_copy["frame_duration"] = df_copy["window_duration"]
    else:
        # Frame-by-frame approach
        df_copy["dominant"] = df_copy[emotions].idxmax(axis=1)
        
        # Calculate frame duration (time to next frame)
        df_copy = df_copy.sort_values(by=["video_id", "timestamp_s"])
        df_copy["next_timestamp"] = df_copy.groupby("video_id")["timestamp_s"].shift(-1)
        df_copy["frame_duration"] = df_copy["next_timestamp"] - df_copy["timestamp_s"]
        
        # Fill NaN durations (last frame of each video) with 0
        df_copy["frame_duration"] = df_copy["frame_duration"].fillna(0)
    
    # Sum frame/window durations by dominant emotion
    time_per_emotion = df_copy.groupby("dominant")["frame_duration"].sum().to_dict()
    
    # Ensure all emotions are present (even if 0)
    for emotion in emotions:
        if emotion not in time_per_emotion:
            time_per_emotion[emotion] = 0.0
    
    return time_per_emotion


def calculate_participants_per_emotion(
    df: pd.DataFrame,
    emotions: List[str],
    window_length_s: Optional[float] = None,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, int]:
    """
    Calculate number of unique participants (video_id) per emotion.
    
    If window_length_s is provided, applies sliding window aggregation first.
    Otherwise, counts participants frame-by-frame.
    
    A participant is counted for an emotion if there's at least one frame/window
    where that emotion is dominant.
    
    Args:
        df: Prepared dataframe with timestamp_s and emotion columns
        emotions: List of emotion column names
        window_length_s: Optional sliding window length in seconds. If None, uses frame-by-frame.
        window_overlap: Overlap as fraction 0-1 (default: 0.5 for 50%)
        confidence_threshold: Minimum emotion score (0-100) for window inclusion (default: 95.0)
    
    Returns:
        Dictionary mapping emotion -> number of unique participants
    """
    df_copy = df.copy()
    
    # Apply sliding window if specified
    if window_length_s is not None and window_length_s > 0:
        df_copy = apply_sliding_window(df_copy, emotions, window_length_s, window_overlap, confidence_threshold)
        participants_per_emotion = (
            df_copy.groupby("dominant")["video_id"].nunique().to_dict()
        )
    else:
        # Frame-by-frame approach
        df_copy["dominant"] = df_copy[emotions].idxmax(axis=1)
        participants_per_emotion = (
            df_copy.groupby("dominant")["video_id"].nunique().to_dict()
        )
    
    # Ensure all emotions are present (even if 0)
    for emotion in emotions:
        if emotion not in participants_per_emotion:
            participants_per_emotion[emotion] = 0
    
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
    p.add_argument(
        "--window-length-s",
        type=float,
        default=None,
        help=f"Sliding window length in seconds (default: None = frame-by-frame). Example: {DEFAULT_WINDOW_LENGTH_S}",
    )
    p.add_argument(
        "--window-overlap",
        type=float,
        default=DEFAULT_WINDOW_OVERLAP,
        help=f"Window overlap as fraction 0-1 (default: {DEFAULT_WINDOW_OVERLAP} = 50%%). Example: 0.5",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Minimum dominant emotion score (0-100) for window inclusion (default: {DEFAULT_CONFIDENCE_THRESHOLD}%%). Filters low-confidence predictions.",
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

    total_videos = df['video_id'].nunique()
    print(
        f"Loaded {len(df):,} rows from {csv_path}; {total_videos:,} unique participants; analyzing {len(emotions)} emotions."
    )
    
    if args.window_length_s is not None:
        print(f"Using sliding window: length={args.window_length_s}s, overlap={args.window_overlap*100:.0f}%, confidence_threshold={args.confidence_threshold:.1f}%")
    else:
        print(f"Using frame-by-frame analysis with confidence_threshold={args.confidence_threshold:.1f}%")

    try:
        prepared = prepare_dataframe(df, emotions)
    except Exception as exc:
        print(f"Error preparing dataframe: {exc}", file=sys.stderr)
        sys.exit(4)

    # Calculate metrics
    print("\nCalculating metrics...")
    time_per_emotion = calculate_time_per_emotion(
        prepared, emotions, window_length_s=args.window_length_s, window_overlap=args.window_overlap, confidence_threshold=args.confidence_threshold
    )
    participants_per_emotion = calculate_participants_per_emotion(
        prepared, emotions, window_length_s=args.window_length_s, window_overlap=args.window_overlap, confidence_threshold=args.confidence_threshold
    )

    print("\nTime per emotion (seconds):")
    for emotion in emotions:
        print(f"  {emotion}: {time_per_emotion.get(emotion, 0):.2f}s")

    print("\nParticipants per emotion:")
    for emotion in emotions:
        print(f"  {emotion}: {participants_per_emotion.get(emotion, 0)} unique video_ids")

    print("\nGenerating plots...")
    if out_time:
        plot_emotion_histogram_time(time_per_emotion, emotions, out_time)

    if out_participants:
        plot_emotion_histogram_participants(participants_per_emotion, emotions, out_participants)

    if not out_time and not out_participants:
        print(
            "No output paths specified (all skipped). Nothing to do.", file=sys.stderr
        )
        sys.exit(0)
    
    print("\nDone!")


if __name__ == "__main__":
    main()