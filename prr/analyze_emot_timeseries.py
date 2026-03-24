#!/usr/bin/env python3
"""
Stacked and combined emotion timeseries (mean ± std for stacked, mean-only for combined)

This script reads a CSV of frame-level emotion predictions and produces:
 - A vertically stacked set of subplots (one per emotion) where each subplot
   shows the mean timecourse across videos and a shaded band indicating ±1
   standard deviation (using seaborn's `errorbar='sd'`).
 - A single combined plot containing only the mean trace per emotion (no sd
   bands) so it's easier to read when multiple emotions are overlaid.

New features:
 - Optional per-emotion color mapping via CLI `--colors` argument.
   Format: "emotion1=color1,emotion2=color2,..."
   Example:
     --colors "neutral=gray,angry=#8B0000,disgust=violet,sad=#00008B,happy=orange,surprise=yellow,fear=green"
 - `--use-default-colors` flag applies a built-in mapping for canonical emotions:
     angry: dark red
     disgust: violet
     fear: green
     happy: orange
     sad: dark blue
     surprise: yellow
     neutral: gray
   You can still override individual colors with `--colors`.

Behavior:
 - Expects at least these columns in the CSV: `video_id`, `timestamp_ms`.
 - Auto-detects numeric emotion columns (prefers canonical order:
   angry, disgust, fear, happy, sad, surprise, neutral).
 - Converts timestamps from milliseconds to seconds (`timestamp_s`) for the x axis.
 - Stacked subplots share the time axis and y-limits are forced to [0, 100].
 - Combined plot contains only the mean traces (estimator='mean', errorbar=None).

Usage:
    python orchestra_analysis/prr/analyze_emot_timeseries.py \
        --csv orchestra_analysis/cyclesix_owl_emotions.csv \
        --out-stacked stacked.png \
        --out-combined combined.png \
        --use-default-colors

    python orchestra_analysis/prr/analyze_emot_timeseries.py \
        --csv orchestra_analysis/cyclesix_owl_emotions.csv \
        --out-combined combined.png \
        --colors "neutral=gray,angry=#8B0000,disgust=violet"

Dependencies:
    pandas, numpy, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
import math
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
DEFAULT_OUT_STACKED = "emotion_timeseries_sd_stacked.png"
DEFAULT_OUT_COMBINED = "emotion_timeseries_mean_combined.png"

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

# Default color mapping requested by user (can be overridden/extended via CLI)
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


def parse_colors_arg(colors_arg: Optional[str]) -> Dict[str, str]:
    """
    Parse a colors CLI argument of the form:
      "neutral=gray,angry=#8B0000,disgust=violet"
    into a dict { 'neutral': 'gray', 'angry': '#8B0000', ... }.
    """
    mapping: Dict[str, str] = {}
    if not colors_arg:
        return mapping
    # split by commas, accept spaces
    pairs = [p.strip() for p in colors_arg.split(",") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            print(
                f"Ignoring invalid color mapping '{pair}' (expected 'emotion=color')",
                file=sys.stderr,
            )
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            print(f"Ignoring invalid color mapping '{pair}'", file=sys.stderr)
            continue
        mapping[k] = v
    return mapping


def build_color_map(
    emotions: List[str], use_default: bool, colors_arg: Optional[str], palette_name: str
) -> Dict[str, str]:
    """
    Create a color map for the provided emotions.

    - If use_default is True, start from DEFAULT_COLOR_MAP for canonical emotions.
    - Parse colors_arg and override/add mappings from it.
    - For any emotion still missing a color, use the palette (by index).
    """
    color_map: Dict[str, str] = {}
    if use_default:
        # copy only entries for requested emotions
        for k, v in DEFAULT_COLOR_MAP.items():
            if k in emotions:
                color_map[k] = v

    # parse user-specified colors and override
    user_map = parse_colors_arg(colors_arg)
    for k, v in user_map.items():
        # allow users to specify colors for any emotion name
        if k in emotions:
            color_map[k] = v
        else:
            # If provided emotion not in detected list, still add it.
            color_map[k] = v

    # Finally fill missing emotions from palette
    palette = sns.color_palette(palette_name, n_colors=max(3, len(emotions)))
    for idx, e in enumerate(emotions):
        if e not in color_map:
            color_map[e] = palette[idx % len(palette)]
    return color_map


def plot_stacked_emotion_means(
    df: pd.DataFrame,
    emotions: List[str],
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    figsize: Optional[tuple] = None,
    mean_linewidth: float = 2.0,
):
    """
    Plot vertically stacked subplots (one per emotion) showing mean ± sd.

    Uses seaborn.lineplot with estimator='mean' and errorbar='sd'.
    """
    if not emotions:
        raise ValueError("No emotion columns to plot.")

    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})

    df = prepare_dataframe(df, emotions)

    n = len(emotions)
    ncols = 1
    nrows = n
    if figsize is None:
        figsize = (10, 2.4 * max(3, nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    # normalize axes to a list
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    # Default palette for fallback (also used to keep consistent ordering)
    palette = sns.color_palette("tab10", n_colors=max(3, n))

    for idx, emotion in enumerate(emotions):
        ax = axes_list[idx]
        color = None
        if color_map and emotion in color_map:
            color = color_map[emotion]
        else:
            color = palette[idx % len(palette)]

        # seaborn with errorbar='sd' shows mean with ±1 std band
        sns.lineplot(
            data=df,
            x="timestamp_s",
            y=emotion,
            estimator="mean",
            errorbar="sd",
            ax=ax,
            color=color,
            linewidth=mean_linewidth,
        )

        # Force y-limits to [0, 100]
        ax.set_ylim(0, 100)

        ax.set_title(emotion)
        ax.set_ylabel("score")
        # show x-label only on bottom axis
        if idx == n - 1:
            ax.set_xlabel("time (s)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    # hide any extra axes (just in case)
    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle("Emotion mean timeseries (±1 std) — stacked vertically", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved stacked plot to: {out_path}")


def plot_combined_emotions(
    df: pd.DataFrame,
    emotions: List[str],
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    figsize: Optional[tuple] = None,
    mean_linewidth: float = 2.0,
):
    """
    Plot a single combined plot with only the mean trace per emotion (no sd bands).

    Each emotion is a different color; the mean is computed with estimator='mean'
    and shading is disabled via errorbar=None so the plot remains readable.
    """
    if not emotions:
        raise ValueError("No emotion columns to plot.")

    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})

    df = prepare_dataframe(df, emotions)

    # melt to long form: timestamp_s, emotion, value, video_id
    df_long = df[["timestamp_s", "video_id"] + emotions].melt(
        id_vars=["timestamp_s", "video_id"],
        value_vars=emotions,
        var_name="emotion",
        value_name="value",
    )
    df_long = df_long.dropna(subset=["value"])
    if df_long.empty:
        raise ValueError("No data available for combined plot after melting.")

    if figsize is None:
        figsize = (12, 5)

    fig, ax = plt.subplots(figsize=figsize)

    # If user provided a color_map, construct palette mapping appropriate for seaborn
    if color_map:
        # restrict to emotions present and preserve their mapping
        palette_dict = {e: color_map[e] for e in emotions if e in color_map}
    else:
        palette_dict = None

    # Draw only mean traces, no sd bands: errorbar=None
    sns.lineplot(
        data=df_long,
        x="timestamp_s",
        y="value",
        hue="emotion",
        estimator="mean",
        errorbar=None,
        ax=ax,
        linewidth=mean_linewidth,
        palette=palette_dict,
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title("All emotions — mean timeseries (no sd bands)")
    ax.legend(title="emotion", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.88, 1.0])  # leave room for legend
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined plot to: {out_path}")


def plot_combined_emotions_se(
    df: pd.DataFrame,
    emotions: List[str],
    out_path: Path,
    color_map: Optional[dict] = None,
    figsize: Optional[tuple] = None,
    mean_linewidth: float = 1.0,
    se_alpha: float = 0.12,
):
    """
    Plot a single combined plot with mean traces for all emotions and a light shaded
    region showing the standard error of the mean (SEM) per timestamp.

    - df: prepared dataframe (must contain `timestamp_s` and emotion columns)
    - emotions: list of emotion column names
    - color_map: optional dict mapping emotion -> color
    - mean_linewidth: thinner mean line width for combined plot
    - se_alpha: alpha for the SEM shading (very light)
    """
    if not emotions:
        raise ValueError("No emotion columns to plot.")

    sns.set(style="whitegrid", rc={"figure.facecolor": "white"})

    df = prepare_dataframe(df, emotions)

    # melt to long form
    df_long = df[["timestamp_s", "video_id"] + emotions].melt(
        id_vars=["timestamp_s", "video_id"],
        value_vars=emotions,
        var_name="emotion",
        value_name="value",
    )
    df_long = df_long.dropna(subset=["value"])
    if df_long.empty:
        raise ValueError("No data available for combined SE plot after melting.")

    # Compute mean and SEM for each emotion at each timestamp
    agg = (
        df_long.groupby(["timestamp_s", "emotion"], observed=True)["value"]
        .agg(
            mean="mean",
            sem=lambda x: x.std(ddof=1) / (len(x) ** 0.5) if len(x) > 0 else np.nan,
        )
        .reset_index()
    )

    # Pivot so we can iterate consistently
    mean_pivot = agg.pivot(index="timestamp_s", columns="emotion", values="mean")
    sem_pivot = agg.pivot(index="timestamp_s", columns="emotion", values="sem")

    if figsize is None:
        figsize = (12, 5)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, emotion in enumerate(emotions):
        if emotion not in mean_pivot.columns:
            continue
        x = mean_pivot.index.values
        y = mean_pivot[emotion].values
        se = sem_pivot.get(emotion)
        # sem_pivot may be a DataFrame; use .values if present
        if se is None:
            se_vals = np.zeros_like(y)
        else:
            se_vals = sem_pivot[emotion].values
            # replace NaN with 0 for plotting
            se_vals = np.nan_to_num(se_vals, nan=0.0)

        # choose color
        col = None
        if color_map and emotion in color_map:
            col = color_map[emotion]
        else:
            # fallback to seaborn palette if no mapping provided
            col = sns.color_palette("tab10", n_colors=max(3, len(emotions)))[
                idx % max(1, len(emotions))
            ]

        # plot mean line (thinner)
        ax.plot(x, y, color=col, linewidth=mean_linewidth, label=emotion)

        # light shaded SEM band
        lower = y - se_vals
        upper = y + se_vals
        ax.fill_between(x, lower, upper, color=col, alpha=se_alpha, linewidth=0)

    ax.set_ylim(0, 100)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title("All emotions — mean timeseries (mean ± SEM)")
    ax.legend(title="emotion", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.88, 1.0])  # leave room for legend
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined mean+SEM plot to: {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Create stacked and/or combined emotion mean timeseries plots."
    )
    p.add_argument("--csv", "-c", default=DEFAULT_CSV, help="Path to input CSV")
    p.add_argument(
        "--out-stacked",
        default=DEFAULT_OUT_STACKED,
        help="Output image path for stacked subplots (PNG). Set empty to skip.",
    )
    p.add_argument(
        "--out-combined",
        default=DEFAULT_OUT_COMBINED,
        help="Output image path for combined single plot (PNG). Set empty to skip.",
    )
    p.add_argument(
        "--out-combined-se",
        default="",
        help="Output image path for combined mean+SEM plot (PNG). Set empty to skip.",
    )
    p.add_argument(
        "--emotions",
        "-e",
        nargs="*",
        default=None,
        help="Explicit list of emotion columns to plot (default: auto-detect).",
    )
    p.add_argument(
        "--palette",
        default="tab10",
        help="Seaborn/matplotlib palette name to use for emotion colors (default: tab10).",
    )
    p.add_argument(
        "--use-default-colors",
        action="store_true",
        help="Use the built-in default color mapping for canonical emotions (can be overridden with --colors).",
    )
    p.add_argument(
        "--colors",
        type=str,
        default="",
        help=(
            "Custom per-emotion colors. Format: 'neutral=gray,angry=#8B0000,disgust=violet'. "
            "These override the defaults from --use-default-colors and fill missing ones."
        ),
    )
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    out_stacked = Path(args.out_stacked) if args.out_stacked else None
    out_combined = Path(args.out_combined) if args.out_combined else None
    out_combined_se = Path(args.out_combined_se) if args.out_combined_se else None

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
        f"Loaded {len(df):,} rows from {csv_path}; {df['video_id'].nunique():,} unique videos; plotting {len(emotions)} emotions."
    )

    try:
        prepared = prepare_dataframe(df, emotions)
    except Exception as exc:
        print(f"Error preparing dataframe: {exc}", file=sys.stderr)
        sys.exit(4)

    # Build color map from defaults, user overrides, and palette fallback
    color_map = build_color_map(
        emotions,
        use_default=args.use_default_colors,
        colors_arg=args.colors,
        palette_name=args.palette,
    )
    # Print the color map for user's awareness
    print("Using color map (emotion -> color):")
    for k in emotions:
        print(f"  {k}: {color_map.get(k)}")

    if out_stacked:
        plot_stacked_emotion_means(
            prepared, emotions, out_stacked, color_map=color_map, mean_linewidth=2.0
        )

    if out_combined:
        plot_combined_emotions(
            prepared, emotions, out_combined, color_map=color_map, mean_linewidth=2.0
        )
    if out_combined_se:
        plot_combined_emotions_se(
            prepared,
            emotions,
            out_combined_se,
            color_map=color_map,
            mean_linewidth=0.6,
            se_alpha=0.20,
        )

    if not out_stacked and not out_combined and not out_combined_se:
        print(
            "No output paths specified (all skipped). Nothing to do.", file=sys.stderr
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
