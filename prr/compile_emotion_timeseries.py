"""
compile_emotion_timeseries.py
=============================
Script to compile emotion timeseries data from annotated CSV files into HDF5 and CSV files.

This script:
1. Reads the cyclesix_owl.csv metadata file
2. Identifies valid videos (those with ROI defined and annotated CSV files existing)
3. Excludes specified test videos
4. Compiles all emotion timeseries data into:
   - A single HDF5 file with structured groups
   - A single large CSV file with all videos concatenated
"""

import os
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def validate_and_get_files(
    csv_path: str,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Scan the metadata CSV and identify valid video sets.

    A valid video set must have:
    - Valid filepath
    - Non-empty ROI definition
    - Existing annotated CSV file
    - Not matching any exclude patterns

    Parameters
    ----------
    csv_path : str
        Path to the cyclesix_owl.csv metadata file.
    exclude_patterns : list[str] or None, optional
        List of filename patterns to exclude. If a video_id contains any of these
        patterns, it will be skipped. Defaults to None (no exclusions).

    Returns
    -------
    list[dict]
        List of valid video information dictionaries containing:
        - video_id: base filename without extension
        - filepath: path to original video
        - roi: ROI string (x1,y1,x2,y2)
        - csv_path: path to annotated CSV file
        - readable_date: date from metadata
        - readable_time: time from metadata
    """
    if exclude_patterns is None:
        exclude_patterns = []

    df = pd.read_csv(csv_path)
    valid_files = []

    print("Scanning for valid video sets...")
    for idx, row in df.iterrows():
        try:
            input_video = row["filepath"]
            roi_str = row["ROI"]

            # Skip rows with missing filepath
            if pd.isna(input_video) or not isinstance(input_video, str):
                continue

            # Skip rows with missing or NaN ROI
            if pd.isna(roi_str) or not isinstance(roi_str, str) or roi_str.strip() == "":
                continue

            # Get output directory (same as input directory)
            input_dir = os.path.dirname(input_video)
            input_filename = os.path.basename(input_video)
            base_name = os.path.splitext(input_filename)[0]

            # Check if video should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in base_name:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Construct CSV path
            csv_filename = f"{base_name}_annot.csv"
            csv_path = os.path.join(input_dir, csv_filename)

            # Skip if CSV file doesn't exist
            if not os.path.exists(csv_path):
                continue

            # Add to valid list
            valid_files.append({
                "video_id": base_name,
                "filepath": input_video,
                "roi": roi_str,
                "csv_path": csv_path,
                "readable_date": row.get("readable_date", ""),
                "readable_time": row.get("readable_time", ""),
                "frames": row.get("frames", 0),
                "duration": row.get("duration", 0.0),
            })

        except Exception as e:
            continue

    print(f"Found {len(valid_files)} valid video sets with annotations\n")
    return valid_files


def compile_to_hdf5(valid_files: list[dict], output_h5_path: str) -> None:
    """
    Compile emotion timeseries data from all valid videos into a single HDF5 file.

    Parameters
    ----------
    valid_files : list[dict]
        List of valid file information from validate_and_get_files().
    output_h5_path : str
        Path where the HDF5 file will be written.
    """

    with h5py.File(output_h5_path, "w") as h5file:
        # Create root metadata group
        metadata_group = h5file.create_group("metadata")

        # Store list of video IDs
        video_ids = [f["video_id"] for f in valid_files]
        metadata_group.create_dataset("video_ids", data=video_ids, dtype=h5py.string_dtype())

        # Create emotion data group
        emotion_group = h5file.create_group("emotions")

        # Process each valid file
        for file_info in tqdm(valid_files, desc="Compiling HDF5", unit="video"):
            try:
                video_id = file_info["video_id"]
                csv_path = file_info["csv_path"]

                # Read the annotated CSV
                df = pd.read_csv(csv_path)

                # Create a group for this video
                video_group = emotion_group.create_group(video_id)

                # Store metadata
                video_group.attrs["filepath"] = file_info["filepath"]
                video_group.attrs["roi"] = file_info["roi"]
                video_group.attrs["readable_date"] = file_info["readable_date"]
                video_group.attrs["readable_time"] = file_info["readable_time"]
                video_group.attrs["frames"] = file_info["frames"]
                video_group.attrs["duration"] = file_info["duration"]

                # Extract emotion columns
                emotion_cols = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

                # Store each emotion as a separate dataset
                for emotion in emotion_cols:
                    if emotion in df.columns:
                        data = df[emotion].values.astype(np.float32)
                        video_group.create_dataset(emotion, data=data, compression="gzip")

                # Store dominant emotion
                if "dominant_emotion" in df.columns:
                    dominant = df["dominant_emotion"].values
                    video_group.create_dataset(
                        "dominant_emotion",
                        data=dominant,
                        dtype=h5py.string_dtype()
                    )

                # Store timestamp
                if "timestamp_ms" in df.columns:
                    timestamp = df["timestamp_ms"].values.astype(np.float32)
                    video_group.create_dataset("timestamp_ms", data=timestamp, compression="gzip")

                # Store frame index
                if "frame_index" in df.columns:
                    frame_idx = df["frame_index"].values.astype(np.int32)
                    video_group.create_dataset("frame_index", data=frame_idx, compression="gzip")

                # Store number of frames
                video_group.attrs["n_frames"] = len(df)

            except Exception as e:
                print(f"  Warning: Error processing {file_info['video_id']} for HDF5: {e}")
                continue

        # Store compilation timestamp
        metadata_group.attrs["compiled_at"] = pd.Timestamp.now().isoformat()
        metadata_group.attrs["n_videos"] = len(valid_files)


def compile_to_csv(valid_files: list[dict], output_csv_path: str) -> None:
    """
    Compile emotion timeseries data from all valid videos into a single large CSV file.

    Parameters
    ----------
    valid_files : list[dict]
        List of valid file information from validate_and_get_files().
    output_csv_path : str
        Path where the CSV file will be written.
    """

    all_data = []

    # Process each valid file
    for file_info in tqdm(valid_files, desc="Compiling CSV", unit="video"):
        try:
            video_id = file_info["video_id"]
            csv_path = file_info["csv_path"]

            # Read the annotated CSV
            df = pd.read_csv(csv_path)

            # Add video metadata columns
            df.insert(0, "video_id", video_id)
            df.insert(1, "filepath", file_info["filepath"])
            df.insert(2, "roi", file_info["roi"])
            df.insert(3, "readable_date", file_info["readable_date"])
            df.insert(4, "readable_time", file_info["readable_time"])
            df.insert(5, "duration_s", file_info["duration"])

            all_data.append(df)

        except Exception as e:
            print(f"  Warning: Error processing {file_info['video_id']} for CSV: {e}")
            continue

    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    combined_df.to_csv(output_csv_path, index=False)


def main():
    """Main execution function."""

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "cyclesix_owl.csv")
    output_h5_path = os.path.join(script_dir, "cyclesix_owl_emotions.h5")
    output_csv_path = os.path.join(script_dir, "cyclesix_owl_emotions.csv")

    # Define exclude patterns (filenames to skip)
    exclude_patterns = [
        "2026-02-02_14-17-51-842263",
        "2026-02-02_14-27-30-186276",
    ]

    # Check if input CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Validate and get valid files
    valid_files = validate_and_get_files(csv_path, exclude_patterns=exclude_patterns)

    if not valid_files:
        print("No valid video sets found. Exiting.")
        return

    # Print summary
    print("Valid videos found:")
    for f in valid_files:
        print(f"  - {f['video_id']}")
        print(f"    ROI: {f['roi']}, Frames: {f['frames']}, Duration: {f['duration']}s")
    print()

    # Compile to HDF5
    print(f"Compiling to HDF5: {output_h5_path}")
    compile_to_hdf5(valid_files, output_h5_path)
    print(f"✓ HDF5 file created")
    print(f"  File size: {os.path.getsize(output_h5_path) / (1024**2):.2f} MB\n")

    # Compile to CSV
    print(f"Compiling to CSV: {output_csv_path}")
    compile_to_csv(valid_files, output_csv_path)
    print(f"✓ CSV file created")
    print(f"  File size: {os.path.getsize(output_csv_path) / (1024**2):.2f} MB\n")

    print(f"✓ Successfully compiled {len(valid_files)} videos to both formats")


if __name__ == "__main__":
    main()
