import cv2
from deepface import DeepFace
from typing import Iterator, List, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd

# ── Frame sources ─────────────────────────────────────────────────────────────

def frames_from_video(video_path: str, frame_skip: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_ms, frame) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing frames")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_skip == 0:
                yield frame_index, cap.get(cv2.CAP_PROP_POS_MSEC), frame
                progress_bar.update(frame_skip)
            frame_index += 1
    finally:
        cap.release()


def frames_from_stream(stream_url: str | int, frame_skip: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield (frame_index, timestamp_ms, frame) from a live stream or webcam.

    Args:
        stream_url: RTSP/HTTP URL, or an integer device index (0 = default webcam).
        frame_skip: Process every Nth frame.
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open stream: {stream_url}")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_skip == 0:
                yield frame_index, cap.get(cv2.CAP_PROP_POS_MSEC), frame
            frame_index += 1
    finally:
        cap.release()


def frames_from_numpy(
    frames: Iterator[np.ndarray], frame_skip: int = 1
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Adapt any iterator of raw numpy frames into the standard (index, ms, frame) format.
    timestamp_ms is estimated from frame index (no real clock).
    """
    for frame_index, frame in enumerate(frames):
        if frame_index % frame_skip == 0:
            yield frame_index, float(frame_index), frame


# ── Detection ─────────────────────────────────────────────────────────────────

def analyze_frames(
    frame_source: Iterator[tuple[int, float, np.ndarray]],
) -> List[Dict]:
    """
    Consume any frame source and run DeepFace face + emotion detection on each frame.

    Returns:
        - A list of dicts, one per frame.
            - Each dict contains the metrics for that frame.
    """
    results = []

    for frame_index, timestamp_ms, frame in frame_source:
        try:
            detections = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",#"retinaface",
                enforce_detection=True,  # raises if no face found
            )

            # analyze() returns a list when multiple faces are present
            if isinstance(detections, dict):
                detections = [detections]

            faces = [
                {
                    "facial_area": d["region"],
                    "confidence": d.get("face_confidence", 1.0),
                    "emotions": d["emotion"],            # all probabilities
                    "dominant_emotion": d["dominant_emotion"],
                }
                for d in detections
            ]
            face_detected = True

        except ValueError:
            # enforce_detection=True raises ValueError when no face is found
            faces = []
            face_detected = False

        except Exception as e:
            print(f"[Frame {frame_index}] Unexpected DeepFace error: {e}")
            faces = []
            face_detected = False

        result = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "face_detected": face_detected,
            "faces": faces,
        }
        results.append(result)

    return results


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    source_arg = sys.argv[1] if len(sys.argv) > 1 else "0"

    # Auto-detect: integer → webcam, otherwise file/URL
    source = int(source_arg) if source_arg.isdigit() else source_arg
    frame_source = frames_from_stream(source, frame_skip=1) if isinstance(source, int) \
        else frames_from_video(source, frame_skip=1)

    results = list(analyze_frames(frame_source))

    # Define the output CSV file path
    output_csv_path = Path("frame_metrics.csv")

    # Create a DataFrame with each emotion in its own column
    emotions_columns = ["frame_index", "timestamp_ms", "face_detected"]
    for emotion in set(emotion for frame_result in results for face in frame_result.get("faces", []) for emotion in face["emotions"]):
        emotions_columns.append(emotion)

    df_results = pd.DataFrame(columns=emotions_columns)

    # Populate the DataFrame
    rows_to_concat = []
    for result in results:
        row_data = {col: None for col in emotions_columns}
        row_data["frame_index"] = result.get("frame_index")
        row_data["timestamp_ms"] = result.get("timestamp_ms")
        row_data["face_detected"] = result.get("face_detected")

        if "faces" in result:
            for face in result["faces"]:
                for emotion, prob in face["emotions"].items():
                    if emotion not in row_data:
                        print(f"Unexpected emotion column: {emotion}")
                        continue
                    row_data[emotion] = prob

        rows_to_concat.append(row_data)

    # Use pd.concat to add the rows to the DataFrame
    df_results = pd.concat([df_results, pd.DataFrame(rows_to_concat)], ignore_index=True)

    # Write results to CSV file using pandas
    df_results.to_csv(output_csv_path, index=False)

    print(f"Frame metrics saved to {output_csv_path}")
