"""
get_emot_timeseries.py
======================
Utilities for per-frame emotion detection using DeepFace, with optional
cropped-ROI support and fully annotated image / video output.
"""

import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def read_image(fullpath: str) -> np.ndarray:
    """Read an image file and return it as an RGB numpy array.

    Parameters
    ----------
    fullpath : str
        Path to the image file (any format supported by PIL).

    Returns
    -------
    np.ndarray
        Image as an ``(H, W, 3)`` RGB uint8 array.
    """
    img = Image.open(fullpath)
    return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Core emotion detection
# ---------------------------------------------------------------------------

def get_emotions_from_frame(
    img: np.ndarray,
    crop: tuple | None = None,
    detector_backend: str = "retinaface",
) -> list[dict]:
    """Detect faces and analyse emotions in a single RGB frame.

    Parameters
    ----------
    img : np.ndarray
        Full-frame image as an RGB uint8 numpy array.
    crop : tuple or None, optional
        Region of interest expressed as ``(x, y, w, h)`` in pixel coordinates
        of *img*.  When provided, only the cropped sub-image is passed to
        DeepFace.  The facial-region coordinates returned by DeepFace are then
        translated back into the original image's coordinate space by adding
        the ``x`` and ``y`` offsets to every face's ``region``.
    detector_backend : str, optional
        DeepFace detector backend.  Defaults to ``'retinaface'``.

    Returns
    -------
    list[dict]
        A list of per-face result dicts exactly as returned by
        ``DeepFace.analyze``, except that ``region`` coordinates are in the
        original image's coordinate space when *crop* is used.
    """
    if crop is not None:
        cx, cy, cw, ch = crop
        sub_img = img[cy : cy + ch, cx : cx + cw]
    else:
        cx, cy = 0, 0
        sub_img = img

    results = DeepFace.analyze(
        sub_img,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend=detector_backend,
        align=True,
        expand_percentage=0,
        silent=False,
        anti_spoofing=False,
    )

    # DeepFace always returns a list when actions is a list; guard anyway.
    if isinstance(results, dict):
        results = [results]

    # Translate face-region coordinates back to original image space.
    if crop is not None:
        for face in results:
            region = face.get("region", {})
            region["x"] = region.get("x", 0) + cx
            region["y"] = region.get("y", 0) + cy
            face["region"] = region

    return results


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_dashed_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple = (200, 80, 0),
    thickness: int = 2,
    dash_len: int = 12,
    gap_len: int = 6,
) -> None:
    """Draw a dashed rectangle on *canvas* in-place (expects BGR).

    Parameters
    ----------
    canvas : np.ndarray
        BGR image to draw on (modified in-place).
    x, y, w, h : int
        Rectangle position and size in pixels.
    color : tuple
        BGR colour triple.
    thickness : int
        Line thickness in pixels.
    dash_len : int
        Length of each drawn dash segment in pixels.
    gap_len : int
        Length of each gap between dashes in pixels.
    """
    step = dash_len + gap_len

    def _h_dashes(x0: int, x1: int, row: int) -> None:
        pos = x0
        while pos < x1:
            end = min(pos + dash_len, x1)
            cv2.line(canvas, (pos, row), (end, row), color, thickness)
            pos += step

    def _v_dashes(y0: int, y1: int, col: int) -> None:
        pos = y0
        while pos < y1:
            end = min(pos + dash_len, y1)
            cv2.line(canvas, (col, pos), (col, end), color, thickness)
            pos += step

    _h_dashes(x, x + w, y)        # top edge
    _h_dashes(x, x + w, y + h)    # bottom edge
    _v_dashes(y, y + h, x)        # left edge
    _v_dashes(y, y + h, x + w)    # right edge


def _draw_emotion_bars(
    canvas: np.ndarray,
    emotions: dict,
    origin_x: int,
    origin_y: int,
    bar_max_width: int = 80,
) -> None:
    """Render a compact table of emotion probability bars on *canvas* in-place (BGR).

    Each row displays a filled rectangle whose width is proportional to the
    probability, followed by a short emotion label and the numeric percentage.

    Parameters
    ----------
    canvas : np.ndarray
        BGR image to draw on (modified in-place).
    emotions : dict
        Mapping ``{emotion_name: probability}`` as returned by DeepFace
        (probability values are in the 0-100 range).
    origin_x, origin_y : int
        Top-left pixel where the bar table begins.
    bar_max_width : int
        Pixel width that represents 100 % probability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    row_h = 13
    text_color = (255, 255, 255)
    bar_color = (0, 200, 80)
    bg_color = (30, 30, 30)

    for i, (emotion, prob) in enumerate(emotions.items()):
        row_y = origin_y + i * row_h
        bar_w = int(bar_max_width * max(0.0, min(prob, 100.0)) / 100.0)

        # Dark background strip for the whole row
        cv2.rectangle(
            canvas,
            (origin_x, row_y),
            (origin_x + bar_max_width + 90, row_y + row_h - 1),
            bg_color,
            -1,
        )
        # Probability bar
        if bar_w > 0:
            cv2.rectangle(
                canvas,
                (origin_x, row_y),
                (origin_x + bar_w, row_y + row_h - 1),
                bar_color,
                -1,
            )
        # Text label: abbreviated emotion name + percentage
        label = f"{emotion[:3]} {prob:5.1f}%"
        cv2.putText(
            canvas,
            label,
            (origin_x + bar_max_width + 3, row_y + row_h - 3),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Annotation API
# ---------------------------------------------------------------------------

def annotate_image(
    img: np.ndarray,
    face_results: list[dict],
    crop: tuple | None = None,
) -> np.ndarray:
    """Draw face regions, dominant emotions, and probability bars on a copy of *img*.

    The function never modifies *img*; it always works on an internal copy.

    Parameters
    ----------
    img : np.ndarray
        Original RGB image.
    face_results : list[dict]
        Output of :func:`get_emotions_from_frame`.  Each entry is a DeepFace
        result dict whose ``region`` coordinates must already be in the
        original image's coordinate space.
    crop : tuple or None, optional
        When the crop ROI ``(x, y, w, h)`` that was passed to
        :func:`get_emotions_from_frame` is provided here, a blue dashed
        rectangle is drawn around that ROI on the output image.

    Returns
    -------
    np.ndarray
        Annotated image as a BGR uint8 numpy array, ready for ``cv2.imwrite``.
    """
    # Convert RGB -> BGR for all cv2 drawing operations.
    canvas: np.ndarray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for face in face_results:
        region = face.get("region", {})
        fx = int(region.get("x", 0))
        fy = int(region.get("y", 0))
        fw = int(region.get("w", 0))
        fh = int(region.get("h", 0))

        dominant: str = face.get("dominant_emotion", "")
        emotions: dict = face.get("emotion", {})

        # Green face rectangle
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2)

        # Dominant-emotion label above the rectangle
        label_y = max(fy - 6, 16)
        (lw, lh), baseline = cv2.getTextSize(dominant, font, 0.55, 1)
        # Filled background pill for readability
        cv2.rectangle(
            canvas,
            (fx, label_y - lh - baseline - 2),
            (fx + lw + 6, label_y + baseline),
            (0, 160, 0),
            -1,
        )
        cv2.putText(
            canvas,
            dominant,
            (fx + 3, label_y),
            font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Emotion probability bars (below the box, or inside if near bottom edge)
        bars_top = fy + fh + 4
        needed_height = len(emotions) * 13
        if bars_top + needed_height > canvas.shape[0]:
            bars_top = fy + 4
        _draw_emotion_bars(canvas, emotions, fx, bars_top)

    # Optional crop ROI: blue dashed rectangle
    if crop is not None:
        cx, cy, cw, ch = crop
        _draw_dashed_rect(canvas, cx, cy, cw, ch, color=(200, 80, 0), thickness=2)

    return canvas


def annotate_image_file(
    input_path: str,
    output_path: str,
    crop: tuple | None = None,
    detector_backend: str = "retinaface",
) -> list[dict]:
    """Detect emotions in an image file and write an annotated copy to disk.

    Parameters
    ----------
    input_path : str
        Path to the source image.
    output_path : str
        Destination path for the annotated BGR image (e.g. ``'out.png'``).
    crop : tuple or None, optional
        Optional ``(x, y, w, h)`` crop region
        (see :func:`get_emotions_from_frame`).
    detector_backend : str, optional
        DeepFace detector backend.  Defaults to ``'retinaface'``.

    Returns
    -------
    list[dict]
        Raw DeepFace results (with coordinate translation applied when *crop*
        is used).
    """
    img = read_image(input_path)
    results = get_emotions_from_frame(img, crop=crop, detector_backend=detector_backend)
    annotated = annotate_image(img, results, crop=crop)
    cv2.imwrite(output_path, annotated)
    return results


def annotate_video(
    input_path: str,
    output_path: str,
    crop: tuple | None = None,
    frame_skip: int = 1,
    detector_backend: str = "retinaface",
    resolution_scale: float = 0.5,
) -> None:
    """Detect emotions on every *frame_skip*-th frame of a video and write annotated output.

    Frames that are skipped re-use the annotation from the most recently
    processed frame.  If no frame has been processed yet (i.e. the very first
    frames are skipped), the raw BGR frame is written instead.

    Parameters
    ----------
    input_path : str
        Path to the source video file.
    output_path : str
        Destination path for the annotated video (mp4v codec).
    crop : tuple or None, optional
        Optional ``(x, y, w, h)`` crop region passed to
        :func:`get_emotions_from_frame` on every processed frame.
    frame_skip : int, optional
        Process every *frame_skip*-th frame.  Must be >= 1.
        Defaults to 1 (every frame is processed).
    detector_backend : str, optional
        DeepFace detector backend.  Defaults to ``'retinaface'``.
    resolution_scale : float, optional
        Scale factor for output video resolution (default 0.5 = 50% of original).
        For example, 0.5 will reduce both width and height to 50% of the original.
        Use 1.0 for full resolution.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale output dimensions
    output_width = int(width * resolution_scale)
    output_height = int(height * resolution_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    last_annotated: np.ndarray | None = None
    frame_index: int = 0

    with tqdm(total=total_frames, desc="Annotating video", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                # DeepFace expects RGB; cv2 gives BGR - convert at the boundary.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                try:
                    results = get_emotions_from_frame(
                        frame_rgb,
                        crop=crop,
                        detector_backend=detector_backend,
                    )
                    last_annotated = annotate_image(frame_rgb, results, crop=crop)
                except Exception as exc:
                    print(f"[Frame {frame_index}] Detection error: {exc}")
                    last_annotated = frame_bgr.copy()

            # Prepare frame for writing
            frame_to_write = last_annotated if last_annotated is not None else frame_bgr

            # Scale down if needed
            if resolution_scale != 1.0:
                frame_to_write = cv2.resize(frame_to_write, (output_width, output_height))

            writer.write(frame_to_write)
            frame_index += 1
            pbar.update(1)

    cap.release()
    writer.release()


def process_video_with_timeseries(
    input_path: str,
    output_video_path: str,
    output_csv_path: str,
    crop: tuple | None = None,
    frame_skip: int = 1,
    detector_backend: str = "retinaface",
    resolution_scale: float = 0.5,
) -> None:
    """Detect emotions on every frame_skip-th frame of a video and export annotated video and CSV timeseries.

    Writes an annotated MP4 video and a CSV file containing per-frame emotion data.

    Parameters
    ----------
    input_path : str
        Path to the source video file.
    output_video_path : str
        Destination path for the annotated video (mp4v codec).
    output_csv_path : str
        Destination path for the CSV file with emotion timeseries data.
    crop : tuple or None, optional
        Optional ``(x, y, w, h)`` crop region passed to
        :func:`get_emotions_from_frame` on every processed frame.
    frame_skip : int, optional
        Process every *frame_skip*-th frame.  Must be >= 1.
        Defaults to 1 (every frame is processed).
    detector_backend : str, optional
        DeepFace detector backend.  Defaults to ``'retinaface'``.
    resolution_scale : float, optional
        Scale factor for output video resolution (default 0.5 = 50% of original).
        For example, 0.5 will reduce both width and height to 50% of the original.
        Use 1.0 for full resolution.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale output dimensions
    output_width = int(width * resolution_scale)
    output_height = int(height * resolution_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    # Collect emotion data per frame
    emotion_data = []
    last_annotated: np.ndarray | None = None
    frame_index: int = 0

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Calculate timestamp in milliseconds
            timestamp_ms = (frame_index / fps) * 1000.0 if fps > 0 else 0.0

            if frame_index % frame_skip == 0:
                # DeepFace expects RGB; cv2 gives BGR - convert at the boundary.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                try:
                    results = get_emotions_from_frame(
                        frame_rgb,
                        crop=crop,
                        detector_backend=detector_backend,
                    )
                    last_annotated = annotate_image(frame_rgb, results, crop=crop)

                    # Extract emotion data from first face (if any detected)
                    if results:
                        face = results[0]
                        dominant_emotion = face.get("dominant_emotion", "unknown")
                        emotions_dict = face.get("emotion", {})
                    else:
                        dominant_emotion = "no_face"
                        emotions_dict = {
                            "angry": 0.0,
                            "disgust": 0.0,
                            "fear": 0.0,
                            "happy": 0.0,
                            "neutral": 0.0,
                            "sad": 0.0,
                            "surprise": 0.0,
                        }

                except Exception as exc:
                    print(f"[Frame {frame_index}] Detection error: {exc}")
                    last_annotated = frame_bgr.copy()
                    dominant_emotion = "error"
                    emotions_dict = {
                        "angry": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "happy": 0.0,
                        "neutral": 0.0,
                        "sad": 0.0,
                        "surprise": 0.0,
                    }
            else:
                # For skipped frames, use the last recorded emotion data
                if emotion_data:
                    last_row = emotion_data[-1]
                    dominant_emotion = last_row["dominant_emotion"]
                    emotions_dict = {
                        "angry": last_row["angry"],
                        "disgust": last_row["disgust"],
                        "fear": last_row["fear"],
                        "happy": last_row["happy"],
                        "neutral": last_row["neutral"],
                        "sad": last_row["sad"],
                        "surprise": last_row["surprise"],
                    }
                else:
                    dominant_emotion = "no_face"
                    emotions_dict = {
                        "angry": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "happy": 0.0,
                        "neutral": 0.0,
                        "sad": 0.0,
                        "surprise": 0.0,
                    }

            # Record emotion data
            row_data = {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "dominant_emotion": dominant_emotion,
            }
            row_data.update(emotions_dict)
            emotion_data.append(row_data)

            # Prepare frame for writing
            frame_to_write = last_annotated if last_annotated is not None else frame_bgr

            # Scale down if needed
            if resolution_scale != 1.0:
                frame_to_write = cv2.resize(frame_to_write, (output_width, output_height))

            writer.write(frame_to_write)
            frame_index += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # Write CSV file
    df = pd.DataFrame(emotion_data)
    df.to_csv(output_csv_path, index=False)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# Read the CSV file with video metadata
df = pd.read_csv("cyclesix_owl.csv")

# Pre-scan to build list of files to process
files_to_process = []

print("Scanning for files to process...")
for idx, row in df.iterrows():
    try:
        input_video = row["filepath"]
        roi_str = row["ROI"]

        # Skip rows with missing filepath
        if pd.isna(input_video) or not isinstance(input_video, str):
            continue

        # Skip rows with missing or NaN ROI
        if pd.isna(roi_str) or not isinstance(roi_str, str):
            continue

        # Get output directory (same as input directory)
        input_dir = os.path.dirname(input_video)
        input_filename = os.path.basename(input_video)
        base_name = os.path.splitext(input_filename)[0]

        # Construct output paths
        output_video_filename = f"{base_name}_annot.mp4"
        output_csv_filename = f"{base_name}_annot.csv"
        output_video_path = os.path.join(input_dir, output_video_filename)
        output_csv_path = os.path.join(input_dir, output_csv_filename)

        # Skip if annotated files already exist
        if os.path.exists(output_video_path) and os.path.exists(output_csv_path):
            continue

        # Parse ROI: "x1,y1,x2,y2" -> (x, y, w, h)
        crop = None
        try:
            coords = list(map(int, roi_str.split(",")))
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                crop = (x1, y1, x2 - x1, y2 - y1)
            else:
                continue
        except (ValueError, IndexError):
            continue

        # Add to processing list
        files_to_process.append({
            "idx": idx,
            "input_video": input_video,
            "output_video_path": output_video_path,
            "output_csv_path": output_csv_path,
            "crop": crop,
            "input_filename": input_filename,
            "output_video_filename": output_video_filename,
            "output_csv_filename": output_csv_filename,
        })

    except Exception as e:
        continue

print(f"Found {len(files_to_process)} files to process\n")

# Process the files with progress bar
for file_info in tqdm(files_to_process, desc="Processing videos", unit="video"):
    try:
        idx = file_info["idx"]
        input_video = file_info["input_video"]
        output_video_path = file_info["output_video_path"]
        output_csv_path = file_info["output_csv_path"]
        crop = file_info["crop"]
        input_filename = file_info["input_filename"]
        output_video_filename = file_info["output_video_filename"]
        output_csv_filename = file_info["output_csv_filename"]

        print(f"\nProcessing: {input_filename}")
        print(f"  Output video: {output_video_filename}")
        print(f"  Output CSV: {output_csv_filename}")
        print(f"  Crop ROI: {crop}")

        # Process video with lower resolution output (50% of original)
        # Adjust resolution_scale as needed: 0.25 (25%), 0.5 (50%), 0.75 (75%), 1.0 (100%)
        process_video_with_timeseries(
            input_path=input_video,
            output_video_path=output_video_path,
            output_csv_path=output_csv_path,
            crop=crop,
            frame_skip=1,
            detector_backend="retinaface",
            resolution_scale=0.5
        )

        print(f"  ✓ Successfully processed")

    except Exception as e:
        print(f"Row {file_info['idx']}: Error processing {file_info['input_filename']}: {e}")
        continue
