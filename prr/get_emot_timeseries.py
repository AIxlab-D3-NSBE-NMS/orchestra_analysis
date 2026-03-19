"""
get_emot_timeseries.py
======================
Utilities for per-frame emotion detection using DeepFace, with optional
cropped-ROI support and fully annotated image / video output.
"""

import cv2
from deepface import DeepFace
import numpy as np
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
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

            writer.write(last_annotated if last_annotated is not None else frame_bgr)
            frame_index += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

annotate_video(
    input_path="annotated_2026-02-02_14-17-51-842263.mp4",
    output_path="annotated_video.mp4",
    crop=None,  # or e.g. (100, 50, 800, 600) for ROI
    frame_skip=1,  # process every frame; use 5 to process every 5th frame
    detector_backend="retinaface"
)
