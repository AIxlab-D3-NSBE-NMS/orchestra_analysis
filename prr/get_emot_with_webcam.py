"""
camera_emotions.py
==================
Standalone real-time emotion detection from camera feed.
Records annotated video and emotion timeseries CSV.
Uses tkinter for display (no GTK dependency).
"""

import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import os
from threading import Thread
from queue import Queue
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Emotion detection
# ---------------------------------------------------------------------------

def get_emotions_from_frame(
    img: np.ndarray,
    crop: tuple | None = None,
    detector_backend: str = "retinaface",
) -> list[dict]:
    """Detect faces and analyse emotions in a single RGB frame."""
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

    if isinstance(results, dict):
        results = [results]

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
) -> None:
    """Draw a dashed rectangle on canvas."""
    dash_len, gap_len = 12, 6
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

    _h_dashes(x, x + w, y)
    _h_dashes(x, x + w, y + h)
    _v_dashes(y, y + h, x)
    _v_dashes(y, y + h, x + w)


def _draw_emotion_bars(
    canvas: np.ndarray,
    emotions: dict,
    origin_x: int,
    origin_y: int,
    bar_max_width: int = 70,
) -> None:
    """Render emotion probability bars on canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    row_h = 12
    text_color = (255, 255, 255)
    bar_color = (0, 200, 80)
    bg_color = (30, 30, 30)

    for i, (emotion, prob) in enumerate(emotions.items()):
        row_y = origin_y + i * row_h
        bar_w = int(bar_max_width * max(0.0, min(prob, 100.0)) / 100.0)

        cv2.rectangle(
            canvas,
            (origin_x, row_y),
            (origin_x + bar_max_width + 70, row_y + row_h - 1),
            bg_color,
            -1,
        )
        if bar_w > 0:
            cv2.rectangle(
                canvas,
                (origin_x, row_y),
                (origin_x + bar_w, row_y + row_h - 1),
                bar_color,
                -1,
            )
        label = f"{emotion[:3]} {prob:4.0f}%"
        cv2.putText(
            canvas,
            label,
            (origin_x + bar_max_width + 3, row_y + row_h - 2),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )


def annotate_image(
    img: np.ndarray,
    face_results: list[dict],
    crop: tuple | None = None,
) -> np.ndarray:
    """Draw face regions and emotions on a copy of img."""
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

        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2)

        label_y = max(fy - 6, 16)
        (lw, lh), baseline = cv2.getTextSize(dominant, font, 0.55, 1)
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

        bars_x = 10
        bars_y = 50
        _draw_emotion_bars(canvas, emotions, bars_x, bars_y)

    if crop is not None:
        cx, cy, cw, ch = crop
        _draw_dashed_rect(canvas, cx, cy, cw, ch, color=(200, 80, 0), thickness=2)

    return canvas


# ---------------------------------------------------------------------------
# Main camera processor
# ---------------------------------------------------------------------------

def run_camera_emotion_detection(
    camera_index: int = 0,
    output_video: str | None = None,
    output_csv: str | None = None,
    crop: tuple | None = None,
    frame_skip: int = 1,
    detector_backend: str = "retinaface",
    max_queue_size: int = 2,
    display_scale: float = 1.5,
) -> None:
    """
    Run real-time emotion detection on camera feed with tkinter display.

    Parameters
    ----------
    camera_index : int
        Camera device index (0 = default).
    output_video : str or None
        Path to save annotated video. If None, video is not saved.
    output_csv : str or None
        Path to save emotion timeseries CSV. If None, CSV is not saved.
    crop : tuple or None
        Optional ``(x, y, w, h)`` crop region for detection.
    frame_skip : int
        Process every Nth frame (1 = every frame, 5 = every 5th frame).
    detector_backend : str
        DeepFace detector backend.
    max_queue_size : int
        Max frames in queue before dropping to prevent lag.
    display_scale : float
        Scale factor for display (1.0 = original, 1.5 = 1.5x larger, 2.0 = 2x larger).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera device {camera_index}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale dimensions for display
    display_width = int(width * display_scale)
    display_height = int(height * display_scale)

    print(f"Camera info: {width}x{height} @ {fps} FPS")
    print(f"Display scale: {display_scale}x ({display_width}x{display_height})")

    # Video writer
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Queues for threading
    frame_queue: Queue = Queue(maxsize=max_queue_size)
    result_queue: Queue = Queue()

    # Emotion data collection
    emotion_data = []
    frame_index = 0
    last_result = None
    last_dominant = "no_face"
    last_emotions = {
        "angry": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.0,
        "neutral": 0.0,
        "sad": 0.0,
        "surprise": 0.0,
    }

    # Detection thread
    def detection_worker():
        while True:
            item = frame_queue.get()
            if item is None:
                break

            idx, frame_rgb = item
            try:
                results = get_emotions_from_frame(
                    frame_rgb,
                    crop=crop,
                    detector_backend=detector_backend,
                )
                result_queue.put((idx, results, None))
            except Exception as e:
                result_queue.put((idx, None, str(e)))

    detector_thread = Thread(target=detection_worker, daemon=True)
    detector_thread.start()

    # Tkinter setup
    root = tk.Tk()
    root.title("Camera Emotion Detection")
    root.geometry(f"{display_width}x{display_height+50}")

    label = tk.Label(root, bg="black")
    label.pack(fill=tk.BOTH, expand=True)

    status_label = tk.Label(root, text="", bg="gray20", fg="white", height=2)
    status_label.pack(fill=tk.X)

    print("\n🎥 Recording from camera...")
    print("   Close window to stop")
    print()

    stopped = False

    def on_closing():
        nonlocal stopped
        stopped = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    def update_frame():
        nonlocal stopped, frame_index, last_result, last_dominant, last_emotions

        if stopped:
            return

        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to read from camera")
            root.destroy()
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = (frame_index / fps) * 1000.0 if fps > 0 else 0.0

        # Queue frame for detection if it's time
        if frame_index % frame_skip == 0:
            try:
                frame_queue.put_nowait((frame_index, frame_rgb))
            except:
                pass

        # Process results from detection thread
        while not result_queue.empty():
            idx, results, error = result_queue.get()
            if error:
                print(f"[Frame {idx}] Error: {error}")
            else:
                last_result = results
                if results:
                    face = results[0]
                    last_dominant = face.get("dominant_emotion", "unknown")
                    last_emotions = face.get("emotion", {})
                else:
                    last_dominant = "no_face"

        # Annotate with last result
        if last_result is not None:
            annotated = annotate_image(frame_rgb, last_result, crop=crop)
        else:
            annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Record emotion data
        row_data = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "dominant_emotion": last_dominant,
        }
        row_data.update(last_emotions)
        emotion_data.append(row_data)

        # Write frame to video
        if writer:
            writer.write(annotated)

        # Convert for tkinter (BGR -> RGB)
        display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Scale for display if needed
        if display_scale != 1.0:
            display_frame = cv2.resize(
                display_frame,
                (display_width, display_height),
                interpolation=cv2.INTER_LINEAR
            )

        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

        # Update status
        status_label.config(
            text=f"Frame: {frame_index} | Emotion: {last_dominant} | FPS: {fps:.1f}"
        )

        frame_index += 1
        root.after(int(1000 / fps), update_frame)

    root.after(0, update_frame)
    root.mainloop()

    # Cleanup
    frame_queue.put(None)
    detector_thread.join(timeout=5)

    cap.release()
    if writer:
        writer.release()

    # Save outputs
    if output_csv:
        df = pd.DataFrame(emotion_data)
        df.to_csv(output_csv, index=False)
        print(f"✓ CSV saved: {output_csv}")

    if output_video:
        print(f"✓ Video saved: {output_video}")

    print(f"✓ Recorded {frame_index} frames")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  📹 Camera Emotion Detection")
    print("=" * 70 + "\n")

    camera_idx = input("Camera index (default 0): ").strip()
    camera_idx = int(camera_idx) if camera_idx else 0

    frame_skip = input("Frame skip (1=every frame, 5=every 5th, default 1): ").strip()
    frame_skip = int(frame_skip) if frame_skip else 1

    display_scale = input("Display scale (1.0=original, 1.5=1.5x, 2.0=2x, default 1.5): ").strip()
    try:
        display_scale = float(display_scale) if display_scale else 1.5
    except ValueError:
        display_scale = 1.5

    crop_input = input("Crop ROI as 'x1,y1,x2,y2' (or press Enter for none): ").strip()
    crop = None
    if crop_input:
        try:
            coords = list(map(int, crop_input.split(",")))
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                crop = (x1, y1, x2 - x1, y2 - y1)
                print(f"✓ Crop region: {crop}\n")
        except (ValueError, IndexError):
            print("Could not parse ROI, proceeding without crop\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"camera_emotions_{timestamp}.mp4"
    output_csv = f"camera_emotions_{timestamp}.csv"

    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}\n")

    run_camera_emotion_detection(
        camera_index=camera_idx,
        output_video=output_video,
        output_csv=output_csv,
        crop=crop,
        frame_skip=frame_skip,
        display_scale=display_scale,
        detector_backend="retinaface",
    )

    print("\n" + "=" * 70)
    print("  ✅ Done!")
    print("=" * 70)
