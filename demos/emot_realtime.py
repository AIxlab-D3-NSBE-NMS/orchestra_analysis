import argparse
import threading
import time

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

try:
    from mediapipe.solutions import pose as mp_pose

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None


EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 165, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 0),
    "neutral": (128, 128, 128),
}


def update_ema(current_value, new_value, alpha=0.2):
    """Update an exponential moving average."""
    if current_value is None:
        return new_value
    return current_value * (1 - alpha) + new_value * alpha


def process_faces(image):
    """
    Process faces in the image for emotion detection.

    Returns:
        tuple: (faces_detected, emotions_data, inference_ms)
    """
    start_time = time.perf_counter()

    try:
        faces = DeepFace.extract_faces(image, enforce_detection=False)

        if not faces:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return False, None, elapsed_ms

        emotions_data = []

        for face in faces:
            facial_area = face["facial_area"]
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]

            try:
                face_region = image[y : y + h, x : x + w]
                emotions = DeepFace.analyze(
                    face_region,
                    actions=["emotion"],
                    enforce_detection=False,
                )
                emotion_dict = emotions[0]["emotion"]

                emotions_data.append(
                    {
                        "coords": (x, y, w, h),
                        "emotions": emotion_dict,
                    }
                )
            except FaceNotDetected:
                continue
            except Exception as exc:
                print(f"Error analyzing emotion: {exc}")
                continue

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return len(emotions_data) > 0, emotions_data, elapsed_ms

    except FaceNotDetected:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return False, None, elapsed_ms
    except Exception as exc:
        print(f"Error in face detection: {exc}")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return False, None, elapsed_ms


def process_pose(image, pose_detector):
    """
    Process pose estimation on the image.

    Returns:
        tuple: (pose_detected, pose_landmarks, inference_ms)
    """
    if not MEDIAPIPE_AVAILABLE or pose_detector is None:
        return False, None, 0.0

    start_time = time.perf_counter()

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if results.pose_landmarks is None:
            return False, None, elapsed_ms

        return True, results.pose_landmarks, elapsed_ms

    except Exception as exc:
        print(f"Error in pose detection: {exc}")
        return False, None, 0.0


def draw_emotion_bars(
    overlay,
    x,
    y,
    w,
    h,
    emotions_dict,
    bar_width=100,
    bar_height=140,
    overlay_scale=1.0,
):
    """
    Draw emotion probability bars onto an overlay image.
    """
    bar_width = max(20, int(bar_width * overlay_scale))
    bar_height = max(28, int(bar_height * overlay_scale))
    margin = max(4, int(10 * overlay_scale))
    label_x_padding = max(2, int(3 * overlay_scale))
    border_thickness = max(1, int(round(overlay_scale)))
    font_scale = max(0.3, 0.35 * overlay_scale)
    font_thickness = max(1, int(round(overlay_scale)))

    img_height, img_width = overlay.shape[:2]

    bar_x = x + w + margin
    bar_y = y

    if bar_x + bar_width > img_width:
        bar_x = max(margin, x - bar_width - margin)
    if bar_y + bar_height > img_height:
        bar_y = max(margin, img_height - bar_height - margin)

    sorted_emotions = sorted(emotions_dict.items())
    num_emotions = len(sorted_emotions)

    if num_emotions == 0:
        return

    bar_spacing = bar_height / num_emotions
    bar_individual_height = max(10, int(bar_spacing * 0.85))

    for idx, (emotion_name, emotion_value) in enumerate(sorted_emotions):
        current_y = int(bar_y + idx * bar_spacing + (bar_spacing - bar_individual_height) / 2)
        bar_bottom = current_y + bar_individual_height

        cv2.rectangle(
            overlay,
            (bar_x, current_y),
            (bar_x + bar_width, bar_bottom),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            overlay,
            (bar_x, current_y),
            (bar_x + bar_width, bar_bottom),
            (200, 200, 200),
            border_thickness,
        )

        normalized_value = emotion_value / 100.0
        fill_width = int(normalized_value * bar_width)

        color = EMOTION_COLORS.get(emotion_name.lower(), (128, 128, 128))
        if fill_width > 0:
            cv2.rectangle(
                overlay,
                (bar_x, current_y),
                (bar_x + fill_width, bar_bottom),
                color,
                -1,
            )

        text_y = current_y + max(bar_individual_height - 2, int(10 * overlay_scale))
        cv2.putText(
            overlay,
            emotion_name.upper(),
            (bar_x + label_x_padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )


def draw_pose(image, pose_landmarks, alpha_pose=0.7):
    """
    Draw pose skeleton on the image.
    """
    if not MEDIAPIPE_AVAILABLE or mp_pose is None or pose_landmarks is None:
        return image

    img_height, img_width = image.shape[:2]
    overlay = image.copy()

    for landmark in pose_landmarks.landmark:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        if landmark.visibility > 0.5:
            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 4, (255, 255, 255), 1)

    try:
        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
            start = pose_landmarks.landmark[start_idx]
            end = pose_landmarks.landmark[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                start_x = int(start.x * img_width)
                start_y = int(start.y * img_height)
                end_x = int(end.x * img_width)
                end_y = int(end.y * img_height)
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    except AttributeError:
        print("Warning: Could not access POSE_CONNECTIONS")

    cv2.addWeighted(overlay, alpha_pose, image, 1 - alpha_pose, 0, image)
    return image


def draw_overlays(image, emotions_data, alpha_bar=0.7, overlay_scale=1.0):
    """
    Draw bounding boxes and emotion bars on the image.
    """
    if emotions_data is None:
        return image

    overlay = image.copy()

    for face_data in emotions_data:
        x, y, w, h = face_data["coords"]
        emotions = face_data["emotions"]

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_emotion_bars(
            overlay,
            x,
            y,
            w,
            h,
            emotions,
            overlay_scale=overlay_scale,
        )

    cv2.addWeighted(overlay, alpha_bar, image, 1 - alpha_bar, 0, image)
    return image


def emotion_worker_loop(shared_state, state_lock, stop_event):
    """Run DeepFace emotion inference in the background on the latest submitted frame."""
    last_processed_frame_id = -1

    while not stop_event.is_set():
        frame = None
        frame_id = -1

        with state_lock:
            submitted_frame = shared_state["submitted_frame"]
            submitted_frame_id = shared_state["submitted_frame_id"]
            emotion_interval_s = shared_state["emotion_interval_s"]

            if submitted_frame is not None and submitted_frame_id != last_processed_frame_id:
                frame = submitted_frame.copy()
                frame_id = submitted_frame_id

        if frame is None:
            stop_event.wait(0.01)
            continue

        faces_detected, emotions_data, inference_ms = process_faces(frame)
        completed_at = time.monotonic()

        with state_lock:
            shared_state["emotion_inference_ms_ema"] = update_ema(
                shared_state["emotion_inference_ms_ema"],
                inference_ms,
            )
            shared_state["emotion_updates"] += 1
            shared_state["last_emotion_completed_at"] = completed_at

            if faces_detected and emotions_data:
                shared_state["last_emotions_data"] = emotions_data
                shared_state["last_emotion_result_at"] = completed_at

        last_processed_frame_id = frame_id
        stop_event.wait(emotion_interval_s)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time facial recognition, emotion detection, and pose estimation"
    )
    parser.add_argument(
        "--skipframes",
        type=int,
        default=1,
        help="Number of frames to skip between emotion submissions (default: 1)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)",
    )
    parser.add_argument(
        "--alpha-bar",
        type=float,
        default=0.7,
        help="Opacity of emotion bars (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--alpha-pose",
        type=float,
        default=0.7,
        help="Opacity of pose skeleton (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--enable-pose",
        action="store_true",
        default=False,
        help="Enable pose estimation (default: disabled for faster processing)",
    )
    parser.add_argument(
        "--skip-pose-frames",
        type=int,
        default=1,
        help="Number of frames to skip for pose processing checks (default: 1)",
    )
    parser.add_argument(
        "--emotion-overlay-scale",
        type=float,
        default=1.0,
        help="Visual scale factor for emotion bar overlays (default: 1.0)",
    )
    parser.add_argument(
        "--emotion-fps",
        type=float,
        default=6.0,
        help="Maximum rate for emotion inference updates (default: 6.0)",
    )
    parser.add_argument(
        "--pose-fps",
        type=float,
        default=10.0,
        help="Maximum rate for pose inference updates when enabled (default: 10.0)",
    )
    parser.add_argument(
        "--emotion-ttl",
        type=float,
        default=1.0,
        help="How long to keep the last emotion overlay visible in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--pose-ttl",
        type=float,
        default=0.5,
        help="How long to keep the last pose overlay visible in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha_bar <= 1.0:
        print("Warning: alpha_bar should be between 0.0 and 1.0. Using default 0.7")
        args.alpha_bar = 0.7

    if not 0.0 <= args.alpha_pose <= 1.0:
        print("Warning: alpha_pose should be between 0.0 and 1.0. Using default 0.7")
        args.alpha_pose = 0.7

    if args.emotion_overlay_scale <= 0.0:
        print("Warning: emotion_overlay_scale should be greater than 0. Using default 1.0")
        args.emotion_overlay_scale = 1.0

    if args.emotion_fps <= 0.0:
        print("Warning: emotion_fps should be greater than 0. Using default 6.0")
        args.emotion_fps = 6.0

    if args.pose_fps <= 0.0:
        print("Warning: pose_fps should be greater than 0. Using default 10.0")
        args.pose_fps = 10.0

    if args.emotion_ttl < 0.0:
        print("Warning: emotion_ttl should be non-negative. Using default 1.0")
        args.emotion_ttl = 1.0

    if args.pose_ttl < 0.0:
        print("Warning: pose_ttl should be non-negative. Using default 0.5")
        args.pose_ttl = 0.5

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if actual_width != args.width or actual_height != args.height:
        print(
            f"Warning: Requested resolution {args.width}x{args.height} "
            f"but camera returned {actual_width}x{actual_height}"
        )
    else:
        print(f"Camera resolution set to {actual_width}x{actual_height}")

    if actual_fps != args.fps:
        print(f"Note: Requested FPS {args.fps} but camera set to {actual_fps:.1f}")

    pose_detector = None
    if args.enable_pose:
        if not MEDIAPIPE_AVAILABLE:
            print("Error: MediaPipe is required for pose estimation. Install with: pip install mediapipe")
            args.enable_pose = False
        else:
            pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print(f"Pose estimation enabled (skip frames: {args.skip_pose_frames})")

    frame_delay = int(1000 / args.fps)
    emotion_interval_s = 1.0 / args.emotion_fps
    pose_interval_s = 1.0 / args.pose_fps

    frame_count = 0
    start_time = time.time()
    emotion_updates = 0
    pose_count = 0
    last_pose_landmarks = None
    last_pose_result_at = None
    pose_inference_ms_ema = None
    last_emotion_submit_at = 0.0
    last_pose_process_at = 0.0

    state_lock = threading.Lock()
    stop_event = threading.Event()
    shared_state = {
        "submitted_frame": None,
        "submitted_frame_id": -1,
        "emotion_interval_s": emotion_interval_s,
        "last_emotions_data": None,
        "last_emotion_result_at": None,
        "last_emotion_completed_at": None,
        "emotion_inference_ms_ema": None,
        "emotion_updates": 0,
    }
    emotion_thread = threading.Thread(
        target=emotion_worker_loop,
        args=(shared_state, state_lock, stop_event),
        daemon=True,
    )
    emotion_thread.start()

    print("Starting real-time emotion detection and pose estimation...")
    print(f"Skip frames (emotion submit): {args.skipframes}")
    print(f"Target FPS: {args.fps}")
    print(f"Emotion update rate: {args.emotion_fps:.1f} Hz")
    print(f"Emotion result TTL: {args.emotion_ttl:.2f} s")
    print(f"Emotion bar opacity: {args.alpha_bar}")
    print(f"Emotion overlay scale: {args.emotion_overlay_scale}")
    if args.enable_pose:
        print(f"Pose update rate: {args.pose_fps:.1f} Hz")
        print(f"Pose result TTL: {args.pose_ttl:.2f} s")
        print(f"Pose skeleton opacity: {args.alpha_pose}")
    print("Press 'q' to quit, 'p' to toggle pose")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break

            now = time.monotonic()

            should_submit_emotion = (
                frame_count % args.skipframes == 0
                and (now - last_emotion_submit_at) >= emotion_interval_s
            )
            if should_submit_emotion:
                with state_lock:
                    shared_state["submitted_frame"] = frame
                    shared_state["submitted_frame_id"] += 1
                last_emotion_submit_at = now

            should_process_pose = (
                args.enable_pose
                and pose_detector is not None
                and frame_count % args.skip_pose_frames == 0
                and (now - last_pose_process_at) >= pose_interval_s
            )
            if should_process_pose:
                pose_detected, pose_landmarks, pose_inference_ms = process_pose(frame, pose_detector)
                pose_inference_ms_ema = update_ema(pose_inference_ms_ema, pose_inference_ms)
                last_pose_process_at = now

                if pose_detected and pose_landmarks is not None:
                    last_pose_landmarks = pose_landmarks
                    last_pose_result_at = now
                    pose_count += 1

            with state_lock:
                last_emotions_data = shared_state["last_emotions_data"]
                last_emotion_result_at = shared_state["last_emotion_result_at"]
                emotion_inference_ms_ema = shared_state["emotion_inference_ms_ema"]
                emotion_updates = shared_state["emotion_updates"]

            if (
                last_emotions_data is not None
                and last_emotion_result_at is not None
                and (now - last_emotion_result_at) <= args.emotion_ttl
            ):
                frame = draw_overlays(
                    frame,
                    last_emotions_data,
                    alpha_bar=args.alpha_bar,
                    overlay_scale=args.emotion_overlay_scale,
                )

            if (
                args.enable_pose
                and last_pose_landmarks is not None
                and last_pose_result_at is not None
                and (now - last_pose_result_at) <= args.pose_ttl
            ):
                frame = draw_pose(frame, last_pose_landmarks, alpha_pose=args.alpha_pose)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                emotion_ms_text = "--" if emotion_inference_ms_ema is None else f"{emotion_inference_ms_ema:.0f}"
                status_text = f"FPS: {fps:.1f} | Emotion updates: {emotion_updates} | Emotion ms: {emotion_ms_text}"
                if args.enable_pose:
                    pose_ms_text = "--" if pose_inference_ms_ema is None else f"{pose_inference_ms_ema:.0f}"
                    status_text += f" | Poses: {pose_count} | Pose ms: {pose_ms_text}"

                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Emotion Detection & Pose Estimation - Press Q to Quit", frame)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                if MEDIAPIPE_AVAILABLE:
                    args.enable_pose = not args.enable_pose
                    if args.enable_pose and pose_detector is None:
                        pose_detector = mp_pose.Pose(
                            static_image_mode=False,
                            model_complexity=1,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                        )
                        print("Pose estimation enabled")
                    elif not args.enable_pose:
                        print("Pose estimation disabled")
                else:
                    print("MediaPipe not available")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        stop_event.set()
        emotion_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

        if pose_detector:
            pose_detector.close()

        elapsed = time.time() - start_time
        print("\nSession Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Emotion updates: {emotion_updates}")
        if shared_state["emotion_inference_ms_ema"] is not None:
            print(f"Average emotion inference (EMA): {shared_state['emotion_inference_ms_ema']:.1f} ms")
        if args.enable_pose:
            print(f"Pose detections: {pose_count}")
            if pose_inference_ms_ema is not None:
                print(f"Average pose inference (EMA): {pose_inference_ms_ema:.1f} ms")
        print(f"Duration: {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average FPS: {frame_count / elapsed:.1f}")


if __name__ == "__main__":
    main()
