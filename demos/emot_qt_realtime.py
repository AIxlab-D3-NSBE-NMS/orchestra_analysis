import argparse
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from PySide6.QtCore import QPointF, QRectF, QTimer, QUrl, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

try:
    from mediapipe.solutions import pose as mp_pose

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None


EMOTION_COLORS_BGR = {
    "angry": (0, 0, 255),
    "disgust": (115, 150, 232),
    "fear": (128, 0, 128),
    "happy": (0, 165, 0),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral": (128, 128, 128),
}

EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def bgr_to_qcolor(color):
    b, g, r = color
    return QColor(r, g, b)


EMOTION_COLORS = {
    emotion: bgr_to_qcolor(color) for emotion, color in EMOTION_COLORS_BGR.items()
}


def update_ema(current_value, new_value, alpha=0.2):
    if current_value is None:
        return new_value
    return current_value * (1 - alpha) + new_value * alpha


def process_faces(image):
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
                emotions_data.append(
                    {
                        "coords": (x, y, w, h),
                        "emotions": emotions[0]["emotion"],
                    }
                )
            except FaceNotDetected:
                continue
            except Exception as exc:
                print(f"Error analyzing emotion: {exc}")

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


def frame_to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    ).copy()


def compute_normalized_emotion_distribution(emotions_data):
    if not emotions_data:
        return None

    totals = {}
    for face_data in emotions_data:
        for emotion_name, emotion_value in face_data.get("emotions", {}).items():
            try:
                value = max(0.0, float(emotion_value))
            except (TypeError, ValueError):
                continue
            key = emotion_name.lower()
            totals[key] = totals.get(key, 0.0) + value

    total_score = sum(totals.values())
    if total_score <= 0.0:
        return None

    return {emotion_name: value / total_score for emotion_name, value in totals.items()}


def update_filtered_emotion_distribution(previous, target, dt, smoothing_s):
    target = target or {}
    emotion_names = set(EMOTION_ORDER)
    emotion_names.update(target.keys())
    if previous:
        emotion_names.update(previous.keys())

    if previous is None or smoothing_s <= 0.0:
        return {emotion_name: target.get(emotion_name, 0.0) for emotion_name in emotion_names}

    alpha = 1.0 - math.exp(-max(0.0, dt) / smoothing_s)
    return {
        emotion_name: (
            previous.get(emotion_name, 0.0) * (1.0 - alpha)
            + target.get(emotion_name, 0.0) * alpha
        )
        for emotion_name in emotion_names
    }


def prune_emotion_plot_history(history, now, history_seconds):
    cutoff = now - history_seconds
    while history and history[0][0] < cutoff:
        history.popleft()


def get_emotion_plot_names(history, current_distribution=None):
    emotion_names = set()
    if current_distribution:
        emotion_names.update(current_distribution.keys())
    for _, sample in history:
        emotion_names.update(sample.keys())

    ordered_names = [emotion for emotion in EMOTION_ORDER if emotion in emotion_names]
    ordered_names.extend(sorted(emotion_names - set(ordered_names)))
    return ordered_names or EMOTION_ORDER


def emotion_worker_loop(shared_state, state_lock, stop_event):
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


class RealtimeCanvas(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.camera_image = None
        self.media_image = None
        self.media_visible = False
        self.frame_size = (args.width, args.height)
        self.emotions_data = None
        self.pose_landmarks = None
        self.plot_history = deque()
        self.current_distribution = None
        self.status_text = ""
        self.setMinimumSize(640, 480)

    def set_camera_frame(self, image, frame_size):
        self.camera_image = image
        self.frame_size = frame_size
        self.update()

    def set_media_frame(self, image):
        self.media_image = image.copy() if not image.isNull() else None
        self.update()

    def set_media_visible(self, visible):
        self.media_visible = visible
        if not visible:
            self.media_image = None
        self.update()

    def set_overlays(self, emotions_data, pose_landmarks, plot_history, current_distribution, status_text):
        self.emotions_data = emotions_data
        self.pose_landmarks = pose_landmarks
        self.plot_history = plot_history
        self.current_distribution = current_distribution
        self.status_text = status_text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        target = QRectF(self.rect())

        if self.camera_image is not None:
            painter.drawImage(target, self.camera_image)

        if self.media_visible and self.media_image is not None:
            painter.save()
            painter.setOpacity(self.args.media_alpha)
            painter.drawImage(target, self.media_image)
            painter.restore()

        if self.emotions_data:
            self.draw_emotion_overlays(painter, target)

        if self.pose_landmarks is not None:
            self.draw_pose(painter, target)

        if not self.args.hide_emotion_plot:
            self.draw_emotion_plot(painter)

        if self.status_text:
            self.draw_status(painter)

    def scale_rect(self, x, y, w, h, target):
        frame_width, frame_height = self.frame_size
        sx = target.width() / max(1, frame_width)
        sy = target.height() / max(1, frame_height)
        return QRectF(target.x() + x * sx, target.y() + y * sy, w * sx, h * sy)

    def draw_emotion_overlays(self, painter, target):
        painter.save()
        painter.setOpacity(self.args.alpha_bar)

        for face_data in self.emotions_data:
            x, y, w, h = face_data["coords"]
            face_rect = self.scale_rect(x, y, w, h, target)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(face_rect)
            self.draw_emotion_bars(painter, face_rect, face_data["emotions"])

        painter.restore()

    def draw_emotion_bars(self, painter, face_rect, emotions):
        scale = self.args.emotion_overlay_scale
        bar_width = max(20, int(100 * scale))
        bar_height = max(28, int(140 * scale))
        margin = max(4, int(10 * scale))
        font_size = max(7, int(9 * scale))

        x = face_rect.right() + margin
        y = face_rect.top()
        if x + bar_width > self.width():
            x = max(margin, face_rect.left() - bar_width - margin)
        if y + bar_height > self.height():
            y = max(margin, self.height() - bar_height - margin)

        sorted_emotions = sorted(emotions.items())
        if not sorted_emotions:
            return

        spacing = bar_height / len(sorted_emotions)
        individual_height = max(10, int(spacing * 0.85))

        painter.setFont(self.font())
        font = painter.font()
        font.setPointSize(font_size)
        painter.setFont(font)

        for idx, (emotion_name, emotion_value) in enumerate(sorted_emotions):
            current_y = y + idx * spacing + (spacing - individual_height) / 2
            rect = QRectF(x, current_y, bar_width, individual_height)
            painter.fillRect(rect, QColor(40, 40, 40))
            painter.setPen(QPen(QColor(200, 200, 200), max(1, round(scale))))
            painter.drawRect(rect)

            fill_width = max(0.0, min(float(emotion_value) / 100.0, 1.0)) * bar_width
            color = EMOTION_COLORS.get(emotion_name.lower(), QColor(128, 128, 128))
            painter.fillRect(QRectF(x, current_y, fill_width, individual_height), color)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QPointF(x + 3 * scale, current_y + individual_height - 3), emotion_name.upper())

    def draw_pose(self, painter, target):
        if not MEDIAPIPE_AVAILABLE or mp_pose is None:
            return

        painter.save()
        painter.setOpacity(self.args.alpha_pose)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        frame_width, frame_height = self.frame_size

        def to_point(landmark):
            return QPointF(
                target.x() + landmark.x * target.width(),
                target.y() + landmark.y * target.height(),
            )

        try:
            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                start = self.pose_landmarks.landmark[start_idx]
                end = self.pose_landmarks.landmark[end_idx]
                if start.visibility > 0.5 and end.visibility > 0.5:
                    painter.drawLine(to_point(start), to_point(end))
        except AttributeError:
            pass

        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QColor(0, 255, 0))
        for landmark in self.pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                point = to_point(landmark)
                painter.drawEllipse(point, 4, 4)

        painter.restore()

    def draw_emotion_plot(self, painter):
        if len(self.plot_history) < 2 or self.args.emotion_plot_height <= 0:
            return

        now = time.monotonic()
        plot_height = min(self.args.emotion_plot_height, max(40, self.height() - 50))
        plot_y = self.height() - plot_height
        plot_rect = QRectF(0, plot_y, self.width(), plot_height)
        graph_top = plot_y + 18
        graph_bottom = self.height() - 12
        graph_height = max(1, graph_bottom - graph_top)

        painter.save()
        painter.setOpacity(self.args.emotion_plot_alpha)
        painter.fillRect(plot_rect, QColor(15, 15, 15))
        painter.restore()

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        for fraction in (0.25, 0.5, 0.75):
            y = graph_bottom - fraction * graph_height
            painter.drawLine(QPointF(0, y), QPointF(self.width(), y))

        emotion_names = get_emotion_plot_names(self.plot_history, self.current_distribution)
        time_start = now - self.args.emotion_plot_history

        for emotion_name in emotion_names:
            color = EMOTION_COLORS.get(emotion_name, QColor(220, 220, 220))
            painter.setPen(QPen(color, 2))
            points = []
            for sample_time, sample in self.plot_history:
                value = max(0.0, min(float(sample.get(emotion_name, 0.0)), 1.0))
                elapsed_fraction = (sample_time - time_start) / self.args.emotion_plot_history
                x = max(0.0, min(elapsed_fraction, 1.0)) * max(1, self.width() - 1)
                y = graph_bottom - value * graph_height
                points.append(QPointF(x, y))

            for start, end in zip(points, points[1:]):
                painter.drawLine(start, end)

        legend_x = 8
        legend_y = plot_y + 13
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for emotion_name in emotion_names:
            color = EMOTION_COLORS.get(emotion_name, QColor(220, 220, 220))
            value = 0.0 if self.current_distribution is None else self.current_distribution.get(emotion_name, 0.0)
            painter.setPen(color)
            painter.drawText(QPointF(legend_x, legend_y), f"{emotion_name[:3].upper()} {value * 100:4.1f}")
            legend_x += 72
            if legend_x > self.width() - 70:
                legend_x = 8
                legend_y += 14

    def draw_status(self, painter):
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(QPointF(11, 31), self.status_text)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QPointF(10, 30), self.status_text)


class EmotionQtWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Emotion Detection Qt Demo")

        self.canvas = RealtimeCanvas(args)
        self.setCentralWidget(self.canvas)
        self.resize(args.width, args.height)

        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {args.camera}")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)

        self.pose_detector = None
        if args.enable_pose:
            self.enable_pose()

        self.media_path = self.resolve_media_path(args.media)
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.video_sink = QVideoSink(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoSink(self.video_sink)
        self.audio_output.setVolume(args.media_volume)
        self.video_sink.videoFrameChanged.connect(self.on_media_frame)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)
        if self.media_path is not None:
            self.media_player.setSource(QUrl.fromLocalFile(str(self.media_path)))

        self.frame_delay_ms = int(1000 / args.fps)
        self.emotion_interval_s = 1.0 / args.emotion_fps
        self.pose_interval_s = 1.0 / args.pose_fps
        self.frame_count = 0
        self.start_time = time.time()
        self.last_emotion_submit_at = 0.0
        self.last_pose_process_at = 0.0
        self.last_pose_result_at = None
        self.last_pose_landmarks = None
        self.pose_count = 0
        self.pose_inference_ms_ema = None
        self.emotion_updates = 0
        self.filtered_emotion_distribution = None
        self.last_plot_update_at = time.monotonic()
        self.emotion_plot_history = deque()

        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.shared_state = {
            "submitted_frame": None,
            "submitted_frame_id": -1,
            "emotion_interval_s": self.emotion_interval_s,
            "last_emotions_data": None,
            "last_emotion_result_at": None,
            "last_emotion_completed_at": None,
            "emotion_inference_ms_ema": None,
            "emotion_updates": 0,
        }
        self.emotion_thread = threading.Thread(
            target=emotion_worker_loop,
            args=(self.shared_state, self.state_lock, self.stop_event),
            daemon=True,
        )
        self.emotion_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_frame)
        self.timer.start(self.frame_delay_ms)

        print("Press 'q' to quit, 'p' to toggle pose, 'v' to toggle media video")

    def resolve_media_path(self, media_path):
        if media_path is None:
            return None

        path = Path(media_path).expanduser().resolve()
        if not path.is_file():
            print(f"Warning: media path is not a file: {path}")
            return None

        print(f"Media playback ready: {path}")
        return path

    def enable_pose(self):
        if not MEDIAPIPE_AVAILABLE:
            print("MediaPipe is not available")
            self.args.enable_pose = False
            return
        if self.pose_detector is None:
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        self.args.enable_pose = True
        print("Pose estimation enabled")

    def disable_pose(self):
        self.args.enable_pose = False
        self.last_pose_landmarks = None
        self.last_pose_result_at = None
        print("Pose estimation disabled")

    def toggle_media(self):
        if self.media_path is None:
            print("No media video loaded")
            return

        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.stop()
            self.canvas.set_media_visible(False)
            print("Media playback stopped")
            return

        self.media_player.setPosition(0)
        self.canvas.set_media_visible(True)
        self.media_player.play()
        print("Media playback started")

    def on_media_frame(self, video_frame):
        if not video_frame.isValid():
            return

        image = video_frame.toImage()
        if image.isNull():
            return

        self.canvas.set_media_frame(image)

    def on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.canvas.set_media_visible(False)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
            return
        if key == Qt.Key.Key_V:
            self.toggle_media()
            return
        if key == Qt.Key.Key_P:
            if self.args.enable_pose:
                self.disable_pose()
            else:
                self.enable_pose()
            return
        super().keyPressEvent(event)

    def on_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            return

        now = time.monotonic()
        height, width = frame.shape[:2]
        self.canvas.set_camera_frame(frame_to_qimage(frame), (width, height))

        should_submit_emotion = (
            self.frame_count % self.args.skipframes == 0
            and (now - self.last_emotion_submit_at) >= self.emotion_interval_s
        )
        if should_submit_emotion:
            with self.state_lock:
                self.shared_state["submitted_frame"] = frame.copy()
                self.shared_state["submitted_frame_id"] += 1
            self.last_emotion_submit_at = now

        should_process_pose = (
            self.args.enable_pose
            and self.pose_detector is not None
            and self.frame_count % self.args.skip_pose_frames == 0
            and (now - self.last_pose_process_at) >= self.pose_interval_s
        )
        if should_process_pose:
            pose_detected, pose_landmarks, pose_inference_ms = process_pose(frame, self.pose_detector)
            self.pose_inference_ms_ema = update_ema(self.pose_inference_ms_ema, pose_inference_ms)
            self.last_pose_process_at = now
            if pose_detected and pose_landmarks is not None:
                self.last_pose_landmarks = pose_landmarks
                self.last_pose_result_at = now
                self.pose_count += 1

        with self.state_lock:
            last_emotions_data = self.shared_state["last_emotions_data"]
            last_emotion_result_at = self.shared_state["last_emotion_result_at"]
            emotion_inference_ms_ema = self.shared_state["emotion_inference_ms_ema"]
            self.emotion_updates = self.shared_state["emotion_updates"]

        current_emotion_distribution = None
        visible_emotions_data = None
        if (
            last_emotions_data is not None
            and last_emotion_result_at is not None
            and (now - last_emotion_result_at) <= self.args.emotion_ttl
        ):
            visible_emotions_data = last_emotions_data
            current_emotion_distribution = compute_normalized_emotion_distribution(last_emotions_data)

        visible_pose = None
        if (
            self.args.enable_pose
            and self.last_pose_landmarks is not None
            and self.last_pose_result_at is not None
            and (now - self.last_pose_result_at) <= self.args.pose_ttl
        ):
            visible_pose = self.last_pose_landmarks

        plot_dt = now - self.last_plot_update_at
        self.last_plot_update_at = now
        self.filtered_emotion_distribution = update_filtered_emotion_distribution(
            self.filtered_emotion_distribution,
            current_emotion_distribution,
            plot_dt,
            self.args.emotion_plot_smoothing,
        )
        self.emotion_plot_history.append((now, self.filtered_emotion_distribution))
        prune_emotion_plot_history(self.emotion_plot_history, now, self.args.emotion_plot_history)

        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        emotion_ms_text = "--" if emotion_inference_ms_ema is None else f"{emotion_inference_ms_ema:.0f}"
        status_text = f"FPS: {fps:.1f} | Emotion updates: {self.emotion_updates} | Emotion ms: {emotion_ms_text}"
        if self.args.enable_pose:
            pose_ms_text = "--" if self.pose_inference_ms_ema is None else f"{self.pose_inference_ms_ema:.0f}"
            status_text += f" | Poses: {self.pose_count} | Pose ms: {pose_ms_text}"

        self.canvas.set_overlays(
            visible_emotions_data,
            visible_pose,
            self.emotion_plot_history,
            self.filtered_emotion_distribution,
            status_text,
        )
        self.frame_count += 1

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_event.set()
        self.emotion_thread.join(timeout=1.0)
        self.media_player.stop()
        self.cap.release()
        if self.pose_detector is not None:
            self.pose_detector.close()

        elapsed = time.time() - self.start_time
        print("\nSession Statistics:")
        print(f"Total frames: {self.frame_count}")
        print(f"Emotion updates: {self.emotion_updates}")
        if self.shared_state["emotion_inference_ms_ema"] is not None:
            print(f"Average emotion inference (EMA): {self.shared_state['emotion_inference_ms_ema']:.1f} ms")
        if self.args.enable_pose:
            print(f"Pose detections: {self.pose_count}")
            if self.pose_inference_ms_ema is not None:
                print(f"Average pose inference (EMA): {self.pose_inference_ms_ema:.1f} ms")
        print(f"Duration: {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average FPS: {self.frame_count / elapsed:.1f}")

        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qt realtime facial emotion detection with synchronized media playback."
    )
    parser.add_argument("--skipframes", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--alpha-bar", type=float, default=0.7)
    parser.add_argument("--alpha-pose", type=float, default=0.7)
    parser.add_argument("--enable-pose", action="store_true", default=False)
    parser.add_argument("--skip-pose-frames", type=int, default=1)
    parser.add_argument("--emotion-overlay-scale", type=float, default=1.0)
    parser.add_argument("--emotion-fps", type=float, default=6.0)
    parser.add_argument("--pose-fps", type=float, default=10.0)
    parser.add_argument("--emotion-ttl", type=float, default=1.0)
    parser.add_argument("--pose-ttl", type=float, default=0.5)
    parser.add_argument("--hide-emotion-plot", action="store_true", default=False)
    parser.add_argument("--emotion-plot-height", type=int, default=140)
    parser.add_argument("--emotion-plot-history", type=float, default=12.0)
    parser.add_argument("--emotion-plot-alpha", type=float, default=0.6)
    parser.add_argument("--emotion-plot-smoothing", type=float, default=0.35)
    parser.add_argument("--media", type=str, default=None)
    parser.add_argument("--media-alpha", type=float, default=0.45)
    parser.add_argument("--media-volume", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args):
    args.alpha_bar = min(max(args.alpha_bar, 0.0), 1.0)
    args.alpha_pose = min(max(args.alpha_pose, 0.0), 1.0)
    args.media_alpha = min(max(args.media_alpha, 0.0), 1.0)
    args.media_volume = min(max(args.media_volume, 0.0), 1.0)
    args.emotion_overlay_scale = max(args.emotion_overlay_scale, 0.1)
    args.emotion_fps = max(args.emotion_fps, 0.1)
    args.pose_fps = max(args.pose_fps, 0.1)
    args.emotion_ttl = max(args.emotion_ttl, 0.0)
    args.pose_ttl = max(args.pose_ttl, 0.0)
    args.emotion_plot_height = max(args.emotion_plot_height, 0)
    args.emotion_plot_history = max(args.emotion_plot_history, 0.1)
    args.emotion_plot_alpha = min(max(args.emotion_plot_alpha, 0.0), 1.0)
    args.emotion_plot_smoothing = max(args.emotion_plot_smoothing, 0.0)
    args.skipframes = max(args.skipframes, 1)
    args.skip_pose_frames = max(args.skip_pose_frames, 1)
    args.fps = max(args.fps, 1)
    return args


def main():
    args = validate_args(parse_args())
    app = QApplication(sys.argv)
    window = EmotionQtWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
