import cv2
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
import argparse
import time
import numpy as np

try:
    from mediapipe.solutions import pose as mp_pose
    from mediapipe.solutions import drawing_utils as mp_drawing
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None

# Emotion color mapping for visualization
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 165, 0),    # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 255, 0),   # Green
    'neutral': (128, 128, 128) # Gray
}

def process_faces(image, skip_frames=1, frame_count=0):
    """
    Process faces in the image for emotion detection.
    
    Args:
        image: Input frame from video capture
        skip_frames: Number of frames to skip between detections
        frame_count: Current frame count
    
    Returns:
        tuple: (processed_image, faces_detected, emotions_data)
    """
    # Check if we should skip this frame
    if frame_count % skip_frames != 0:
        return image, False, None
    
    try:
        faces = DeepFace.extract_faces(image, enforce_detection=False)
        
        if not faces or len(faces) == 0:
            return image, False, None
        
        emotions_data = []
        
        for face in faces:
            facial_area = face['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            
            try:
                # Extract the face region and analyze emotion
                face_region = image[y:y+h, x:x+w]
                emotions = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                emotion_dict = emotions[0]['emotion']
                
                emotions_data.append({
                    'coords': (x, y, w, h),
                    'emotions': emotion_dict
                })
            except FaceNotDetected:
                # Skip this face if emotion analysis fails
                continue
            except Exception as e:
                # Log other errors but continue processing
                print(f"Error analyzing emotion: {e}")
                continue
        
        return image, len(emotions_data) > 0, emotions_data
    
    except FaceNotDetected:
        return image, False, None
    except Exception as e:
        print(f"Error in face detection: {e}")
        return image, False, None


def process_pose(image, pose_detector, skip_frames=1, frame_count=0):
    """
    Process pose estimation on the image.
    
    Args:
        image: Input frame from video capture
        pose_detector: MediaPipe Pose detector
        skip_frames: Number of frames to skip between detections
        frame_count: Current frame count
    
    Returns:
        tuple: (image, pose_detected, pose_landmarks)
    """
    if not MEDIAPIPE_AVAILABLE or pose_detector is None:
        return image, False, None
    
    # Check if we should skip this frame
    if frame_count % skip_frames != 0:
        return image, False, None
    
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        
        if results.pose_landmarks is None:
            return image, False, None
        
        return image, True, results.pose_landmarks
    
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return image, False, None


def draw_emotion_bars(image, x, y, w, h, emotions_dict, alpha_bar=0.7, bar_width=100, bar_height=140):
    """
    Draw emotion probability bars to the right of a face bounding box.
    
    Args:
        image: Input frame
        x, y, w, h: Face bounding box coordinates
        emotions_dict: Dictionary of emotions and their probabilities (0-100)
        alpha_bar: Transparency level (0-1)
        bar_width: Width of the bar chart
        bar_height: Height of the bar chart
    
    Returns:
        Modified image with emotion bars
    """
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Position bars to the right of the bounding box
    bar_x = x + w + 10
    bar_y = y
    
    # Ensure bars don't go off-screen
    if bar_x + bar_width > img_width:
        bar_x = max(10, x - bar_width - 10)
    if bar_y + bar_height > img_height:
        bar_y = max(10, img_height - bar_height)
    
    # Emotion colors (BGR format)
    emotion_colors = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 165, 0),    # Dark Green
        'fear': (128, 0, 128),     # Purple
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 255, 0),   # Green
        'neutral': (128, 128, 128) # Gray
    }
    
    # Sort emotions by name for consistent display
    sorted_emotions = sorted(emotions_dict.items())
    num_emotions = len(sorted_emotions)
    
    if num_emotions == 0:
        return image
    
    # Calculate bar spacing
    bar_spacing = bar_height / num_emotions
    bar_individual_height = int(bar_spacing * 0.85)
    
    # Create overlay surface for transparency
    overlay = image.copy()
    
    for idx, (emotion_name, emotion_value) in enumerate(sorted_emotions):
        # Calculate bar position
        current_y = int(bar_y + idx * bar_spacing + (bar_spacing - bar_individual_height) / 2)
        bar_bottom = current_y + bar_individual_height
        
        # Draw background box for each emotion bar (dark background)
        cv2.rectangle(overlay, (bar_x, current_y), (bar_x + bar_width, bar_bottom), 
                     (40, 40, 40), -1)
        
        # Draw outline (light border)
        cv2.rectangle(overlay, (bar_x, current_y), (bar_x + bar_width, bar_bottom), 
                     (200, 200, 200), 1)
        
        # Normalize emotion value (DeepFace returns 0-100)
        normalized_value = emotion_value / 100.0
        fill_width = int(normalized_value * bar_width)
        
        # Draw filled portion
        color = emotion_colors.get(emotion_name.lower(), (128, 128, 128))
        if fill_width > 0:
            cv2.rectangle(overlay, (bar_x, current_y), (bar_x + fill_width, bar_bottom), 
                         color, -1)
        
        # Draw emotion label (short abbreviation)
        label = emotion_name[:3].upper()
        cv2.putText(overlay, label, (bar_x + 3, current_y + bar_individual_height - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Blend overlay with original image based on alpha
    cv2.addWeighted(overlay, alpha_bar, image, 1 - alpha_bar, 0, image)
    
    return image


def draw_pose(image, pose_landmarks, alpha_pose=0.7):
    """
    Draw pose skeleton on the image.
    
    Args:
        image: Input frame
        pose_landmarks: MediaPipe pose landmarks
        alpha_pose: Transparency level (0-1)
    
    Returns:
        Modified image with pose skeleton
    """
    if not MEDIAPIPE_AVAILABLE or mp_pose is None or pose_landmarks is None:
        return image
    
    img_height, img_width = image.shape[:2]
    
    # Create overlay for pose
    overlay = image.copy()
    
    # Draw landmarks (joints)
    for landmark in pose_landmarks.landmark:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        
        # Only draw if landmark is visible
        if landmark.visibility > 0.5:
            # Color green for visible joints
            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 4, (255, 255, 255), 1)
    
    # Draw connections between joints
    try:
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = pose_landmarks.landmark[start_idx]
            end = pose_landmarks.landmark[end_idx]
            
            # Check visibility
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_x = int(start.x * img_width)
                start_y = int(start.y * img_height)
                end_x = int(end.x * img_width)
                end_y = int(end.y * img_height)
                
                # Draw connection line
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    except AttributeError:
        # Fallback if POSE_CONNECTIONS not available
        print("Warning: Could not access POSE_CONNECTIONS")
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha_pose, image, 1 - alpha_pose, 0, image)
    
    return image


def draw_overlays(image, emotions_data, alpha_bar=0.7):
    """
    Draw bounding boxes and emotion bars on the image.
    
    Args:
        image: Input frame
        emotions_data: List of detected faces and emotions
        alpha_bar: Transparency level for emotion bars (0-1)
    
    Returns:
        Processed image with overlays
    """
    if emotions_data is None:
        return image
    
    for face_data in emotions_data:
        x, y, w, h = face_data['coords']
        emotions = face_data['emotions']
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw emotion probability bars
        image = draw_emotion_bars(image, x, y, w, h, emotions, alpha_bar=alpha_bar)
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description='Real-time facial recognition, emotion detection, and pose estimation'
    )
    parser.add_argument(
        '--skipframes',
        type=int,
        default=1,
        help='Number of frames to skip between emotion detections (default: 1)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Frame width (default: 640)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Frame height (default: 480)'
    )
    parser.add_argument(
        '--alpha-bar',
        type=float,
        default=0.7,
        help='Opacity of emotion bars (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--alpha-pose',
        type=float,
        default=0.7,
        help='Opacity of pose skeleton (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--enable-pose',
        action='store_true',
        default=False,
        help='Enable pose estimation (default: disabled for faster processing)'
    )
    parser.add_argument(
        '--skip-pose-frames',
        type=int,
        default=1,
        help='Number of frames to skip for pose detection (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate alpha values
    if not 0.0 <= args.alpha_bar <= 1.0:
        print("Warning: alpha_bar should be between 0.0 and 1.0. Using default 0.7")
        args.alpha_bar = 0.7
    
    if not 0.0 <= args.alpha_pose <= 1.0:
        print("Warning: alpha_pose should be between 0.0 and 1.0. Using default 0.7")
        args.alpha_pose = 0.7
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera}")
        return
    
    # Set camera codec to MJPEG for better compatibility with high resolutions
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if actual_width != args.width or actual_height != args.height:
        print(f"Warning: Requested resolution {args.width}x{args.height} but camera returned {actual_width}x{actual_height}")
    else:
        print(f"Camera resolution set to {actual_width}x{actual_height}")
    
    if actual_fps != args.fps:
        print(f"Note: Requested FPS {args.fps} but camera set to {actual_fps:.1f}")
    
    # Initialize MediaPipe Pose
    pose_detector = None
    if args.enable_pose:
        if not MEDIAPIPE_AVAILABLE:
            print("Error: MediaPipe is required for pose estimation. Install with: pip install mediapipe")
            args.enable_pose = False
        else:
            pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=light, 1=full, 2=heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"Pose estimation enabled (skip frames: {args.skip_pose_frames})")
    
    # Calculate frame timing
    frame_delay = int(1000 / args.fps)  # milliseconds
    
    frame_count = 0
    start_time = time.time()
    processed_count = 0
    pose_count = 0
    last_emotions_data = None
    last_pose_landmarks = None
    
    print(f"Starting real-time emotion detection and pose estimation...")
    print(f"Skip frames (emotion): {args.skipframes}")
    print(f"Target FPS: {args.fps}")
    print(f"Emotion bar opacity: {args.alpha_bar}")
    if args.enable_pose:
        print(f"Pose skeleton opacity: {args.alpha_pose}")
    print(f"Press 'q' to quit, 'p' to toggle pose")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Process faces (with frame skipping)
            frame, faces_detected, emotions_data = process_faces(
                frame, 
                skip_frames=args.skipframes,
                frame_count=frame_count
            )
            
            # Process pose if enabled
            if args.enable_pose and pose_detector:
                frame, pose_detected, pose_landmarks = process_pose(
                    frame,
                    pose_detector,
                    skip_frames=args.skip_pose_frames,
                    frame_count=frame_count
                )
                
                if pose_detected and pose_landmarks:
                    last_pose_landmarks = pose_landmarks
                    pose_count += 1
            
            # Update last emotions data if new detection occurred
            if faces_detected and emotions_data:
                last_emotions_data = emotions_data
                processed_count += 1
            
            # Draw emotion overlays if we have emotions data
            if last_emotions_data:
                frame = draw_overlays(frame, last_emotions_data, alpha_bar=args.alpha_bar)
            
            # Draw pose overlay if we have pose data
            if args.enable_pose and last_pose_landmarks:
                frame = draw_pose(frame, last_pose_landmarks, alpha_pose=args.alpha_pose)
            
            # Add FPS counter
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                status_text = f"FPS: {fps:.1f} | Faces: {processed_count}"
                if args.enable_pose:
                    status_text += f" | Poses: {pose_count}"
                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display frame
            cv2.imshow('Emotion Detection & Pose Estimation - Press Q to Quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                if MEDIAPIPE_AVAILABLE:
                    args.enable_pose = not args.enable_pose
                    if args.enable_pose and pose_detector is None:
                        pose_detector = mp_pose.Pose(
                            static_image_mode=False,
                            model_complexity=1,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5
                        )
                        print("Pose estimation enabled")
                    elif not args.enable_pose:
                        print("Pose estimation disabled")
                else:
                    print("MediaPipe not available")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if pose_detector:
            pose_detector.close()
        
        # Print statistics
        elapsed = time.time() - start_time
        print(f"\nSession Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Face detections: {processed_count}")
        if args.enable_pose:
            print(f"Pose detections: {pose_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average FPS: {frame_count / elapsed:.1f}")


if __name__ == '__main__':
    main()