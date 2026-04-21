import cv2
import argparse
import time
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

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
                dominant_emotion = emotions[0]['dominant_emotion']
                
                emotions_data.append({
                    'coords': (x, y, w, h),
                    'emotion': dominant_emotion
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

def draw_overlays(image, emotions_data):
    """
    Draw bounding boxes and emotion labels on the image.
    
    Args:
        image: Input frame
        emotions_data: List of detected faces and emotions
    
    Returns:
        Processed image with overlays
    """
    if emotions_data is None:
        return image
    
    for face_data in emotions_data:
        x, y, w, h = face_data['coords']
        emotion = face_data['emotion']
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw emotion text with background
        text = emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Text background
        cv2.rectangle(image, (x, y - 30), (x + text_size[0] + 5, y - 5), (0, 255, 0), -1)
        
        # Text
        cv2.putText(image, text, (x + 2, y - 10), font, font_scale, (0, 0, 0), font_thickness)
    
    return image

def main():
    parser = argparse.ArgumentParser(
        description='Real-time facial recognition and emotion detection'
    )
    parser.add_argument(
        '--skipframes',
        type=int,
        default=1,
        help='Number of frames to skip between detections (default: 1, means detect every frame)'
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
    
    args = parser.parse_args()
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Calculate frame timing
    frame_delay = int(1000 / args.fps)  # milliseconds
    
    frame_count = 0
    start_time = time.time()
    processed_count = 0
    
    print(f"Starting real-time emotion detection...")
    print(f"Skip frames: {args.skipframes}")
    print(f"Target FPS: {args.fps}")
    print(f"Press 'q' to quit")
    
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
            
            # Draw overlays if faces were detected
            if faces_detected and emotions_data:
                frame = draw_overlays(frame, emotions_data)
                processed_count += 1
            
            # Add FPS counter
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Detections: {processed_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display frame
            cv2.imshow('Emotion Detection - Press Q to Quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed = time.time() - start_time
        print(f"\nSession Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average FPS: {frame_count / elapsed:.1f}")

if __name__ == '__main__':
    main()