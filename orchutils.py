import subprocess
import json
import cv2
import pytesseract
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import easyocr
from matplotlib import pyplot as plt
import warnings
import h5py
# import time
from datetime import datetime, time, timedelta
from typing import Union

import orchutils


def ffprobe_frame_info(input_file) -> list:
    result = subprocess.run(
        ['ffprobe', '-select_streams', 'v', '-show_frames', '-print_format', 'json', input_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    frames_data = None
    # Check for errors
    if result.returncode != 0:
        raise Exception("Error:", result.stderr)
    else:
        # Parse the JSON output
        frames_data = json.loads(result.stdout)

    terminaloutput = subprocess.run(['stat', f'{input_file}'], capture_output=True)
    parse_idx = str(terminaloutput.stdout).find('Birth: ')
    date_time_str = str(terminaloutput.stdout)[parse_idx+7:parse_idx+33]
    dt = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S.%f")

    for ii in tqdm(frames_data):
        frame_seconds = float(ii['pts_time'])
        dt_new = dt + timedelta(seconds=frame_seconds)
        frame_seconds_since_midnight = (
                    dt_new - datetime.combine(dt_new.date(), time(0))).total_seconds()
        ii['pts_absolute'] = frame_seconds_since_midnight
        ii['pts_timeobj']  = dt_new.time()

    return frames_data['frames']

def get_overlay_info_easyocr(video_path: Path,
                             verbose: bool = True,
                             stop_at: int = np.inf,
                             use_gpu: bool = True):
    mask: tuple[int, int, int, int] = (0, 10, 250, 58)
    timestamps = []
    frames = []

    # Initialize EasyOCR reader (one-time cost)
    if verbose:
        print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

    # Open video
    video = cv2.VideoCapture(str(video_path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames = min(num_frames, stop_at) if stop_at != np.inf else num_frames

    # Pre-extract mask coordinates
    top, bottom, left, right = mask[1], mask[3], mask[0], mask[2]

    if verbose:
        pbar = tqdm(total=actual_frames)

    for frame_count in range(actual_frames):
        ret, frame = video.read()
        if not ret:
            continue
            warnings.warn(f"Frame {frame_count} of {actual_frames} is invalid.")
        thresh_crop = None
        # Crop the frame
        crop = frame[top:bottom, left:right]

        # Convert to grayscale for faster OCR
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        __, thresh_crop  = cv2.threshold(gray_crop, 245, 255, cv2.THRESH_BINARY)

        # EasyOCR returns list of (bbox, text, confidence)
        hhmmss = reader.readtext(thresh_crop[0:18,10:110],
                                  paragraph=False,
                                  allowlist='0123456789',
                                  detail=0,
                                  batch_size=1)
        ms = reader.readtext(thresh_crop[0:18,120:170],
                                  paragraph=False,
                                  allowlist='0123456789',
                                  detail=0,
                                  batch_size=1)
        frame_str = reader.readtext(thresh_crop[30:,105:250],
                             paragraph=False,
                             allowlist='0123456789',
                             detail=0,
                             batch_size=1)

        timestamp = float(hhmmss[0] + '.' + ms[0])
        frame = int(frame_str[0])

        timestamps.append(timestamp)
        frames.append(frame)

        frame_count += 1

        if verbose:
            pbar.update(1)

    video.release()

    if verbose:
        pbar.close()

    return timestamps, frames


def extract_timestamp_block(video_path: Path,
                            mask: tuple[int, int, int, int] = (0, 0, 365, 70),
                            stop_at: int = np.inf):
    video = cv2.VideoCapture(str(video_path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames = min(num_frames, stop_at) if stop_at != np.inf else num_frames

    # Calculate crop dimensions
    top, bottom, left, right = mask[1], mask[3], mask[0], mask[2]
    height = bottom - top  # 70
    width = right - left  # 365

    # Pre-allocate 3D array: (frames, height, width)
    timestamp_block = np.empty((actual_frames, height, width), dtype=np.uint8)

    frame_idx = 0
    pbar = tqdm(total=actual_frames, desc="Extracting frames")
    while frame_idx < actual_frames:
        ret, frame = video.read()
        if not ret:
            break

        # Crop and convert to grayscale
        crop = frame[top:bottom, left:right]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Store directly in pre-allocated array
        timestamp_block[frame_idx] = gray_crop
        frame_idx += 1
        pbar.update(1)

    video.release()

    # Trim if we stopped early
    if frame_idx < actual_frames:
        timestamp_block = timestamp_block[:frame_idx]

    return timestamp_block
def estimate_block_memory(num_frames, height=70, width=365):
    bytes_needed = num_frames * height * width  # uint8 = 1 byte per pixel
    gb_needed = bytes_needed / (1024**3)
    print(f"Memory needed: {gb_needed:.2f} GB for {num_frames} frames")
    return gb_needed


def tesseract_timestamp_block(timestamp_block, frame_dim=0):
    timestamps = []
    frames = []
    for i in tqdm(range(timestamp_block.shape[frame_dim]), desc="OCR Processing"):
        frame_crop = timestamp_block[i]

        # Tesseract
        text = pytesseract.image_to_string(
            frame_crop,
            config='--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.:Frame '
        )

    return timestamps, frames

def easyocr_timestamp_block(timestamp_block, frame_dim=0):
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    timestamps = []
    frames = []
    for i in tqdm(range(timestamp_block.shape[frame_dim]), desc="OCR Processing"):
        frame_crop = timestamp_block[i]

        # EasyOCR
        results = reader.readtext(frame_crop,
                                  paragraph=False,
                                  allowlist='0123456789.:Frame ',
                                  detail=0,
                                  batch_size=32)
        # batch_size doesn't make that much of a difference
        timestamp   = float(results[0])
        frame       = int(results[1][6:])

        timestamps.append(timestamp)
        frames.append(frame)

    return timestamps, frames

def show_overlay(video_path: Path,
                 mask: tuple[int, int, int, int] = (0, 0, 365, 70),
                 frame=0):
    video = cv2.VideoCapture(str(video_path))
    top, bottom, left, right = mask[1], mask[3], mask[0], mask[2]
    for ii in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        __, frameimg = video.read()
        if ii >= frame:
            break
    video.release()
    crop = frameimg[top:bottom, left:right]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_crop)
    plt.show()
    return gray_crop


def create_compressed_video_hdf5(video_path: Path,
                                 output_path: Path = None,
                                 chunk_size: int = 100,
                                 compression_level: int = 1,
                                 stop_at: int = np.inf):
    """
    Convert video to compressed HDF5 format for fast random access

    Args:
        video_path: Path to input video
        output_path: Path for HDF5 file (defaults to video_path.h5)
        mask: Crop region (left, top, right, bottom)
        chunk_size: Frames per chunk (affects random access speed vs compression)
        compression_level: 1-9, higher = smaller file but slower access
        stop_at: Max frames to process
    """

    video_path = Path(video_path) # force cast

    if output_path is None:
        output_path = video_path.with_suffix('.h5')

    # Open video and get info
    video = cv2.VideoCapture(str(video_path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    actual_frames = min(num_frames, stop_at) if stop_at != np.inf else num_frames


    print(f"Converting video to HDF5:")
    print(f"  Input: {video_path}")
    print(f"  Output: {output_path}")
    print(f"  Frames: {actual_frames}")
    print(f"  Crop size: {width}x{height}")
    print(f"  Chunk size: {chunk_size} frames")
    print(f"  Compression level: {compression_level}")

    # Estimate file size
    uncompressed_size = actual_frames * height * width
    estimated_compressed = uncompressed_size * 0.1  # Rough estimate
    print(f"  Estimated size: {estimated_compressed / 1024 ** 3:.1f} GB")

    with h5py.File(output_path, 'w') as f:
        # Create the main dataset with chunking and compression
        dataset = f.create_dataset(
            'frames',
            shape=(actual_frames, height, width),
            dtype=np.uint8,
            chunks=(chunk_size, height, width),  # Chunk for random access
            compression='gzip',  # Good compression ratio
            compression_opts=compression_level,  # 1=fast, 9=small
            shuffle=True,  # Reorder bytes for better compression
            fletcher32=True  # Checksum for data integrity
        )

        # Store metadata
        f.attrs['video_path'] = str(video_path)
        f.attrs['fps'] = fps
        f.attrs['original_frames'] = num_frames
        f.attrs['chunk_size'] = chunk_size
        f.attrs['compression_level'] = compression_level

        # Fill dataset
        print("Processing frames...")
        start_time = time.time()

        for i in tqdm(range(actual_frames)):
            ret, frame = video.read()
            if not ret:
                print(f"Warning: Video ended at frame {i}")
                break

            # Crop and convert to grayscale
            gray_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Store in dataset
            dataset[i] = gray_crop

            # Periodic flush for large datasets
            if i % 10000 == 0 and i > 0:
                f.flush()

        # Final flush
        f.flush()

    video.release()

    # Report results
    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size

    print(f"\nConversion complete!")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  File size: {file_size / 1024 ** 3:.2f} GB")
    print(f"  Compression ratio: {uncompressed_size / file_size:.1f}x")
    print(f"  Speed: {actual_frames / elapsed:.1f} frames/second")

    return output_path


def load_hdf5_video(hdf5_path: Path):
    """Load HDF5 video file and return dataset handle"""
    f = h5py.File(hdf5_path, 'r')

    # Print info
    frames_dataset = f['frames']
    print(f"Loaded HDF5 video:")
    print(f"  Shape: {frames_dataset.shape}")
    print(f"  Chunk size: {frames_dataset.chunks}")
    print(f"  Compression: {frames_dataset.compression}")
    print(f"  Original video: {f.attrs.get('video_path', 'unknown')}")
    print(f"  FPS: {f.attrs.get('fps', 'unknown')}")

    return f, frames_dataset


class HDF5VideoReader:
    """Context manager for HDF5 video reading"""

    def __init__(self, hdf5_path: Path):
        self.hdf5_path = hdf5_path
        self.file = None
        self.frames = None

    def __enter__(self):
        self.file, self.frames = load_hdf5_video(self.hdf5_path)
        return self.frames

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def benchmark_hdf5_access(hdf5_path: Path, num_tests: int = 1000):
    """Benchmark random access performance"""

    with HDF5VideoReader(hdf5_path) as frames:
        total_frames = frames.shape[0]

        # Test sequential access
        print("Testing sequential access...")
        start_time = time.time()
        for i in range(min(1000, total_frames)):
            frame = frames[i]
        sequential_time = time.time() - start_time

        # Test random access
        print("Testing random access...")
        random_indices = np.random.randint(0, total_frames, num_tests)

        start_time = time.time()
        for idx in random_indices:
            frame = frames[idx]
        random_time = time.time() - start_time

        # Test slice access
        print("Testing slice access...")
        start_time = time.time()
        for _ in range(100):
            start_idx = np.random.randint(0, total_frames - 100)
            slice_frames = frames[start_idx:start_idx + 100]
        slice_time = time.time() - start_time

        print(f"\nPerformance Results:")
        print(f"  Sequential access: {sequential_time / 1000 * 1000:.2f} ms/frame")
        print(f"  Random access: {random_time / num_tests * 1000:.2f} ms/frame")
        print(f"  Slice access (100 frames): {slice_time / 100:.2f} ms/slice")


def process_hdf5_frames_ocr(hdf5_path: Path,
                            frame_indices: list = None,
                            use_easyocr: bool = True):
    """Process specific frames with OCR"""

    try:
        if use_easyocr:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        else:
            import pytesseract
    except ImportError as e:
        print(f"OCR library not available: {e}")
        return None, None

    timestamps = []
    frame_numbers = []

    with HDF5VideoReader(hdf5_path) as frames:
        total_frames = frames.shape[0]

        # Default to all frames if none specified
        if frame_indices is None:
            frame_indices = range(total_frames)

        print(f"Processing {len(frame_indices)} frames with {'EasyOCR' if use_easyocr else 'Tesseract'}...")

        for i in tqdm(frame_indices):
            if i >= total_frames:
                continue

            frame = frames[i]

            try:
                if use_easyocr:
                    results = reader.readtext(frame, paragraph=False)
                    text_parts = [text.strip() for bbox, text, conf in results if conf > 0.3]

                    if len(text_parts) >= 2:
                        timestamp_txt = text_parts[0].replace(',', '').replace('O', '0')
                        frame_txt = text_parts[1]

                        if frame_txt.lower().startswith('frame'):
                            frame_txt = frame_txt[5:].strip()
                        frame_txt = frame_txt.replace(',', '').replace('O', '0')

                        timestamps.append(float(timestamp_txt))
                        frame_numbers.append(int(frame_txt))
                    else:
                        timestamps.append(0.0)
                        frame_numbers.append(0)

                else:
                    text = pytesseract.image_to_string(
                        frame,
                        config='--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.,:Frame '
                    )

                    lines = text.split('\n', 2)
                    if len(lines) >= 2:
                        timestamp_float = float(lines[0].replace(',', ''))
                        frame_txt = lines[1][6:] if lines[1].startswith('Frame ') else lines[1]
                        frame_int = int(frame_txt.replace(',', ''))

                        timestamps.append(timestamp_float)
                        frame_numbers.append(frame_int)
                    else:
                        timestamps.append(0.0)
                        frame_numbers.append(0)

            except (ValueError, IndexError):
                timestamps.append(0.0)
                frame_numbers.append(0)

    return timestamps, frame_numbers


def convert_and_process_workflow(video_path: Path,
                                 mask: tuple[int, int, int, int] = (0, 0, 365, 70),
                                 force_recreate: bool = False):
    """Complete workflow: convert to HDF5 and process with OCR"""

    hdf5_path = video_path.with_suffix('.h5')

    # Step 1: Create HDF5 if needed
    if force_recreate or not hdf5_path.exists():
        print("Creating HDF5 compressed video...")
        create_compressed_video_hdf5(
            video_path,
            hdf5_path,
            mask=mask,
            chunk_size=1000,
            compression_level=1  # Fast compression
        )
    else:
        print(f"Using existing HDF5: {hdf5_path}")

    # Step 2: Benchmark access speed
    print("\nBenchmarking access performance...")
    benchmark_hdf5_access(hdf5_path)

    # Step 3: Process with OCR (sample first)
    print("\nProcessing sample frames with OCR...")
    sample_indices = list(range(0, 1000, 10))  # Every 10th frame for first 1000

    timestamps, frames = process_hdf5_frames_ocr(
        hdf5_path,
        frame_indices=sample_indices,
        use_easyocr=True
    )

    print(f"Sample results:")
    print(f"  Processed: {len(timestamps)} frames")
    success_rate = sum(1 for t, f in zip(timestamps, frames) if t > 0 and f > 0) / len(timestamps) * 100
    print(f"  Success rate: {success_rate:.1f}%")

    return hdf5_path, timestamps, frames

def extract_from_frames(video_path: Path,
                        vars: list = ['frames', 'timestamps', 'ttl'],
                        verbose: bool = True,
                        stop_at: int = np.inf,
                        use_gpu: bool = True,
                        ttl_roi: tuple[int, int, int, int] = (0, 10, 250, 58)):
    mask: tuple[int, int, int, int] = (0, 10, 250, 58)
    timestamps  = []
    frames      = []
    ttl         = []

    # Initialize EasyOCR reader (one-time cost)
    if verbose:
        print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

    # Open video
    video = cv2.VideoCapture(str(video_path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames = min(num_frames, stop_at) if stop_at != np.inf else num_frames

    # Pre-extract mask coordinates
    top, bottom, left, right = mask[1], mask[3], mask[0], mask[2]
    topTTL, bottomTTL, leftTTL, rightTTL = ttl_roi[1], ttl_roi[3], ttl_roi[0], ttl_roi[2]
    # reads as left, top, right, bottom

    if verbose:
        pbar = tqdm(total=actual_frames)

    for frame_count in range(actual_frames):
        ret, frame = video.read()
        if not ret:
            warnings.warn(f"Frame {frame_count} of {actual_frames} is invalid.")
            continue
        thresh_crop = None
        # Crop the frame
        crop = frame[top:bottom, left:right]
        # Convert to grayscale for faster OCR
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if 'timestamps' in vars:
            __, thresh_crop  = cv2.threshold(gray_crop, 245, 255, cv2.THRESH_BINARY)

            # EasyOCR returns list of (bbox, text, confidence)
            hhmmss = reader.readtext(thresh_crop[0:18,10:110],
                                      paragraph=False,
                                      allowlist='0123456789',
                                      detail=0,
                                      batch_size=1)
            ms = reader.readtext(thresh_crop[0:18,120:170],
                                      paragraph=False,
                                      allowlist='0123456789',
                                      detail=0,
                                      batch_size=1)
            timestamp = float(hhmmss[0] + '.' + ms[0])
            timestamps.append(timestamp)

        if 'frames' in vars:
            frame_str = reader.readtext(thresh_crop[30:,105:250],
                                 paragraph=False,
                                 allowlist='0123456789',
                                 detail=0,
                                 batch_size=1)
            frame = int(frame_str[0])
            frames.append(frame)

        if 'ttl' in vars:
            ttl_crop = frame[topTTL:bottomTTL, leftTTL:rightTTL]
            ttl_grayscale = cv2.cvtColor(ttl_crop, cv2.COLOR_BGR2GRAY)
            ttl.append(float(np.mean(ttl_grayscale)))

        frame_count += 1


        if verbose:
            pbar.update(1)

    video.release()

    if verbose:
        pbar.close()

    # return timestamps, frames, ttl


def hhmmss_to_datetime(time_value: Union[float, str], base_date: str = None) -> datetime:
    """
    Convert HHMMSS.XXX format to datetime object.

    Args:
        time_value: Time in format HHMMSS.XXX (e.g., 164423.445)
        base_date: Base date as string 'YYYY-MM-DD'. If None, uses today's date.

    Returns:
        datetime object
    """
    # Convert to string and handle formatting
    time_str = f"{float(time_value):.3f}".zfill(10)  # Ensure proper padding

    # Extract components
    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    milliseconds = int(time_str[7:10])

    # Convert milliseconds to microseconds
    microseconds = milliseconds * 1000

    # Create time object
    time_obj = time(hours, minutes, seconds, microseconds)

    # Create datetime with base date (default to today)
    if base_date is None:
        base_date = datetime.now().date()
    else:
        base_date = datetime.strptime(base_date, '%Y-%m-%d').date()

    return datetime.combine(base_date, time_obj)


def hhmmss_to_time_only(time_value: Union[float, str]) -> time:
    """
    Convert HHMMSS.XXX format to time object only.

    Args:
        time_value: Time in format HHMMSS.XXX (e.g., 164423.445)

    Returns:
        time object
    """
    time_str = f"{float(time_value):.3f}".zfill(10)

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    milliseconds = int(time_str[7:10])
    microseconds = milliseconds * 1000

    return time(hours, minutes, seconds, microseconds)


def hhmmss_to_timedelta(time_value: Union[float, str]) -> timedelta:
    """
    Convert HHMMSS.XXX format to timedelta (duration from midnight).

    Args:
        time_value: Time in format HHMMSS.XXX (e.g., 164423.445)

    Returns:
        timedelta object
    """
    time_str = f"{float(time_value):.3f}".zfill(10)

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    milliseconds = int(time_str[7:10])

    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

def hhmmss_to_float(time_value: Union[float, str]) -> float:
    a = orchutils.hhmmss_to_timedelta(time_value)
    return float(a.seconds + a.microseconds/1e6)