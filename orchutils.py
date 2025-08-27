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

    return frames_data['frames']

def get_overlay_info(video_path: Path,
                     verbose : bool = True,
                     mask : tuple[int,int,int,int] = (0,0,365,70)):
    # these are the outputs
    timestamps  = []
    frames      = []

    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # if verbose, create a progres bar
    if verbose:
        pbar = tqdm(total=num_frames)

    for i in range(num_frames):
        # read the video
        ret, frame = video.read()
        if not ret:
            break
        if verbose:
            pbar.update(1)

        # crop the frame
        crop = frame[mask[1]:mask[3], mask[0]:mask[2]]

        # feed it to tesseract
        text = pytesseract.image_to_string(crop,
                                           config='--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.:Frame ')

        timestamp_txt   = text.split('\n')[0]
        frame_txt       = text.split('\n')[1][6:]

        # timestamp_float = float(timestamp_txt.replace(',',''))
        timestamp_float = float(timestamp_txt)
        frame_int       = int(frame_txt)

        timestamps.append(timestamp_float)
        frames.append(frame_int)

    video.release()

    if verbose:
        pbar.close()

    return timestamps, frames


def get_overlay_info_easyocr(video_path: Path,
                             verbose: bool = True,
                             mask: tuple[int, int, int, int] = (0, 10, 250, 58),
                             stop_at: int = np.inf,
                             use_gpu: bool = True):

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

    frame_count = 0

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
        results = reader.readtext(thresh_crop,
                                  paragraph=False,
                                  allowlist='0123456789.:Frame',
                                  detail=0,
                                  batch_size=1)

        # Extract text with decent confidence
        if len(results) != 2:
            raise Exception("Error: wrong number of results in OCR")

        timestamp = float(results[0])
        frame = int(results[1].split(sep=':')[1])

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
