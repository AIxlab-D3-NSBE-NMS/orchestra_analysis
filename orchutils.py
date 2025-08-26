import subprocess
import json
import cv2
import pytesseract
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np


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