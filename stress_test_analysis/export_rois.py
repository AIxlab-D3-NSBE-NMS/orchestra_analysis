import os
from pathlib import Path
from sre_parse import parse
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import pdb

def extract_and_save_metadata(video_dir):

    # Define ROI dimensions
    roi_height = 28
    roi_width = 148

    # Get the list of video files in the directory tree
    print('Search for videos in file tree...')
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):  # Add more extensions if needed
                video_files.append(os.path.join(root, file))
    print('...done.')

    for video_path in tqdm(video_files, desc='videos'):
        # Extract the base name without extension and add 'metadata' prefix
        output_file = Path(video_path).parent / Path('metadata_' + Path(video_path).stem + '.npy')

        # Read the video and extract ROI
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue


        frames = []
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract ROI
            roi = frame[:roi_height, :roi_width]

            # Append frame to list (or add directly to numpy array)
            frames.append(roi)
            pbar.update(1)

        # Convert list of ROIs to a NumPy 3D array
        metadata_array = np.array(frames)

        # Save the 3D NumPy array to file
        print('saving metadata file...')
        np.save(output_file, metadata_array)

        print(f"Saved metadata for {video_path} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=Path)
    args = parser.parse_args()
    extract_and_save_metadata(args.video_dir)
