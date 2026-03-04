from pathlib import Path
import pandas as pd
import numpy as np

bucket_folder_local_path = Path('/data/awsbucket/stress_test')
for station in bucket_folder_local_path.iterdir():
    if station.is_dir():
        #print(station.name) # this is a full path
        for stream in station.iterdir():
            if stream.is_dir():
                # print(stream)
                for video in stream.iterdir():
                    if video.suffix == '.mp4':
                        get_video_props(video.absolute(), stream.name) # video full path, owl/screen

def get_video_props(vid_full_path, stream):
    