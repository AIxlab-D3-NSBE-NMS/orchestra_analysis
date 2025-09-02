import time

import pandas as pd
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from vidgear.gears import VideoGear
import cv2
from tqdm import tqdm
import orchutils
import numpy as np
from pathlib import Path
import skimage
import seaborn as sns

# orchutils.estimate_block_memory(10000, height=70, width=365)
# data = orchutils.extract_timestamp_block(vid_path)
# np.save(Path(vid_path).parent.joinpath('screen_timestamp_block.npy'), data)
# np.savez_compressed(Path(vid_path).parent.joinpath('screen_timestamp_block_comp.npy'), data)
#data = np.load(Path(vid_path).parent.joinpath('screen_timestamp_block.npy'))

# timestamps, frames = orchutils.get_overlay_info(vid_path)
#timestamps, frames = orchutils.get_overlay_info_easyocr(vid_path)

# timestamps, frames = orchutils.easyocr_timestamp_block(data, frame_dim=0)
vid_path = '/home/labadmin/Desktop/video_test/media/August25th/webcam_2025-08-25_16-44-21-212217.mp4'
# timestamps, frames = orchutils.get_overlay_info_easyocr(vid_path)

# -----------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------

basePath = '/home/labadmin/Desktop/video_test/media/August25th'
#
# frames_owl = np.load(Path(basePath) / 'owl_frames.npy')
# frames_screen = np.load(Path(basePath) / 'screen_frames.npy')
# frames_webcam = np.load(Path(basePath) / 'webcam_frames.npy')
#
# timestamps_owl = np.load(Path(basePath) / 'owl_timestamps.npy')
# timestamps_screen = np.load(Path(basePath) / 'screen_timestamps.npy')
# timestamps_webcam = np.load(Path(basePath) / 'webcam_timestamps.npy')
#
# plt.figure()
# plt.scatter(frames_screen, timestamps_screen)
# plt.scatter(frames_owl, timestamps_owl)
# plt.show()

# vid_path = Path(basePath) / 'screen_2025-08-25_16-44-21-602573.mp4'
# __, __, ttl_screen = orchutils.extract_from_frames(vid_path,
#                                     vars=['ttl'],
#                                     ttl_roi = [400, 200, 600, 400])
# vid_path = Path(basePath) / 'owl_2025-08-25_16-44-21-208902.mp4'
# __, __, ttl_owl = orchutils.extract_from_frames(vid_path,
#                                     vars=['ttl'],
#                                     ttl_roi = (1583, 379, 1886, 502))
# vid_path = Path(basePath) / 'webcam_2025-08-25_16-44-21-212217.mp4'
# __, __, ttl_webcam = orchutils.extract_from_frames(vid_path,
#                                     vars=['ttl'],
#                                     ttl_roi = (323, 220, 390, 460))


owl = {'frames': np.load(Path(basePath) / 'owl_frames.npy'),
       'timestamps': np.load(Path(basePath) / 'owl_timestamps.npy'),
       'ttl': np.load(Path(basePath) / 'owl_ttl.npy'),}
screen = {'frames': np.load(Path(basePath) / 'screen_frames.npy'),
          'timestamps': np.load(Path(basePath) / 'screen_timestamps.npy'),
          'ttl': np.load(Path(basePath) / 'screen_ttl.npy'),}
webcam = {'frames': np.load(Path(basePath) / 'webcam_frames.npy'),
          'timestamps': np.load(Path(basePath) / 'webcam_timestamps.npy'),
          'ttl': np.load(Path(basePath) / 'webcam_ttl.npy'),}

owl_df = pd.DataFrame.from_dict(owl)
screen_df = pd.DataFrame.from_dict(screen)
webcam_df = pd.DataFrame.from_dict(webcam)

owl_df['time_as_float'] = owl_df.timestamps.transform(orchutils.hhmmss_to_float)
screen_df['time_as_float'] = screen_df.timestamps.transform(orchutils.hhmmss_to_float)
# webcam_df['time_as_float'] = webcam_df.timestamps.transform(orchutils.hhmmss_to_float)
# for ii in webcam_df.timestamps:
#     orchutils.hhmmss_to_float(ii)
# webcam OCR data was corrupted
# test by frame number
# and test by frame number extracted by ffprobe

# orchutils.ffprobe_frame_info(Path(basePath) / 'owl_2025-08-25_16-44-21-208902.mp4')
# __, frames, __ = orchutils.extract_from_frames(
#                     Path(basePath) / 'owl_2025-08-25_16-44-21-208902.mp4',
#                     vars='frames')

import time
tic = time.perf_counter()
metadata_screen = orchutils.ffprobe_frame_info(Path(basePath) / 'screen_2025-08-25_16-44-21-602573.mp4')
toc = time.perf_counter()-tic
print(f'ffprobe took {toc:.2f} seconds')
md_screen = pd.DataFrame(metadata_screen)

tic = time.perf_counter()
metadata_webcam = orchutils.ffprobe_frame_info(Path(basePath) / 'webcam_2025-08-25_16-44-21-212217.mp4')
toc = time.perf_counter()-tic
print(f'ffprobe took {toc:.2f} seconds')
md_webcam = pd.DataFrame(metadata_webcam)

tic = time.perf_counter()
metadata_owl = orchutils.ffprobe_frame_info(Path(basePath) / 'owl_2025-08-25_16-44-21-208902.mp4')
toc = time.perf_counter()-tic
print(f'ffprobe took {toc:.2f} seconds')
md_owl = pd.DataFrame(metadata_owl)

md_screen['ttl']    = (screen['ttl'] > 130).astype(int)
md_webcam['ttl']    = (webcam['ttl'] > 32).astype(int)
md_owl['ttl']       = (owl['ttl'] > 100).astype(int)

md_screen['best_effort_timestamp_time'] = md_screen['best_effort_timestamp_time'].transform(float)
md_owl['best_effort_timestamp_time'] = md_owl['best_effort_timestamp_time'].transform(float)
md_webcam['best_effort_timestamp_time'] = md_webcam['best_effort_timestamp_time'].transform(float)

md_screen['wallclock'] = screen['timestamps']
md_owl['wallclock'] = owl['timestamps']


# timestamp read from metadata
sns.lineplot(data=md_screen, x='best_effort_timestamp_time', y='ttl')
sns.lineplot(data=md_owl, x='best_effort_timestamp_time', y='ttl')
sns.lineplot(data=md_webcam, x='best_effort_timestamp_time', y='ttl')


# timestamp from system time embedded in OCR
sns.lineplot(data=md_screen, x='wallclock', y='ttl')
sns.lineplot(data=md_owl, x='wallclock', y='ttl')

rising_edges_screen = md_screen.wallclock[np.hstack((False,np.diff(md_screen.ttl)==-1))]
rising_edges_owl = md_owl.wallclock[np.hstack((False,np.diff(md_owl.ttl)==-1))]

offset = rising_edges_owl[:300].values - rising_edges_screen[:300].values

pd.Series(offset).describe()



# timestamp read from OCR the embedded system time

