import os
import orchutils
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

media_folder = '/home/labadmin/Desktop/video_test/media/August19th'
owl_folder = os.path.join(media_folder, 'owl')
webcam_folder = os.path.join(media_folder, 'webcam')
screen_folder = os.path.join(media_folder, 'screen')

owl_file = os.path.join(owl_folder, os.listdir(owl_folder)[-1])
webcam_file = os.path.join(webcam_folder, os.listdir(webcam_folder)[-1])
screen_file = os.path.join(screen_folder, os.listdir(screen_folder)[-1])

# live_flv recommmended instead of use_wallclock_as_timestamps


# Run ffprobe and capture the output
print('getting owl frame info ... ', end='')
tic = time.perf_counter()
owl_frames = orchutils.ffprobe_frame_info(owl_file)
toc = time.perf_counter()
print('Elapsed time: {:.1f} seconds'.format(toc-tic))
# pts	                        Presentation Time Stamp — when the frame should be presented (in stream timebase units).
# pts_time	                    Same as pts, but converted to seconds (float).
# pkt_dts	                    Decode Time Stamp — when the frame should be decoded (in stream timebase units).
# pkt_dts_time	                Same as pkt_dts, but in seconds.
# best_effort_timestamp	        FFmpeg's best guess for the frame's timestamp when pts or dts is missing or unreliable.
# best_effort_timestamp_time	Same as above, but in seconds.
# pkt_duration	                Duration of the packet (in timebase units).
# pkt_duration_time	            Duration of the packet in seconds.
# duration	                    Duration of the frame (may be missing or unreliable).
# duration_time	                Duration in seconds.

print('getting webcam frame info ... ', end='')
tic = time.perf_counter()
webcam_frames = orchutils.ffprobe_frame_info(webcam_file)
toc = time.perf_counter()
print('Elapsed time: {:.1f} seconds'.format(toc-tic))

print('getting screen frame info ... ', end='')
tic = time.perf_counter()
screen_frames = orchutils.ffprobe_frame_info(screen_file)
toc = time.perf_counter()
print('Elapsed time: {:.1f} seconds'.format(toc-tic))

owl_pd = pd.DataFrame(owl_frames)
webcam_pd = pd.DataFrame(webcam_frames)
screen_pd = pd.DataFrame(screen_frames)

owl_pd.media_type = 'owl'
webcam_pd.media_type = 'webcam'
screen_pd.media_type = 'screen'

owl_pd['frame'] = np.arange(len(owl_frames))
webcam_pd['frame'] = np.arange(len(webcam_frames))
screen_pd['frame'] = np.arange(len(screen_frames))

data = pd.concat([owl_pd, webcam_pd, screen_pd], ignore_index=True)
columns_to_convert = [
    'pts_time',
    'pts',
    'pkt_dts',
    'pkt_dts_time',
    'best_effort_timestamp',
    'best_effort_timestamp_time',
    'pkt_duration',
    'pkt_duration_time',
    'duration',
    'duration_time']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col])


plt.figure()
sns.scatterplot(data=data, x='frame', y='pts', hue='media_type')
plt.title("Timestamps: Owl vs Webcam")
plt.ylabel("Timestamp (seconds)")
plt.xlabel("Frame Index")
plt.legend(title="Media Type")
plt.tight_layout()
plt.show()
