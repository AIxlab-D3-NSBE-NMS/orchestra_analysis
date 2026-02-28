# this assumes that the s3 bucket is locally mounted via mount-s3
from pathlib import Path
import pandas as pd
import numpy as np

# df = pd.DataFrame(  columns=['station', 'stream', 'video', 'frames'],
#                     data = [])
df_rows = []

bucket_folder_local_path = Path('/data/awsbucket/stress_test')
for station in bucket_folder_local_path.iterdir():
    if station.is_dir():
        #print(station.name) # this is a full path
        for stream in station.iterdir():
            if stream.is_dir():
                # print(stream)
                for video in stream.iterdir():
                    if video.suffix == '.mp4':
                        data_row = [station.name, stream.name, str(video.resolve()), np.nan]
                        df_rows.append(data_row)

df = pd.DataFrame(df_rows, columns=['station', 'stream', 'video', 'frames'])
df.to_csv('stress_test_number_of_frames.csv', sep='\t', index=False)
