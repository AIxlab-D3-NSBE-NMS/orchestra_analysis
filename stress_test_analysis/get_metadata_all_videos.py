# this assumes that the s3 bucket is locally mounted via mount-s3
import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
# import pdb
from tqdm import tqdm

df = pd.read_csv('stress_test_number_of_frames.csv', sep='\t')

for ii in tqdm(df['frames'].isna().index):
    cap = cv2.VideoCapture(df['video'][ii])
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    df.at[ii, 'frames'] = n_frames
    df.to_csv('stress_test_number_of_frames.csv', sep='\t', index=False)




# cap = cv2.VideoCapture(video)
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print( n_frames )
