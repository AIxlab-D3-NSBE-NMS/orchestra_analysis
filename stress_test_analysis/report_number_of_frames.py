import pandas as pd
import pdb
import cv2
import orchutils

data = pd.read_csv('stress_test_analysis/filtered_and_paired.csv', sep='\t', index_col=None)

data['frames'] = data['frames'].astype(int)
#data['pair_id'] = data['pair_id'].astype(int)

corrupted = '/media/labadmin/Windows/Users/diogo/aixlab/data/stress_test/aixlab-10/owl/2026-01-22_18-35-03-379501.mp4'
preserved = '/media/labadmin/Windows/Users/diogo/aixlab/data/stress_test/aixlab-10/owl/2026-01-22_17-19-15-617741.mp4'

orchutils.count_frames(corrupted)



breakpoint()
