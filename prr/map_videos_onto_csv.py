import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pdb
def get_video_paths(input_folder, match_string=None):
    video_paths = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if match_string is None or match_string.lower() in file.lower():
                full_path = os.path.join(root, file)
                video_paths.append(full_path)
            elif os.sep in match_string and match_string.lower() in os.path.join(root, file):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)

    return video_paths

def make_csv_with_videopaths(   input_folder, output_csv,
                                match_string=None,
                                date_range=(-15,-7),
                                time_range=(-6,None),
                                date_format = "%Y%m%d",
                                time_format = "%H%M%S"):
    video_paths = get_video_paths(input_folder, match_string)

    df = pd.DataFrame(video_paths, columns=['filepath'])

    # Extract date and time from the filename
    readable_dates = []
    readable_times = []

    for path in tqdm(df['filepath']):

        base_name = Path(path).stem # only filename and without extension

        # assuming appended date and time in the format: '20261231_181735'
        date_string = base_name[date_range[0]:date_range[1]]
        time_string = base_name[time_range[0]:time_range[1]]

        date_obj = datetime.strptime(date_string, date_format)
        readable_date = date_obj.strftime("%d.%m.%Y")

        time_obj = datetime.strptime(time_string, time_format)
        readable_time = time_obj.strftime("%Hh%Mm%Ss")

        readable_dates.append(readable_date)
        readable_times.append(readable_time)

    df['readable_date'] = readable_dates
    df['readable_time'] = readable_times

    df['status_website'] = None
    df['login']	= None
    df['submitted'] = None
    df['tentativas'] =None
    df['qualtrics'] = None

    df.to_csv(output_csv, sep='\t', index=False)

if __name__ == "__main__":
    input_folder = '/media/labadmin/Windows/Users/diogo/aixlab/data/raw/'
    output_csv = 'cyclesix_owl.csv'
    match_string = 'owl/'  # Change this to None if you don't want to filter by string

    make_csv_with_videopaths(   input_folder,
                                output_csv,
                                match_string,
                                date_range=(-26,-16),
                                time_range=(-15,-7),
                                date_format = "%Y-%m-%d",
                                time_format = "%H-%M-%S"  )
