import pandas as pd
import pdb
import orchutils
from pathlib import Path
from datetime import datetime

readable_dates = []
readable_times = []

data = pd.read_csv('cyclesix_week1.csv', sep='\t')

for ii in range(data.shape[0]):

    date_string     = Path(data['filepath'][ii]).name[16:24]
    time_string     = Path(data['filepath'][ii]).name[25:31]

    date_obj = datetime.strptime(date_string, "%Y%m%d")
    readable_date = date_obj.strftime("%d.%m.%Y")

    time_obj = datetime.strptime(time_string, "%H%M%S")
    readable_time = time_obj.strftime("%Hh%Mm%Ss")

    readable_dates.append(readable_date)
    readable_times.append(readable_time)

data['readable_date'] = readable_dates
data['readable_time'] = readable_times

data.to_csv('cyclesix_week1_updated.csv', sep='\t', index=False)
