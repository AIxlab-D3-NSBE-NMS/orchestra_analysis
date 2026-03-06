import pandas as pd
from pathlib import Path
import subprocess
from subprocess import DEVNULL

def open_and_input(row, basepath):
    video_path = basepath / Path(row['filepath'])

    # Construct the VLC command to open the video with full controls
    vlc_command = ['vlc', str(video_path)]

    # Start VLC as a subprocess
    subprocess.Popen(vlc_command, stdout=DEVNULL, stderr=DEVNULL)

    # Wait for the user to finish viewing the video before proceeding
    input(f"Press Enter to continue with {video_path}...")

    # Ask questions after the video has been opened
    status_website = input(f"Status do website: ")
    login = input(f"Login : ")
    submitted = input(f"Conseguiu submeter: ")
    tentativas = input(f"Tentativas: ")
    qualtrics = input(f"Qualtrics: ")
    comments = input(f"Addicional comments: ")

    # Create a new row with updated values
    updated_row = {
        'filepath': row['filepath'],
        'readable_date': row['readable_date'],
        'readable_time': row['readable_time'],
        'status_website': status_website,
        'login': login,
        'submitted': submitted,
        'tentativas': tentativas,
        'qualtrics': qualtrics
    }

    return updated_row

def main():
    file_path = '/home/labadmin/aixlab/code/orchestra_analysis/prr/cyclesix_week1_updated.csv'
    basepath = Path('/media/labadmin/Windows/Users/diogo/aixlab/data/raw/')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t')

    updated_rows = []

    for idx, row in df.iterrows():
        if pd.isna(row['status_website']):
            print(f"Processing video {idx} of {len(df) - 1}")

            updated_row = open_and_input(row, basepath)
            updated_rows.append(updated_row)

            # Create a new DataFrame with the updated rows
            updated_df = pd.DataFrame(updated_rows)

            # Save the updated DataFrame to the same file
            updated_df.to_csv(file_path, sep='\t', index=False)

if __name__ == "__main__":
    main()
