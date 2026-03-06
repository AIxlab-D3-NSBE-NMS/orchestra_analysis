import pandas as pd
from vlc import Instance
from pathlib import Path
import pdb

# Initialize VLC instance
vlc_instance = Instance()

def open_and_input(row, basepath):
    video_path      = basepath / Path(row['filepath'])
    breakpoint()
    status_website  = row['status_website']
    login           = row['login']
    submitted       = row['submitted']
    tentativas      = row['tentativas']
    qualtrics       = row['qualtrics']


    # Open the video file with VLC
    player = vlc_instance.media_player_new()
    media = vlc_instance.media_new(video_path)
    player.set_media(media)
    player.play()

    try:
        status_website      = input(f"Status do website: ")
        login               = input(f"Login : ")
        submitted           = input(f"Conseguiu submeter: ")
        tentativas          = input(f"Tentativas: ")
        qualtrics           = input(f"Qualtrics: ")

        # Update the row with new inputs
        row['status_website']   = status_website
        row['login']            = login
        row['submitted']        = submitted
        row['tentativas']       = tentativas
        row['qualtrics']        = qualtrics

    finally:
        # Stop and release VLC resources
        player.stop()
        player.release()

def main():
    file_path = '/home/labadmin/aixlab/code/orchestra_analysis/prr/cyclesix_week1_updated.csv'
    basepath = Path('/media/labadmin/Windows/Users/diogo/aixlab/data/raw/')
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t')

    for idx, row in df.iterrows():
        if pd.isna(row['status_website']):
            print(f"Processing video {idx} of {len(df) - 1}")

            open_and_input(row, basepath)

            # Save the updated DataFrame after each input
            df.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()
