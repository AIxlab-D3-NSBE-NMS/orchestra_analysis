import pandas as pd
from vlc import Instance

# Initialize VLC instance
vlc_instance = Instance()

def open_and_input(row):
    video_path = row['filepath']
    status_website = row['status_website']
    submitted = row['submitted']
    tentativas = row['tentativas']

    # Open the video file with VLC
    player = vlc_instance.media_player_new()
    media = vlc_instance.media_new(video_path)
    player.set_media(media)
    player.play()

    try:
        login = input(f"Login for {video_path}: ")
        qualtrics = input(f"Qualtrics for {video_path}: ")

        # Update the row with new inputs
        row['login'] = login
        row['qualtrics'] = qualtrics

    finally:
        # Stop and release VLC resources
        player.stop()
        player.release()

def main():
    file_path = '/home/labadmin/aixlab/code/orchestra_analysis/prr/cyclesix_week1_updated.csv'

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        if pd.isna(row['submitted']):
            print(f"Processing video {idx} of {len(df) - 1}")

            open_and_input(row)

            # Save the updated DataFrame after each input
            df.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()
