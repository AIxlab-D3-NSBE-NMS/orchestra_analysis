import pandas as pd
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('TkAgg')   # or 'Qt5Agg' if Tk is not available


# --------------------------------------------------------------------------- #
#  Video analysis                                                              #
# --------------------------------------------------------------------------- #

def count_frames_and_duration(video_path: Path) -> tuple[int, float]:
    """
    Count frames by iterating with cap.read() (avoids codec-dependent
    CAP_PROP_FRAME_COUNT inaccuracies) and derive duration from the
    frame count + FPS reported by the container.

    Returns
    -------
    frame_count : int
    duration_seconds : float
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   # fallback to 25 if unknown

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()

    duration = frame_count / fps if fps else 0.0
    return frame_count, duration


def get_first_frame(video_path: Path):
    """Return the first decoded frame as a BGR numpy array, or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# --------------------------------------------------------------------------- #
#  Interactive ROI selection                                                   #
# --------------------------------------------------------------------------- #

def select_roi(video_path: Path) -> dict:
    """
    Display the first frame and let the user pick an ROI using
    matplotlib's built-in RectangleSelector widget.

    Returns a dict with keys: roi_x1, roi_y1, roi_x2, roi_y2
    (all None if the user skipped or closed the window).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector, Button

    frame = get_first_frame(video_path)
    empty = dict(roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None)

    if frame is None:
        print(f"  [ROI] Could not read first frame from {video_path}")
        return empty

    print("  [ROI] Drag to draw a rectangle, then click Confirm (or Skip).")

    frame_rgb = frame[:, :, ::-1]
    result    = {}   # filled by _on_confirm

    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        fig.canvas.manager.set_window_title("ROI selection")
    except AttributeError:
        pass
    ax.imshow(frame_rgb)
    ax.set_title("Drag to select ROI, then click Confirm", fontsize=11)
    ax.axis('off')
    fig.subplots_adjust(bottom=0.10)

    selector = RectangleSelector(
        ax,
        onselect=lambda eclick, erelease: None,   # we read extents on confirm
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True,
    )

    ax_confirm = fig.add_axes([0.65, 0.01, 0.10, 0.05])
    ax_skip    = fig.add_axes([0.77, 0.01, 0.10, 0.05])
    btn_confirm = Button(ax_confirm, 'Confirm', color='limegreen')
    btn_skip    = Button(ax_skip,    'Skip',    color='tomato')

    def _on_confirm(_event):
        if not selector.extents or selector.extents == (0, 0, 0, 0):
            ax.set_title("No rectangle drawn yet – drag one first!", color='red', fontsize=11)
            fig.canvas.draw_idle()
            return
        x1, x2, y1, y2 = selector.extents
        result['roi'] = (int(x1), int(y1), int(x2), int(y2))
        plt.close(fig)

    def _on_skip(_event):
        plt.close(fig)

    btn_confirm.on_clicked(_on_confirm)
    btn_skip.on_clicked(_on_skip)

    plt.show()   # blocks until window closes

    if 'roi' not in result:
        print("  [ROI] Skipped.")
        return empty

    x1, y1, x2, y2 = result['roi']
    print(f"  [ROI] Selected: ({x1}, {y1}) → ({x2}, {y2})")
    return dict(roi_x1=x1, roi_y1=y1, roi_x2=x2, roi_y2=y2)





# --------------------------------------------------------------------------- #
#  Per-row processing                                                          #
# --------------------------------------------------------------------------- #

def process_row(row: pd.Series, basepath: Path) -> dict:
    """
    Analyse one video row:
      1. Count frames (while loop, not CAP_PROP_FRAME_COUNT).
      2. Derive duration from frame_count / fps.
      3. Show first frame for interactive ROI selection.

    Returns an updated dict ready to be written back to the DataFrame.
    """
    video_path = basepath / Path(row['filepath'])
    print(f"  Analysing: {video_path}")

    # ── 1 & 2 : frame count + duration ────────────────────────────────── #
    try:
        frame_count, duration = count_frames_and_duration(video_path)
        print(f"  Frames : {frame_count}   Duration : {duration:.3f} s")
    except IOError as exc:
        print(f"  ERROR  : {exc}")
        frame_count, duration = None, None

    # ── 3 : ROI selection ─────────────────────────────────────────────── #
    roi = select_roi(video_path)

    # ── Assemble updated row ───────────────────────────────────────────── #
    updated_row = {
        **{k: row[k] for k in row.index},   # carry all existing columns
        'frames':   str(frame_count) if frame_count is not None else None,
        'duration': str(round(duration, 6)) if duration is not None else None,
        'ROI':      f"{roi['roi_x1']},{roi['roi_y1']},{roi['roi_x2']},{roi['roi_y2']}" if roi['roi_x1'] is not None else None,
    }

    return updated_row


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    file_path = '/home/labadmin/aixlab/code/orchestra_analysis/prr/cyclesix_owl.csv'
    basepath  = Path('/media/labadmin/Windows/Users/diogo/aixlab/data/raw/')

    df = pd.read_csv(file_path, sep=',', dtype='str')
    print("Columns found:", df.columns.tolist())  # diagnostic – remove once confirmed

    # Ensure the new columns exist so iloc assignment works cleanly
    for col in ('frames', 'duration', 'ROI'):
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        # Skip rows that have already been processed (frames is populated)
        if pd.notna(row.get('frames')):
            print(f"[{idx}] Already processed – skipping.")
            continue

        print(f"\n[{idx}/{len(df) - 1}] Processing: {row['filepath']}")

        updated_row = process_row(row, basepath)

        df.iloc[idx] = pd.Series(updated_row)
        df.to_csv(file_path, sep=',', index=False)
        print(f"  Saved to {file_path}")


if __name__ == "__main__":
    main()
