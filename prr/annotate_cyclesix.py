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

class ROISelector:
    """
    Shows the first frame in a matplotlib window and lets the user draw one
    rectangle either by clicking-and-dragging OR clicking two corner points.

    Controls
    --------
    Left-click + drag  : draw rectangle live
    Left-click (×2)    : define two opposite corners
    'Confirm' button   : accept the current rectangle
    'Reset' button     : clear and start over
    'Skip' button      : store None for this video
    """

    def __init__(self, frame_bgr):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button

        self._plt    = plt
        self._result = None          # (x1,y1,x2,y2) or None
        self._done   = False

        # Convert BGR → RGB for display
        frame_rgb = frame_bgr[:, :, ::-1]

        self._fig, self._ax = plt.subplots(figsize=(10, 7))
        self._fig.canvas.manager.set_window_title(
            "ROI — drag or click ×2 | Confirm / Reset / Skip")
        self._ax.imshow(frame_rgb)
        self._ax.set_title("Draw a rectangle, then click Confirm", fontsize=11)
        self._ax.axis('off')

        # Rectangle patch (invisible until drawn)
        self._rect = mpatches.Rectangle(
            (0, 0), 0, 0,
            linewidth=2, edgecolor='lime', facecolor='none', visible=False)
        self._ax.add_patch(self._rect)

        # Dot marker for first single-click
        self._dot, = self._ax.plot([], [], 'go', markersize=8)

        # State
        self._pressing  = False
        self._pt1       = None
        self._pt2       = None
        self._click_pts = []

        # Buttons
        ax_confirm = self._fig.add_axes([0.65, 0.01, 0.10, 0.05])
        ax_reset   = self._fig.add_axes([0.76, 0.01, 0.10, 0.05])
        ax_skip    = self._fig.add_axes([0.87, 0.01, 0.10, 0.05])
        self._btn_confirm = Button(ax_confirm, 'Confirm', color='limegreen')
        self._btn_reset   = Button(ax_reset,   'Reset',   color='gold')
        self._btn_skip    = Button(ax_skip,    'Skip',    color='tomato')
        self._btn_confirm.on_clicked(self._on_confirm)
        self._btn_reset.on_clicked(self._on_reset)
        self._btn_skip.on_clicked(self._on_skip)

        # Mouse events
        self._fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self._fig.canvas.mpl_connect('motion_notify_event',  self._on_move)
        self._fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._fig.canvas.mpl_connect('close_event',          self._on_close)

    # ------------------------------------------------------------------ #

    def _update_rect(self):
        if self._pt1 and self._pt2:
            x1 = min(self._pt1[0], self._pt2[0])
            y1 = min(self._pt1[1], self._pt2[1])
            w  = abs(self._pt2[0] - self._pt1[0])
            h  = abs(self._pt2[1] - self._pt1[1])
            self._rect.set_xy((x1, y1))
            self._rect.set_width(w)
            self._rect.set_height(h)
            self._rect.set_visible(True)
        else:
            self._rect.set_visible(False)
        self._fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        self._pressing = True
        self._pt1 = (event.xdata, event.ydata)
        self._pt2 = None
        self._click_pts = [(event.xdata, event.ydata)]
        self._dot.set_data([], [])
        self._update_rect()

    def _on_move(self, event):
        if not self._pressing or event.inaxes != self._ax:
            return
        self._pt2 = (event.xdata, event.ydata)
        self._update_rect()

    def _on_release(self, event):
        if not self._pressing or event.inaxes != self._ax or event.button != 1:
            self._pressing = False
            return
        self._pressing = False
        self._pt2 = (event.xdata, event.ydata)

        drag_dist = (abs(self._pt2[0] - self._pt1[0]) +
                     abs(self._pt2[1] - self._pt1[1]))

        if drag_dist > 5:
            # Valid drag – rectangle is complete
            self._click_pts = []
            self._dot.set_data([], [])
        else:
            # Treat as a single click
            self._click_pts.append((event.xdata, event.ydata))
            if len(self._click_pts) == 2:
                self._pt1 = self._click_pts[0]
                self._pt2 = self._click_pts[1]
                self._click_pts = []
                self._dot.set_data([], [])
            else:
                # Waiting for second click – show marker, clear rect
                self._pt2 = None
                self._dot.set_data([self._pt1[0]], [self._pt1[1]])
                self._rect.set_visible(False)
                self._fig.canvas.draw_idle()
                return

        self._update_rect()

    def _on_confirm(self, _event):
        if self._pt1 and self._pt2:
            x1 = int(min(self._pt1[0], self._pt2[0]))
            y1 = int(min(self._pt1[1], self._pt2[1]))
            x2 = int(max(self._pt1[0], self._pt2[0]))
            y2 = int(max(self._pt1[1], self._pt2[1]))
            self._result = (x1, y1, x2, y2)
            self._done = True
            self._plt.close(self._fig)
        else:
            self._ax.set_title("No rectangle drawn yet – draw one first!",
                               color='red', fontsize=11)
            self._fig.canvas.draw_idle()

    def _on_reset(self, _event):
        self._pt1 = self._pt2 = None
        self._click_pts = []
        self._pressing  = False
        self._dot.set_data([], [])
        self._rect.set_visible(False)
        self._ax.set_title("Draw a rectangle, then click Confirm", fontsize=11, color='black')
        self._fig.canvas.draw_idle()

    def _on_skip(self, _event):
        self._result = None
        self._done   = True
        self._plt.close(self._fig)

    def _on_close(self, _event):
        self._done = True   # treat window-close as skip

    # ------------------------------------------------------------------ #

    def select(self) -> tuple[int, int, int, int] | None:
        self._plt.show()    # blocks until window is closed
        return self._result


def select_roi(video_path: Path) -> dict:
    """
    Display the first frame and let the user pick an ROI.

    Returns a dict with keys: roi_x1, roi_y1, roi_x2, roi_y2
    (all None if the user skipped).
    """
    frame = get_first_frame(video_path)
    empty = dict(roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None)  # internal keys

    if frame is None:
        print(f"  [ROI] Could not read first frame from {video_path}")
        return empty

    print("  [ROI] Draw a rectangle on the frame.")
    print("        Drag  OR  click two corners | ENTER=confirm  R=reset  Q=skip")

    selector = ROISelector(frame)
    result   = selector.select()

    if result is None:
        print("  [ROI] Skipped.")
        return empty

    x1, y1, x2, y2 = result
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
