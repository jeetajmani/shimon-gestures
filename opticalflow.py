import cv2
import numpy as np
import time
from collections import deque

# === Tunables ===
RESIZE_FACTOR = 0.5

# Motion scaling
FLOW_NORMALIZE = 1.5        # magnitude that maps to 100
MAX_FLOW_DISPLAY = 100.0

# Raise the floor (dead-zone for tiny noise)
BASELINE_FLOOR = 0.12       # increase if idle values are still > 0

# Use a robust statistic
USE_PERCENTILE = True
PERCENTILE = 90             # stronger motions dominate; less noise than mean

# Throttle compute for lower CPU
COMPUTE_EVERY_N = 3         # compute flow 1 out of N frames

# Light smoothing on displayed value
WINDOW_SIZE = 5

# Print once per second
PRINT_INTERVAL_SEC = 1.0

# === State ===
flow_history = deque(maxlen=WINDOW_SIZE)
last_flow_value = 0.0
sum_since_print = 0.0
count_since_print = 0
last_print_ts = time.time()
frame_idx = 0

# === Capture ===
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab frame")
    raise SystemExit

prev_small = cv2.resize(prev_frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

print("Press ESC or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    computed_this_frame = False
    if frame_idx % COMPUTE_EVERY_N == 0:
        # --- Dense optical flow (Farneback) ---
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=10,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )

        # --- Motion measure ---
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_measure = np.percentile(mag, PERCENTILE) if USE_PERCENTILE else np.mean(mag)

        # --- Raise the floor (dead-zone) ---
        flow_measure = max(flow_measure - BASELINE_FLOOR, 0.0)

        # --- Normalize to 0–100 and smooth lightly ---
        flow_value = min((flow_measure / FLOW_NORMALIZE) * 100.0, MAX_FLOW_DISPLAY)
        flow_history.append(flow_value)
        last_flow_value = float(np.mean(flow_history))

        # Update reference only when we compute (temporal downsampling)
        prev_gray = gray
        computed_this_frame = True

        # Accumulate for 1-second average print
        sum_since_print += last_flow_value
        count_since_print += 1

    # --- Print average once per second ---
    now = time.time()
    if now - last_print_ts >= PRINT_INTERVAL_SEC:
        if count_since_print > 0:
            avg_last_sec = sum_since_print / count_since_print
            print(f"Optical Flow avg (last {PRINT_INTERVAL_SEC:.0f}s): {avg_last_sec:.1f}/100")
        sum_since_print = 0.0
        count_since_print = 0
        last_print_ts = now

    # --- Display (shows last computed value) ---
    cv2.putText(frame, f"Optical Flow: {last_flow_value:.1f}/100",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    if not computed_this_frame:
        cv2.putText(frame, f"(skipping compute #{frame_idx % COMPUTE_EVERY_N})",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Optical Flow Intensity (throttled + floor)", frame)

    frame_idx += 1
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
