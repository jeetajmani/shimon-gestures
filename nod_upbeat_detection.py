# nose_tip_events.py
# Requires: pip install opencv-python mediapipe numpy

import time
import cv2
import numpy as np
import mediapipe as mp

# tunables
NOD_VEL_THRESH     = +0.30   # what velocity counts as a nod
UPBEAT_VEL_THRESH  = -0.80   # what velocity counts as upbeat
NOD_REFRACT_MS     = 350
UPBEAT_REFRACT_MS  = 900
EMA_ALPHA          = 0.35
EVENT_DISPLAY_MS   = 700

NOSE_TIP_IDX = 1

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Could not open webcam")

# storing previous state data
prev_t = None
prev_y_ema = None
last_nod_ms = 0
last_upbeat_ms = 0
last_event = None
last_event_time_ms = 0

def ema(prev, x, a):
    return a * x + (1 - a) * prev if prev is not None else x

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    img_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    res = face_mesh.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    now = time.time()
    now_ms = int(now * 1000)
    vel = 0.0

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        nose_lm = lms[NOSE_TIP_IDX]
        nose_y = float(nose_lm.y)  # normalized [0..1]; downwards increases

        # smoothing + velocity
        y_ema = ema(prev_y_ema, nose_y, EMA_ALPHA)
        if prev_t is None:
            prev_t = now
            prev_y_ema = y_ema
        dt = max(1e-3, now - prev_t)
        vel = (y_ema - prev_y_ema) / dt  # if positive vel, downwards motion. if negative vel, upwards motion.

        # events
        if vel <= UPBEAT_VEL_THRESH and (now_ms - last_upbeat_ms) >= UPBEAT_REFRACT_MS:
            print(f"[UPBEAT] t={now:.3f}  vel={vel:+.3f}")
            last_event = "UPBEAT"
            last_event_time_ms = now_ms
            last_upbeat_ms = now_ms
        elif vel >= NOD_VEL_THRESH and (now_ms - last_nod_ms) >= NOD_REFRACT_MS:
            print(f"[NOD]    t={now:.3f}  vel={vel:+.3f}")
            last_event = "NOD"
            last_event_time_ms = now_ms
            last_nod_ms = now_ms

        prev_t = now
        prev_y_ema = y_ema

        # show nose point on screen
        nose_px = (int(nose_lm.x * w), int(nose_lm.y * h))
        cv2.circle(img, nose_px, 5, (0, 255, 0), -1)

    # UI overlays
    cv2.putText(img, f"nose_y_vel: {vel:+.3f} norm/s", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    if last_event and (now_ms - last_event_time_ms) <= EVENT_DISPLAY_MS:
        color = (0, 255, 255) if last_event == "UPBEAT" else (0, 255, 0)
        cv2.putText(img, last_event, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    cv2.imshow("Nose Tip Events (NOD / UPBEAT)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
