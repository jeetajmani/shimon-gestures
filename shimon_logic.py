# combined_face_music_control_downbeat_toggle.py
# Requires: pip install opencv-python mediapipe numpy

import time
import cv2
import numpy as np
import mediapipe as mp

# ========= Tunables =========
# Match nose_tip_events.py for gesture thresholds & timing
NOD_VEL_THRESH      = +0.30   # downward (positive y-velocity) counts as a downbeat (nod)
UPBEAT_VEL_THRESH   = -0.80   # upward (negative y-velocity) counts as an upbeat
NOD_REFRACT_MS      = 350
UPBEAT_REFRACT_MS   = 900
EMA_ALPHA           = 0.35
EVENT_DISPLAY_MS    = 700

# Face "looking at camera" thresholds (same logic as your camera-facing script)
YAW_RATIO_MIN       = 0.7     # dist_left/dist_right within [min, max]
YAW_RATIO_MAX       = 1.3
PITCH_RATIO_MIN     = 0.85    # (nose-forehead)/(chin-nose) within [min, max]
PITCH_RATIO_MAX     = 1.4

# Landmark indices
NOSE_TIP_IDX   = 1
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX= 454
FOREHEAD_IDX   = 10
CHIN_IDX       = 152

# ========= Setup Face Mesh =========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,          # same as your "looking" script; indices still valid either way
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Could not open webcam")

# ========= State for motion detection =========
prev_t = None
prev_y_ema = None
last_nod_ms = 0
last_upbeat_ms = 0
last_event = None
last_event_time_ms = 0

def ema(prev, x, a):
    return a * x + (1 - a) * prev if prev is not None else x

# ========= State for looking detection & playback FSM =========
face_is_looking = False

# Playback FSM:
#   - playing: current transport state
#   - armed_toggle: set True by an UPBEAT while looking; on the NEXT DOWNBEAT while looking, toggle playing and print.
playing = False
armed_toggle = False

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

    # ---------- Determine "looking at camera" (same yaw/pitch logic) ----------
    face_is_looking = False
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        nose = lms[NOSE_TIP_IDX]
        left_cheek = lms[LEFT_CHEEK_IDX]
        right_cheek = lms[RIGHT_CHEEK_IDX]
        forehead = lms[FOREHEAD_IDX]
        chin = lms[CHIN_IDX]

        # Yaw (left/right)
        dist_left = abs(nose.x - left_cheek.x)
        dist_right = abs(nose.x - right_cheek.x)
        ratio = dist_left / dist_right if dist_right != 0 else 0.0
        yaw_ok = (YAW_RATIO_MIN < ratio < YAW_RATIO_MAX)

        # Pitch (up/down)
        denom = (chin.y - nose.y)
        vertical_ratio = (nose.y - forehead.y) / denom if denom != 0 else 0.0
        pitch_ok = (PITCH_RATIO_MIN < vertical_ratio < PITCH_RATIO_MAX)

        if yaw_ok and pitch_ok:
            face_is_looking = True

        # ---------- Motion: EXACT upbeat logic from nose_tip_events.py ----------
        nose_y = float(nose.y)                 # normalized [0..1]; downwards increases
        y_ema = ema(prev_y_ema, nose_y, EMA_ALPHA)
        if prev_t is None:
            prev_t = now
            prev_y_ema = y_ema
        dt = max(1e-3, now - prev_t)           # same dt floor
        vel = (y_ema - prev_y_ema) / dt        # +vel = downward; -vel = upward

        if face_is_looking:
            # --- UPBEAT event (keep exact order & refractory style) ---
            if vel <= UPBEAT_VEL_THRESH and (now_ms - last_upbeat_ms) >= UPBEAT_REFRACT_MS:
                # (no print here; we only act on the following downbeat)
                armed_toggle = True
                last_event = "UPBEAT"
                last_event_time_ms = now_ms
                last_upbeat_ms = now_ms

            # --- NOD / DOWNBEAT event (same nod logic) ---
            elif vel >= NOD_VEL_THRESH and (now_ms - last_nod_ms) >= NOD_REFRACT_MS:
                last_event = "DOWNBEAT"
                last_event_time_ms = now_ms
                last_nod_ms = now_ms

                # Toggle only if we were armed by an earlier upbeat (while looking)
                if armed_toggle:
                    playing = not playing
                    armed_toggle = False
                    if playing:
                        print("Shimon start playing: downbeat detected")
                    else:
                        print("Shimon stop playing: downbeat detected")

        prev_t = now
        prev_y_ema = y_ema

        # show a few landmarks (visual only)
        for lm in (nose, left_cheek, right_cheek, forehead, chin):
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

    # ---------- UI overlays ----------
    cv2.putText(img, f"nose_y_vel: {vel:+.3f} norm/s", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    if last_event and (now_ms - last_event_time_ms) <= EVENT_DISPLAY_MS:
        color = (0, 255, 255) if last_event == "UPBEAT" else (0, 255, 0)
        cv2.putText(img, last_event, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Optional: show current playing state
    play_text = f"{'SHIMON PLAYING' if playing else 'SHIMON NOT PLAYING'}"
    play_color = (0, 200, 0) if playing else (0, 0, 200)
    cv2.putText(img, play_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, play_color, 2, cv2.LINE_AA)

    cv2.imshow("Face-Controlled Upbeat/Downbeat Transport", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
