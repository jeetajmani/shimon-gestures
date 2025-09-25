# gaze_center_robust.py
# pip install opencv-python mediapipe numpy

import cv2
import numpy as np
import mediapipe as mp

# Tunables
H_RADIUS = 0.18
V_RADIUS = 0.18

# Hysteresis (frames). Need this many consecutive frames to switch states.
LOOK_GAIN_FRAMES = 5
LOOK_LOSS_FRAMES = 3

# Ignore frames where eye is nearly closed
MIN_EYE_OPEN = 0.18

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

L_OUT, L_INN, L_TOP, L_BOT = 33, 133, 159, 145
R_OUT, R_INN, R_TOP, R_BOT = 263, 362, 386, 374
L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]

def eye_center_ratio(lms, left_corner, right_corner, top_idx, bot_idx, iris_idxs):
    """Return (h_ratio, v_ratio, openness, cx, cy) within the eye box (0..1)."""
    lx, ly = lms[left_corner].x,  lms[left_corner].y
    rx, ry = lms[right_corner].x, lms[right_corner].y
    tx, ty = lms[top_idx].x,      lms[top_idx].y
    bx, by = lms[bot_idx].x,      lms[bot_idx].y

    iris = np.array([[lms[i].x, lms[i].y] for i in iris_idxs], dtype=np.float32)
    cx, cy = iris.mean(axis=0)

    eye_w = abs(rx - lx) + 1e-6
    eye_h = abs(by - ty) + 1e-6
    openness = eye_h / eye_w

    h_ratio = (cx - lx) / eye_w
    v_ratio = (cy - ty) / eye_h
    return float(h_ratio), float(v_ratio), float(openness), float(cx), float(cy)

def in_center_ellipse(h, v):
    dx = (h - 0.5) / H_RADIUS
    dy = (v - 0.5) / V_RADIUS
    return (dx*dx + dy*dy) <= 1.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera")

state_looking = False
gain_ctr = 0
loss_ctr = 0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    rgb.flags.writeable = True

    looking_now = False
    h_, w_ = frame.shape[:2]

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark

        lh, lv, lopen, lcx, lcy = eye_center_ratio(lms, L_OUT, L_INN, L_TOP, L_BOT, L_IRIS)
        rh, rv, ropen, rcx, rcy = eye_center_ratio(lms, R_INN, R_OUT, R_TOP, R_BOT, R_IRIS)

        # Ignore near-blinks to avoid flicker
        valid_left  = lopen >= MIN_EYE_OPEN
        valid_right = ropen >= MIN_EYE_OPEN

        left_ok  = valid_left  and in_center_ellipse(lh, lv)
        right_ok = valid_right and in_center_ellipse(rh, rv)
        looking_now = left_ok and right_ok

        cv2.circle(frame, (int(lcx*w_), int(lcy*h_)), 2, (0,255,0) if left_ok else (0,0,255), -1)
        cv2.circle(frame, (int(rcx*w_), int(rcy*h_)), 2, (0,255,0) if right_ok else (0,0,255), -1)

    if looking_now:
        gain_ctr += 1
        loss_ctr = 0
        if not state_looking and gain_ctr >= LOOK_GAIN_FRAMES:
            state_looking = True
    else:
        loss_ctr += 1
        gain_ctr = 0
        if state_looking and loss_ctr >= LOOK_LOSS_FRAMES:
            state_looking = False

    # Print per frame (debounced by state)
    print("LOOKING" if state_looking else "NOT_LOOKING")

    # On-screen label
    cv2.putText(frame, "LOOKING" if state_looking else "NOT LOOKING",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,255,0) if state_looking else (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Gaze Center (Robust, anti-flicker)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
