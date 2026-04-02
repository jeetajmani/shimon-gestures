import cv2
import time
import threading
import math
import numpy as np
import mediapipe as mp
from collections import deque

tempo_detection_enabled = threading.Event()

SHOW_WINDOW = False
DEBUG_FPS = False

# General parameters
WINDOW_SIZE = 15
DELTA_THRESHOLD = 0.5

# Approval nod thresholds (stronger movement)
APPROVAL_DOWN_THR = -12
APPROVAL_UP_THR   = -4

# Disapproval shake thresholds
SHAKE_LEFT_THR  = -10     # yaw angle left
SHAKE_RIGHT_THR = +10     # yaw angle right
SHAKE_DEBOUNCE  = 1.0     # seconds required between shakes

# Tempo nod thresholds (lighter/faster)
TEMPO_DOWN_THR = -5
TEMPO_UP_THR   = -0.5
TEMPO_MIN_NODS = 3
TEMPO_WINDOW_SEC = 3.45

# Buffers
pitches = deque(maxlen=WINDOW_SIZE)
yaws = deque(maxlen=WINDOW_SIZE)

SMOOTH_WINDOW = 5
SMOOTH_KERNEL = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW

# === MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# ========= UTILITIES ========= #

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    return math.degrees(math.atan2(np.linalg.norm(cross), dot))


def get_head_angles(face_landmarks):
    lm = face_landmarks.landmark
    left_eye = np.array([lm[33].x,  lm[33].y,  lm[33].z])
    right_eye = np.array([lm[263].x, lm[263].y, lm[263].z])
    chin = np.array([lm[152].x, lm[152].y, lm[152].z])
    fore = np.array([lm[10].x,  lm[10].y,  lm[10].z])

    eye_vec = right_eye - left_eye
    eye_vec /= np.linalg.norm(eye_vec)

    vert_vec = chin - fore
    vert_vec /= np.linalg.norm(vert_vec)

    yaw = math.degrees(math.atan2(eye_vec[2], eye_vec[0]))
    pitch = math.degrees(math.atan2(vert_vec[2], vert_vec[1]))

    return pitch, yaw


def estimate_tempo(nod_times):
    now = time.time()
    recent = [t for t in nod_times if now - t < TEMPO_WINDOW_SEC]
    if len(recent) < TEMPO_MIN_NODS:
        return None

    intervals = [t2 - t1 for t1, t2 in zip(recent[:-1], recent[1:])]
    if len(intervals) == 0:
        return None

    return int(60 / np.median(intervals))


def face_is_looking_at_camera(face_landmarks):
    nose = face_landmarks.landmark[1]
    lc = face_landmarks.landmark[234]
    rc = face_landmarks.landmark[454]
    fore = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]

    dl = abs(nose.x - lc.x)
    dr = abs(nose.x - rc.x)
    if dr == 0:
        return False
    yaw_ok = 0.7 < dl/dr < 1.3

    denom = (chin.y - nose.y)
    if denom == 0:
        return False
    pitch_ok = 0.85 < (nose.y - fore.y) / denom < 1.4

    return yaw_ok and pitch_ok


# ========= MAIN LOOP ========== #

def start_gestures_monitor(on_eye_contact_callback,
                           on_approval_callback,
                           on_tempo_callback):

    def camera_loop():
        prev_face_looking = False

        # Approval nod state
        nodding = False

        # Shake state
        last_shake_time = 0
        shake_stage = None  # "L" → "R" → "L" OR "R" → "L" → "R"

        # Tempo nod state
        tempo_nodding = False
        nod_times = []

        frame_count = 0
        last_fps_time = time.time()

        prev_tempo_mode = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = face_mesh.process(rgb)

            tempo_mode = tempo_detection_enabled.is_set()

            if tempo_mode and not prev_tempo_mode:
                tempo_nodding = False
                nod_times.clear()
                print(">>> ENTER TEMPO MODE (reset nod buffers)")
            prev_tempo_mode = tempo_mode

            # FPS counter
            if DEBUG_FPS:
                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1:
                    print(f"FPS: {frame_count}")
                    frame_count = 0
                    last_fps_time = now

            if not result.multi_face_landmarks:
                cv2.waitKey(1)
                continue

            face = result.multi_face_landmarks[0]

            # ====== EYE CONTACT ======
            looking = face_is_looking_at_camera(face)
            if looking and not prev_face_looking:
                print("👀 Eye contact detected")
                on_eye_contact_callback()
            prev_face_looking = looking

            # ====== HEAD ANGLES ======
            pitch, yaw = get_head_angles(face)

            # ====== APPROVAL NOD DETECTION ======
            if not tempo_mode:
                if not nodding and pitch < APPROVAL_DOWN_THR:
                    nodding = True

                elif nodding and pitch > APPROVAL_UP_THR:
                    nodding = False
                    print("🟢 APPROVAL — Completed nod gesture")
                    on_approval_callback(1)

            # ====== DISAPPROVAL SHAKE DETECTION ======
            now = time.time()

            if now - last_shake_time > SHAKE_DEBOUNCE:

                # 3-phase shake detection
                if shake_stage is None:
                    if yaw < SHAKE_LEFT_THR:
                        shake_stage = "L"
                    elif yaw > SHAKE_RIGHT_THR:
                        shake_stage = "R"

                elif shake_stage == "L":
                    if yaw > SHAKE_RIGHT_THR:
                        shake_stage = "LR"

                elif shake_stage == "R":
                    if yaw < SHAKE_LEFT_THR:
                        shake_stage = "RL"

                elif shake_stage == "LR":
                    if yaw < SHAKE_LEFT_THR:
                        print("❌ DISAPPROVAL — Completed shake gesture")
                        on_approval_callback(0)
                        shake_stage = None
                        last_shake_time = now

                elif shake_stage == "RL":
                    if yaw > SHAKE_RIGHT_THR:
                        print("❌ DISAPPROVAL — Completed shake gesture")
                        on_approval_callback(0)
                        shake_stage = None
                        last_shake_time = now

            else:
                shake_stage = None

            # ====== TEMPO NOD DETECTION ======
            if tempo_mode:

                if not tempo_nodding and pitch < TEMPO_DOWN_THR:
                    tempo_nodding = True

                elif tempo_nodding and pitch > TEMPO_UP_THR:
                    tempo_nodding = False
                    print("💛 TEMPO — Completed nod gesture")
                    nod_times.append(time.time())

                    tempo = estimate_tempo(nod_times)
                    if tempo:
                        print(f"💛 TEMPO — Final Tempo = {tempo} BPM")
                        on_tempo_callback(tempo)

            # ====== SHOW WINDOW ======
            if SHOW_WINDOW:
                cv2.imshow("Gesture Monitor", frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=camera_loop, daemon=True).start()
