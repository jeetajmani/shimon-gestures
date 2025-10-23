import cv2
import time
import threading
import math
import numpy as np
import mediapipe as mp
from pythonosc import udp_client

# === Setup OSC clients ===
note_client = udp_client.SimpleUDPClient("192.168.1.1", 9010)
head_client = udp_client.SimpleUDPClient("192.168.1.1", 9000)

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# === Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# === Globals ===
note_thread = None
phrase_playing = False
last_gesture = None  # "thumbs_up", "thumbs_down", or None
lock = threading.Lock()
gesture_timestamp = 0
velocity = 60       # base note velocity
last_volume_gesture = None
phrase = [
    (-0.5, 60), (-0.5, 62), (-0.5, 64),
    (0.5, 65), (0.5, 67), (-0.5, 69),
    (-0.5, 71), (-0.5, 72)
]


# === Gesture detection helpers ===
WRIST = 0
TH_MCP, TH_IP, TH_TIP = 2, 3, 4
IX_MCP, IX_PIP, IX_DIP, IX_TIP = 5, 6, 7, 8
MI_MCP, MI_PIP, MI_DIP, MI_TIP = 9, 10, 11, 12
RI_MCP, RI_PIP, RI_DIP, RI_TIP = 13, 14, 15, 16
PI_MCP, PI_PIP, PI_DIP, PI_TIP = 17, 18, 19, 20

def _pip_angle(pts, mcp, pip, dip):
    v1 = pts[mcp] - pts[pip]
    v2 = pts[dip] - pts[pip]
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)
    if a == 0 or b == 0: return 180
    cosang = np.clip(np.dot(v1, v2) / (a*b), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _curl_score(pts, mcp, pip, dip):
    ang = _pip_angle(pts, mcp, pip, dip)
    return float(np.clip((180 - ang) / 120, 0, 1))

def _others_mostly_folded(pts, tol=0.45):
    curls = [_curl_score(pts, IX_MCP, IX_PIP, IX_DIP),
             _curl_score(pts, MI_MCP, MI_PIP, MI_DIP),
             _curl_score(pts, RI_MCP, RI_PIP, RI_DIP),
             _curl_score(pts, PI_MCP, PI_PIP, PI_DIP)]
    return sum(c >= tol for c in curls) >= 3

def _thumb_extended_and_dir(pts):
    v1 = pts[TH_MCP] - pts[TH_IP]
    v2 = pts[TH_TIP] - pts[TH_IP]
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)
    ang = 180 if a == 0 or b == 0 else math.degrees(math.acos(np.clip(np.dot(v1,v2)/(a*b), -1,1)))
    dir_vec = pts[TH_TIP] - pts[TH_MCP]
    n = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / n if n > 0 else np.array([0.0, 0.0])
    return (ang > 150, -dir_vec[1])

def _finger_extended(pts, mcp, pip, dip, tip, thres_deg=160):
    return _pip_angle(pts, mcp, pip, dip) > thres_deg

def is_open_palm(pts):
    idx = _finger_extended(pts, IX_MCP, IX_PIP, IX_DIP, IX_TIP)
    mid = _finger_extended(pts, MI_MCP, MI_PIP, MI_DIP, MI_TIP)
    rin = _finger_extended(pts, RI_MCP, RI_PIP, RI_DIP, RI_TIP)
    pin = _finger_extended(pts, PI_MCP, PI_PIP, PI_DIP, PI_TIP)
    return idx and mid and rin and pin

def is_thumbs_up(pts):
    if not _others_mostly_folded(pts): return False
    th_ext, upness = _thumb_extended_and_dir(pts)
    return th_ext and upness > 0.35

def is_thumbs_down(pts):
    if not _others_mostly_folded(pts): return False
    th_ext, upness = _thumb_extended_and_dir(pts)
    return th_ext and upness < -0.35

# === Volume gesture detection (hand height → velocity) ===
VEL_MIN, VEL_MAX = 30, 120
DYNAMICS_SMOOTH = 0.2
dyn_state = velocity  # initial smoothed velocity

def detect_volume_gesture(pts):
    global dyn_state, velocity

    # wrist y is normalized [0,1], smaller = higher
    wrist_y_norm = pts[WRIST][1]

    # invert so top = 1, bottom = 0
    gain = 1.0 - wrist_y_norm
    target_vel = int(round(VEL_MIN + gain * (VEL_MAX - VEL_MIN)))

    # smooth velocity
    dyn_state = int(round(dyn_state * (1 - DYNAMICS_SMOOTH) + target_vel * DYNAMICS_SMOOTH))
    velocity = dyn_state


# === Robot phrase ===
# def play_phrase():
#     global phrase_playing, last_gesture, gesture_timestamp

#     with lock:
#         phrase_playing = True
#         last_gesture = None
#         gesture_timestamp = time.time()

#     print("🎵 Shimon: Starting phrase")

#     for head_angle, note in phrase:
#         with lock:
#             v = velocity
#         send_head_message_to_shimon(head_angle)
#         send_note_to_shimon(note, v)
#         time.sleep(1)

#     print("🎵 Shimon: Finished phrase")
#     with lock:
#         phrase_playing = False

#     # Decide what to do next
#     if last_gesture == "thumbs_down":
#         print("👎 Replaying phrase")
#         start_phrase()
#     elif last_gesture == "thumbs_up":
#         send_head_message_to_shimon(0.0)
#         print("👍 Waiting for eye contact again")
#     else:
#         send_head_message_to_shimon(0.0)
#         print("No gesture detected, waiting for eye contact again")

# def start_phrase():
#     global note_thread
#     with lock:
#         if not phrase_playing:
#             note_thread = threading.Thread(target=play_phrase, daemon=True)
#             note_thread.start()

# === Face detection ===
def face_is_looking_at_camera(face_landmarks):
    nose_tip = face_landmarks.landmark[1]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]
    forehead = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]
    dist_left = abs(nose_tip.x - left_cheek.x)
    dist_right = abs(nose_tip.x - right_cheek.x)
    ratio = dist_left / dist_right if dist_right != 0 else 0
    yaw_ok = 0.7 < ratio < 1.3
    vertical_ratio = (nose_tip.y - forehead.y) / (chin.y - nose_tip.y)
    pitch_ok = 0.85 < vertical_ratio < 1.4
    return yaw_ok and pitch_ok

# === Main loop ===
def start_gestures_monitor(on_eye_contact_callback, on_hand_callback):
    """
    Starts camera loop in a background thread.
    Calls `on_eye_contact_callback()` whenever eye contact is detected.
    """
    def camera_loop():
        prev_face_is_looking = False
        prev_thumbs_up = False
        prev_thumbs_down = False
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # PREPROCESS FRAME
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # FACE DETECTION
            face_is_looking = False
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_is_looking = face_is_looking_at_camera(face_landmarks)

            # Detect eye contact state change
            if face_is_looking and not prev_face_is_looking:
                print("👀 Eye contact detected")
                on_eye_contact_callback()

            prev_face_is_looking = face_is_looking

            # cv2.imshow('Eye Contact Detector', image)
            thumbs_up = False
            thumbs_down = False
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    detect_volume_gesture(pts)

                    if is_thumbs_up(pts):
                        thumbs_up = True
                    elif is_thumbs_down(pts):
                        thumbs_down = True

                    # Call the gesture callback only when gesture changes
                    if thumbs_up and not prev_thumbs_up:
                        print("🖐 Gesture detected: thumbs_up")
                        on_hand_callback(1)
                    
                    prev_thumbs_up = thumbs_up
                    
                    if thumbs_down and not prev_thumbs_down:
                        print("🖐 Gesture detected: thumbs_down")
                        on_hand_callback(0)
                    
                    prev_thumbs_down = thumbs_down
                    
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

    # Run in a background thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    
# prev_face_is_looking = False

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     face_results = face_mesh.process(image)
#     hand_results = hands.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     face_is_looking = False
#     if face_results.multi_face_landmarks:
#         for face_landmarks in face_results.multi_face_landmarks:
#             face_is_looking = face_is_looking_at_camera(face_landmarks)
#             mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=drawing_spec
#             )

#     # --- Gesture detection ---
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
#             )

#             pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
#             detect_volume_gesture(pts)
#             if is_thumbs_up(pts):
#                 with lock:
#                     last_gesture = "thumbs_up"
#             elif is_thumbs_down(pts):
#                 with lock:
#                     last_gesture = "thumbs_down"

#     # --- State transitions ---
#     if face_is_looking and not prev_face_is_looking:
#         print("👀 Eye contact detected")
#         # start_phrase()

#     prev_face_is_looking = face_is_looking

#     # --- UI Overlay ---
#     text = "Looking at camera" if face_is_looking else "Waiting..."
#     color = (0, 255, 0) if face_is_looking else (0, 0, 255)
#     cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#     if last_gesture:
#         cv2.putText(image, f"Gesture: {last_gesture}", (50, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(image, f"Velocity: {velocity}", (50, 150),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow('Shimon Interaction', image)

#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()