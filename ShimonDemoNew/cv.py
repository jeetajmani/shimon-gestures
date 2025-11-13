import cv2
import time
import threading
import math
import numpy as np
import mediapipe as mp
from pythonosc import udp_client
from collections import deque

tempo_detection_enabled = threading.Event()

SHOW_WINDOW = False

# # === Setup OSC clients ===
# note_client = udp_client.SimpleUDPClient("192.168.1.1", 9010)
# head_client = udp_client.SimpleUDPClient("192.168.1.1", 9000)

# Parameters
WINDOW_SIZE = 15                  # number of recent frames to analyze
NOD_THRESHOLD = 1.5               # vertical movement amplitude threshold
SHAKE_THRESHOLD = 1.5             # horizontal movement amplitude threshold
MOVEMENT_RATIO_THRESHOLD = 0.4    # ratio of frames that must show motion
DELTA_THRESHOLD = 0.5             # per-frame delta threshold
ZC_THRESHOLD = 1                  # zero-crossing threshold

# Buffers for smoothing
pitches = deque(maxlen=WINDOW_SIZE)
yaws = deque(maxlen=WINDOW_SIZE)

# pitch_vals = deque(maxlen=WINDOW_SIZE)
# yaw_vals = deque(maxlen=WINDOW_SIZE)

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
# mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
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

def get_head_angles(face_landmarks):
    # Use normalized landmark coordinates (0–1 range)
    lm = face_landmarks.landmark

    # Key points
    # nose = np.array([lm[1].x, lm[1].y, lm[1].z])
    left_eye = np.array([lm[33].x, lm[33].y, lm[33].z])
    right_eye = np.array([lm[263].x, lm[263].y, lm[263].z])
    chin = np.array([lm[152].x, lm[152].y, lm[152].z])
    forehead = np.array([lm[10].x, lm[10].y, lm[10].z])

    # Horizontal axis (eye to eye)
    eye_vec = right_eye - left_eye
    eye_vec /= np.linalg.norm(eye_vec)

    # Vertical axis (nose to chin)
    vert_vec = chin - forehead
    vert_vec /= np.linalg.norm(vert_vec)

    # Compute angles (in degrees)
    yaw = math.degrees(math.atan2(eye_vec[2], eye_vec[0]))   # left-right
    pitch = math.degrees(math.atan2(vert_vec[2], vert_vec[1]))  # up-down

    return pitch, yaw

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

def detect_motion_state():
    """Return 'Idle', 'Nodding', or 'Shaking' based on head angle motion."""
    # Only start once we have at least half a window
    if len(pitches) < WINDOW_SIZE // 2:
        return "Idle"

    x = np.array(yaws)
    y = np.array(pitches)
    
    def smooth(signal, window=5):
        if len(signal) < window:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')

    # Smooth signals a bit
    x = smooth(x)
    y = smooth(y)

    # Center signals around zero
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Per-frame motion
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))

    moving_x_ratio = np.mean(dx > DELTA_THRESHOLD)
    moving_y_ratio = np.mean(dy > DELTA_THRESHOLD)

    # Overall amplitude
    x_std = np.std(x)
    y_std = np.std(y)

    # Oscillations (zero crossings)
    def zero_crossings(signal):
        return np.count_nonzero(np.diff(np.sign(signal)))

    x_zc = zero_crossings(x)
    y_zc = zero_crossings(y)

    # Debug prints if you want:
    # print(f"x_std={x_std:.2f}, y_std={y_std:.2f}, "
    #       f"x_zc={x_zc}, y_zc={y_zc}, "
    #       f"mx={moving_x_ratio:.2f}, my={moving_y_ratio:.2f}")

    is_nodding = (
        y_std > NOD_THRESHOLD
        and y_zc >= ZC_THRESHOLD
        and moving_y_ratio > MOVEMENT_RATIO_THRESHOLD
        and y_std > x_std * 0.8   # mostly vertical
    )

    is_shaking = (
        x_std > SHAKE_THRESHOLD
        and x_zc >= ZC_THRESHOLD
        and moving_x_ratio > MOVEMENT_RATIO_THRESHOLD
        and x_std > y_std * 0.8   # mostly horizontal
    )

    if is_nodding and not is_shaking:
        return "Nodding"
    elif is_shaking and not is_nodding:
        return "Shaking"
    else:
        return "Idle"
    
def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    sign = np.sign(np.dot(cross, np.array([1, 0, 0])))
    return math.degrees(math.atan2(np.linalg.norm(cross), dot)) * sign

def estimate_tempo_from_nod(nod_times, window=5):
    if (len(nod_times) < 5):
        return None
    
    recent = nod_times[-window:]
    
    intervals = [t2 - t1 for t1, t2 in zip(recent[:-1], recent[1:])]
    
    avg_interval = np.median(intervals)
    
    tempo = 60 / avg_interval
    
    return np.floor(tempo)

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
def start_gestures_monitor(on_eye_contact_callback, on_approval_callback, on_tempo_callback):
    """
    Starts camera loop in a background thread.
    Calls `on_eye_contact_callback()` whenever eye contact is detected.
    """
    def camera_loop():
        prev_face_is_looking = False
        prev_thumbs_up = False
        prev_thumbs_down = False
        
        last_nod_time = 0
        last_shake_time = 0
        NOD_COOLDOWN = 1.0   
        SHAKE_COOLDOWN = 1.0
        
        prev_time = time.time()
        frame_count = 0
        
        nose_y_list = []
        avg_nose_y = 0

        nodding = False
        nod_count = 0
        nod_times = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"Current loop FPS: {fps:.2f}")
                frame_count = 0
                prev_time = now

            # Flip and convert only once (flip horizontally so head motion matches screen)
            image = cv2.flip(image, 1)

            # Convert to RGB only if needed by MediaPipe (it requires RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run detection
            rgb_image.flags.writeable = False
            face_results = face_mesh.process(rgb_image)
            # hand_results = hands.process(rgb_image)
            rgb_image.flags.writeable = True

            # Only draw if window is enabled
            if SHOW_WINDOW and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        drawing_spec,
                        drawing_spec
                    )

            # FACE DETECTION
            face_is_looking = False
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_is_looking = face_is_looking_at_camera(face_landmarks)

            # Detect eye contact state change
            if not tempo_detection_enabled.is_set() and face_is_looking and not prev_face_is_looking:
                print("👀 Eye contact detected")
                on_eye_contact_callback()

            prev_face_is_looking = face_is_looking

            # APPROVAL DETECTION - NOD/SHAKE
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                pitch, yaw = get_head_angles(face_landmarks)

                pitches.append(pitch)
                yaws.append(yaw)
                
                state = detect_motion_state()
                
                cv2.putText(
                    image,
                    f"State: {state}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if state == "Nodding" else (0, 0, 255) if state == "Shaking" else (255, 255, 255),
                    2,
                )
                    
                now = time.time()

                if not tempo_detection_enabled.is_set():
                    if state == "Nodding" and (now - last_nod_time > NOD_COOLDOWN):
                        print("-- Gesture detected: Nod")
                        on_approval_callback(1)
                        last_nod_time = now

                    if state == "Shaking" and (now - last_shake_time > SHAKE_COOLDOWN):
                        print("-- Gesture detected: Shake")
                        on_approval_callback(0)
                        last_shake_time = now
            
            # TEMPO DETECTION
            if tempo_detection_enabled.is_set() and face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                h, w = image.shape[:2]
                
                # Get nose tip (landmark #1)
                # nose_tip = landmarks.landmark[1]
                # nose_x = int(nose_tip.x * w)
                # nose_y = int(nose_tip.y * h)
                
                chin = landmarks.landmark[152]
                forehead = landmarks.landmark[10]
                
                chin_x, chin_y = int(chin.x * w), int(chin.y * h)
                forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)

                chin_point = np.array([chin.x * w, chin.y * h, chin.z * w])
                forehead_point = np.array([forehead.x * w, forehead.y * h, forehead.z * w])

                v = forehead_point - chin_point
                
                vertical = np.array([0, -1, 0])
                pitch_angle = angle_between(v, vertical)
                
                # cv2.putText(image, f"Pitch: {pitch_angle:.2f}", (30, 30),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if pitch_angle < -6: 
                    # cv2.putText(frame, "NOD", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    if (not nodding):
                        nodding = True
                        print("NOD")
                        nod_times.append(time.time())
                        tempo = estimate_tempo_from_nod(nod_times)
                        if tempo:
                            print("Tempo: ", tempo)
                            on_tempo_callback(tempo)
                            # send_to_max("/tempo", tempo, max_port=7401)
                else: 
                    if (nodding):
                        nodding = False
                        nod_count += 1
                        print("Nod Done -- total: ", nod_count)
                # try:
                #     if tempo:
                #         cv2.putText(image, f"Tempo: {tempo:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # except:
                #     print("NO TEMPO")

            # # THUMB DETECTION
            # thumbs_up = False
            # thumbs_down = False
            # if hand_results.multi_hand_landmarks:
            #     for hand_landmarks in hand_results.multi_hand_landmarks:
            #         pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            #         detect_volume_gesture(pts)

            #         if is_thumbs_up(pts):
            #             thumbs_up = True
            #         elif is_thumbs_down(pts):
            #             thumbs_down = True

            #         # Call the gesture callback only when gesture changes
            #         if thumbs_up and not prev_thumbs_up:
            #             print("🖐 Gesture detected: thumbs_up")
            #             on_hand_callback(1)
                    
            #         prev_thumbs_up = thumbs_up
                    
            #         if thumbs_down and not prev_thumbs_down:
            #             print("🖐 Gesture detected: thumbs_down")
            #             on_hand_callback(0)
                    
            #         prev_thumbs_down = thumbs_down
                    
            # cv2.imshow("Gesture Monitor", image)
            # cv2.waitKey(1)
            if SHOW_WINDOW:
                cv2.imshow("Gesture Monitor", image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                # still call waitKey to let OpenCV tick internally (no-op)
                cv2.waitKey(1)


        cap.release()
        cv2.destroyAllWindows()

    # Run in a background thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    # SHOW_WINDOW = True
    # camera_loop()
    
