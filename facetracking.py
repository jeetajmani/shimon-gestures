import cv2
import time
import mediapipe as mp
import threading
from pythonosc import udp_client

# Setup UDP clients
note_client = udp_client.SimpleUDPClient("192.168.1.1", 9010)
head_client = udp_client.SimpleUDPClient("192.168.1.1", 9000)

def send_note_to_shimon(note, velocity):
    note_client.send_message("/arm", [note, velocity])

def send_head_message_to_shimon(angle):
    head_client.send_message("/head-commands", ["NECK", angle, 6])

# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Threading controls
note_thread = None
note_thread_running = False

def safe_sleep(seconds):
    """Sleep in small steps so we can check the stop flag often."""
    global note_thread_running
    start_time = time.time()
    while note_thread_running and (time.time() - start_time) < seconds:
        time.sleep(0.05)  # check roughly every 50ms

def note_loop():
    """Continuously send notes while running flag is True."""
    global note_thread_running
    pattern = [
        (-0.5, 60),
        (-0.4, 62),
        (-0.5, 64),
    ]
    while note_thread_running:
        for head_angle, note in pattern:
            if not note_thread_running:
                break  # stop immediately
            send_head_message_to_shimon(head_angle)
            send_note_to_shimon(note, 50)
            safe_sleep(1)  # more responsive sleep

def start_note_loop():
    """Start a background thread if not already running."""
    global note_thread, note_thread_running
    if not note_thread_running:
        note_thread_running = True
        note_thread = threading.Thread(target=note_loop, daemon=True)
        note_thread.start()
        print("Note loop started")

def stop_note_loop():
    """Stop the background note thread."""
    global note_thread_running
    if note_thread_running:
        note_thread_running = False
        print("Note loop stopped")

# Main loop
prev_face_is_looking = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_is_looking = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]

            # --- YAW check ---
            dist_left = abs(nose_tip.x - left_cheek.x)
            dist_right = abs(nose_tip.x - right_cheek.x)
            ratio = dist_left / dist_right if dist_right != 0 else 0
            yaw_ok = 0.7 < ratio < 1.3

            # --- PITCH check ---
            vertical_ratio = (nose_tip.y - forehead.y) / (chin.y - nose_tip.y)
            pitch_ok = 0.85 < vertical_ratio < 1.4

            if yaw_ok and pitch_ok:
                face_is_looking = True

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

    # Detect state change
    if prev_face_is_looking is not None and face_is_looking != prev_face_is_looking:
        if face_is_looking:
            print(">>> Changed: Now LOOKING at camera")
            start_note_loop()
        else:
            print(">>> Changed: Now LOOKING AWAY from camera")
            send_head_message_to_shimon(-0.1)
            stop_note_loop()

    prev_face_is_looking = face_is_looking

    # UI overlay
    text = "Looking at camera" if face_is_looking else "Looking away"
    color = (0, 255, 0) if face_is_looking else (0, 0, 255)
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

stop_note_loop()
cap.release()
cv2.destroyAllWindows()
