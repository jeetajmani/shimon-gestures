import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Parameters
WINDOW_SIZE = 30                 # number of recent frames to analyze
NOD_THRESHOLD = 2.5              # vertical movement amplitude threshold
SHAKE_THRESHOLD = 2.5            # horizontal movement amplitude threshold
MOVEMENT_RATIO_THRESHOLD = 0.6   # ratio of frames that must show motion

# Buffers for smoothing
x_positions = deque(maxlen=WINDOW_SIZE)
y_positions = deque(maxlen=WINDOW_SIZE)

def detect_motion_state():
    """Return 'Idle', 'Nodding', or 'Shaking' based on recent nose motion."""
    if len(x_positions) < WINDOW_SIZE:
        return "Idle"

    x = np.array(x_positions)
    y = np.array(y_positions)

    # Center signals around zero
    x -= np.mean(x)
    y -= np.mean(y)

    # Compute per-frame deltas (movement magnitude)
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))

    # Compute percentage of frames with noticeable motion
    moving_x_ratio = np.mean(dx > 1.0)
    moving_y_ratio = np.mean(dy > 1.0)

    # Compute movement amplitude (std deviation)
    x_std = np.std(x)
    y_std = np.std(y)

    # Count zero crossings (oscillations)
    def zero_crossings(signal):
        return np.count_nonzero(np.diff(np.sign(signal)))

    x_zc = zero_crossings(x)
    y_zc = zero_crossings(y)

    # Detect based on amplitude, consistency, and oscillation
    is_nodding = (
        y_std > NOD_THRESHOLD
        and y_zc > 4
        and moving_y_ratio > MOVEMENT_RATIO_THRESHOLD
    )
    is_shaking = (
        x_std > SHAKE_THRESHOLD
        and x_zc > 4
        and moving_x_ratio > MOVEMENT_RATIO_THRESHOLD
    )

    if is_nodding and not is_shaking:
        return "Nodding"
    elif is_shaking and not is_nodding:
        return "Shaking"
    else:
        return "Idle"

# Main loop
cap = cv2.VideoCapture(0)
print("Press ESC or 'q' to quit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Nose tip landmark index = 1
        nose_tip = face_landmarks.landmark[1]
        x = nose_tip.x * w
        y = nose_tip.y * h

        x_positions.append(x)
        y_positions.append(y)

        state = detect_motion_state()

        # Display state
        cv2.putText(
            frame,
            f"State: {state}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0) if state != "Idle" else (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    else:
        cv2.putText(frame, "No face detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Head Movement Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
