import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import math

# === Parameters (match your main script) ===
WINDOW_SIZE = 15                  # number of recent frames to analyze
NOD_THRESHOLD = 1.5               # vertical movement amplitude threshold
SHAKE_THRESHOLD = 1.5             # horizontal movement amplitude threshold
MOVEMENT_RATIO_THRESHOLD = 0.4    # ratio of frames that must show motion
DELTA_THRESHOLD = 0.5             # per-frame delta threshold
ZC_THRESHOLD = 1                  # zero-crossing threshold

# Buffers for smoothing
pitches = deque(maxlen=WINDOW_SIZE)
yaws = deque(maxlen=WINDOW_SIZE)

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_head_angles(face_landmarks):
    # Use normalized landmark coordinates (0–1 range)
    lm = face_landmarks.landmark

    # Key points
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

def detect_motion_state():
    """Return 'Idle', 'Nodding', or 'Shaking' based on recent nose motion."""
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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press ESC or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run face mesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        state = "Idle"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            pitch, yaw = get_head_angles(face_landmarks)

            pitches.append(pitch)
            yaws.append(yaw)

            state = detect_motion_state()

        

        # Draw state
        cv2.putText(
            frame,
            f"State: {state}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0) if state == "Nodding" else
            (0, 0, 255) if state == "Shaking" else
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow("Test Motion State", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
