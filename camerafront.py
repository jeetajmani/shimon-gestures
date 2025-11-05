import cv2
import mediapipe as mp

# Settings
MIRROR = False     # no selfie flip
ALPHA  = 0.2       # smoothing factor
RIGHT_ENTER = -0.35   # become "facing right"
RIGHT_EXIT  = -0.25   # return to "facing forward"

# MediaPipe landmark indices (Face Mesh)
IDX_NOSE_TIP = 1
IDX_EYE_L_OUT = 33
IDX_EYE_R_OUT = 263

mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

def ema(prev, cur, a):
    return cur if prev is None else (a*prev + (1-a)*cur)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("No webcam found")

state = "facing forward"
offset_s = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if MIRROR:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        x_nose = lm[IDX_NOSE_TIP].x
        x_le   = lm[IDX_EYE_L_OUT].x
        x_re   = lm[IDX_EYE_R_OUT].x

        x_mid = 0.5*(x_le + x_re)
        eye_span = max(abs(x_re - x_le), 1e-6)

        offset = (x_nose - x_mid) / eye_span
        offset_s = ema(offset_s, offset, ALPHA)

        prev_state = state

        if state == "facing forward" and offset_s <= RIGHT_ENTER:
            state = "facing right"
        elif state == "facing right" and offset_s >= RIGHT_EXIT:
            state = "facing forward"

        if state != prev_state:
            print(f"STATE CHANGE → {state}")

        # UI
        cv2.rectangle(frame, (10,10), (300,70), (0,0,0), -1)
        cv2.putText(frame, state, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)

    else:
        # Lost face: reset to forward
        if state != "facing forward":
            state = "facing forward"
            print("STATE CHANGE → facing forward")

        cv2.rectangle(frame, (10,10), (240,70), (0,0,0), -1)
        cv2.putText(frame, "no face", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
        offset_s = None

    cv2.imshow("Forward vs Right", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
