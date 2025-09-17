import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Run the face tracking loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and get face landmarks
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

            # YAW CHECK (left/right rotation)
            dist_left = abs(nose_tip.x - left_cheek.x)
            dist_right = abs(nose_tip.x - right_cheek.x)
            ratio = dist_left / dist_right if dist_right != 0 else 0

            yaw_ok = 0.7 < ratio < 1.3   # tweak threshold if needed

            # PITCH CHECK (up/down tilt)
            vertical_ratio = (nose_tip.y - forehead.y) / (chin.y - nose_tip.y)

            pitch_ok = 0.85 < vertical_ratio < 1.4   # tweak threshold if needed

            if yaw_ok and pitch_ok:
                face_is_looking = True

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

    # Check if face is looking
    if face_is_looking:
        cv2.putText(image, "Looking at camera", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "Looking away", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed image
    cv2.imshow('MediaPipe Face Mesh', image)

    # Press 'q' to quit the application
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
