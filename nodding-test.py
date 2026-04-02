import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import time

def send_to_max(max_host="127.0.0.1", max_port=7400):
    client = udp_client.SimpleUDPClient(max_host, max_port)

    client.send_message(f"/nod", 1)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

nose_y_list = []
avg_nose_y = 0

last_nod_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get nose tip (landmark #1)
        nose_tip = landmarks.landmark[1]
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        
        nose_y_list.append(nose_y)
        
        if (len(nose_y_list) > 50):
            nose_y_list.pop(0)
        
        avg_nose_y = np.mean(nose_y_list)
        # print(avg_nose_y)
        
        if len(nose_y_list) > 1 and (nose_y_list[-1] - avg_nose_y) >= 20:
            current_time = time.time()
            if current_time - last_nod_time > 0.5:  # 0.5 second cooldown
                print("NOD")
                send_to_max()
                last_nod_time = current_time
                
        # Draw nose position
        cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)
        
        # Show coordinates
        cv2.putText(frame, f"Nose: ({nose_x}, {nose_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Nose Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()