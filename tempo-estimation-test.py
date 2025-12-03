import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import time
import math

def send_to_max(message, value, max_host="127.0.0.1", max_port=7400):
    client = udp_client.SimpleUDPClient(max_host, max_port)

    # client.send_message(f"/nod", 1)
    client.send_message(message, value)
    
def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    sign = np.sign(np.dot(cross, np.array([1, 0, 0])))
    return math.degrees(math.atan2(np.linalg.norm(cross), dot)) * sign


def estimate_tempo_from_nod(nod_times, window=5):
    if (len(nod_times) < 4):
        return None
    
    recent = nod_times[-window:]
    
    intervals = [t2 - t1 for t1, t2 in zip(recent[:-1], recent[1:])]
    
    avg_interval = np.median(intervals)
    
    tempo = 60 / avg_interval
    
    return tempo


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

nodding = False
nod_count = 0
nod_times = []

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
        
        cv2.putText(frame, f"Pitch: {pitch_angle:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if pitch_angle < -6: 
            cv2.putText(frame, "NOD", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            if (not nodding):
                nodding = True
                # print("NOD")
                nod_times.append(time.time())
                tempo = estimate_tempo_from_nod(nod_times)
                if tempo:
                    print("Tempo: ", tempo)
                    send_to_max("/tempo", tempo, max_port=7401)
        else: 
            if (nodding):
                nodding = False
                nod_count += 1
                # print("Nod Done -- total: ", nod_count)
        try:
            if tempo:
                cv2.putText(frame, f"Tempo: {tempo:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        except:
            print("NO TEMPO")
        
        
        
        # nose_y_list.append(nose_y)
        
        # if (len(nose_y_list) > 50):
        #     nose_y_list.pop(0)
        
        # avg_nose_y = np.mean(nose_y_list)
        # print(avg_nose_y)
        
        # if len(nose_y_list) > 1 and (nose_y_list[-1] - avg_nose_y) >= 20:
        #     if (not nodding):
        #         current_time = time.time()
        #         print("NOD")
        #         # send_to_max()
        #         last_nod_time = current_time
        #         nodding = True
        #     else: 
        #         print("still nodding lol")
        # else: 
        #     if (nodding):
        #         nodding = False
        #         nod_count += 1
        #         print("Nod Done -- total: ", nod_count)
                
        # Draw nose position
        # cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)
        # cv2.line(frame, (chin_x, chin_y), (forehead_x, forehead_y), (0, 255, 0), 2)
        mid_x, mid_y = (chin_x + forehead_x) // 2, (chin_y + forehead_y) // 2
        cv2.arrowedLine(frame, (mid_x, mid_y), (mid_x, mid_y - int(pitch_angle * 2)), (255, 255, 0), 2)

        # # Show coordinates
        # cv2.putText(frame, f"Nose: ({nose_x}, {nose_y})", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Nod Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()