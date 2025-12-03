import csv
import copy
import itertools
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("192.168.1.1", 9000)

def send_gesture_to_shimon(part, pos, vel):
    client.send_message("/head-commands", [part, float(pos), int(vel)])

def set_rest_pose():
    send_gesture_to_shimon("NECK", 0.0, 10)
    send_gesture_to_shimon("HEADTILT", 0.0, 10)
    send_gesture_to_shimon("BASEPAN", 0.0, 8)
    send_gesture_to_shimon("MOUTH", .03, 12)

def set_happy_pose():
    send_gesture_to_shimon("NECK", 0.7, 12)
    send_gesture_to_shimon("HEADTILT", -0.75, 12)
    send_gesture_to_shimon("MOUTH", .25, 12)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text, robot_state_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 44), (0, 0, 0), -1)
    if facial_text != "":
        info_text = 'Emotion: ' + facial_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    if robot_state_text:
        cv.putText(image, 'Shimon: ' + robot_state_text, (brect[0] + 5, brect[1] - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv.LINE_AA)
    return image

cap_device = 0
cap_width = 1920
cap_height = 1080
use_brect = True

cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

keypoint_classifier = KeyPointClassifier()
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

last_sent_state = None 
last_change_time = 0.0
MIN_INTERVAL = 0.25

set_rest_pose()
last_sent_state = "neutral"
last_change_time = time.time()

while True:
    key = cv.waitKey(10)
    if key == 27:
        break

    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    robot_state_text = ""
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            brect = calc_bounding_rect(debug_image, face_landmarks)
            landmark_list = calc_landmark_list(debug_image, face_landmarks)
            pre_processed = pre_process_landmark(landmark_list)

            facial_emotion_id = keypoint_classifier(pre_processed)
            label = keypoint_classifier_labels[facial_emotion_id].strip().lower()

            desired_state = None

            if "happy" in label:
                desired_state = "happy"
            elif "neutral" in label or "neutrality" in label or label == "neutral":
                desired_state = "neutral"

            now = time.time()
            if desired_state and desired_state != last_sent_state and (now - last_change_time) >= MIN_INTERVAL:
                if desired_state == "happy":
                    set_happy_pose()
                    robot_state_text = "head up (happy)"
                elif desired_state == "neutral":
                    set_rest_pose()
                    robot_state_text = "rest (neutral)"
                last_sent_state = desired_state
                last_change_time = now
            else:
                if last_sent_state == "happy":
                    robot_state_text = "head up (happy)"
                elif last_sent_state == "neutral":
                    robot_state_text = "rest (neutral)"

            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, label, robot_state_text)

    cv.imshow('Facial Emotion Recognition → Shimon Posture', debug_image)

cap.release()
cv.destroyAllWindows()