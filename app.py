#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import threading
import time
import math
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from pythonosc import udp_client

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# ===================== Shimon control =====================
HOST = "192.168.1.1"   # <-- set your robot IP
PORT = 9000
OSC_PATH = "/head-commands"
UP_ANGLE = 0.10
DOWN_ANGLE = -0.10
SPEED = 3

# Ramped cadence (seconds between bobs)
INTERVAL_MIN = 0.30   # fastest
INTERVAL_MAX = 1.20   # slowest
RAMP_FASTER_PER_S = 0.25  # shrink interval/sec when "Point" (speed up)
RAMP_SLOWER_PER_S = 0.12  # grow interval/sec otherwise (slow down)

# >>> Thumbs-up gate config
START_STABLE_FRAMES = 15  # ~0.5s at ~30fps; how long 👍 must be held to start

# ===================== Playback hook (EDIT ME) =====================
def on_start_playback():
    """
    🔧 Put your actual 'begin playing' code here.
    This will be called once when a stable 👍 is detected while armed.
    Examples:
        - Start your Audio->MIDI pipeline
        - Kick off PrettyMIDI / Mido playback
        - Send your custom OSC/UDP message to the music engine
    """
    print("[PLAYBACK] Starting music playback after thumbs-up!")
    # Example template:
    # from audioToMidi import AudioMidiConverter
    # AudioMidiConverter(...).start()

# ===================== Head bob control =====================
class HeadBobber:
    """Background head-bobbing loop you can pause/resume and retime."""
    def __init__(self, host=HOST, port=PORT, path=OSC_PATH,
                 up=UP_ANGLE, down=DOWN_ANGLE, speed=SPEED,
                 interval=1.0):
        self.client = udp_client.SimpleUDPClient(host, port)
        self.path = path
        self.up, self.down = float(up), float(down)
        self.speed = int(speed)
        self.interval = float(interval)
        self.running = False  # >>> start paused (wait for 👍)
        self._beat = 0
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            if self.running:
                angle = self.up if (self._beat % 2 == 0) else self.down
                try:
                    self.client.send_message(self.path, ["NECK", angle, self.speed])
                except Exception as e:
                    print("[OSC ERROR]", e)
                self._beat += 1
                time.sleep(self.interval)
            else:
                time.sleep(0.05)

    def pause(self):
        if self.running:
            print("[HeadBob] PAUSE")
        self.running = False

    def resume(self):
        if not self.running:
            print("[HeadBob] RESUME")
        self.running = True

    def set_interval(self, new_interval: float):
        new_interval = float(new_interval)
        new_interval = max(INTERVAL_MIN, min(INTERVAL_MAX, new_interval))
        if abs(new_interval - self.interval) > 1e-6:
            print(f"[HeadBob] interval -> {new_interval:.2f}s")
        self.interval = new_interval

    def nudge_interval(self, delta: float):
        self.set_interval(self.interval + float(delta))

    def shutdown(self):
        self.running = False
        self._stop.set()
        if self._thr.is_alive():
            self._thr.join(timeout=1.0)

# ===================== CLI args ===========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    parser.add_argument("--max_hands", type=int, default=2)  # <— multiple hands
    return parser.parse_args()

# ===================== Hand helpers (indices) =======================
WRIST = 0
TH_MCP, TH_IP, TH_TIP = 2, 3, 4
IX_MCP, IX_PIP, IX_DIP, IX_TIP = 5, 6, 7, 8
MI_MCP, MI_PIP, MI_DIP, MI_TIP = 9, 10, 11, 12
RI_MCP, RI_PIP, RI_DIP, RI_TIP = 13, 14, 15, 16
PI_MCP, PI_PIP, PI_DIP, PI_TIP = 17, 18, 19, 20

# Point-history labels that count as "finger spin"
SPIN_KEYWORDS = {
    "spin", "spinning", "circle", "circling",
    "cw", "clockwise", "ccw", "counterclockwise",
    "rotate", "rotation"
}

# Hand-sign labels that count as "Point" (case-insensitive)
POINT_LABELS = {"point", "pointer", "pointing"}

def _scale_from_points(pts):
    x, y, w, h = cv.boundingRect(pts.astype(np.int32))
    return (w**2 + h**2) ** 0.5 + 1e-6

def _pip_angle(pts, mcp, pip, dip):
    v1 = pts[mcp] - pts[pip]
    v2 = pts[dip] - pts[pip]
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a == 0 or b == 0: return 180.0
    cosang = np.clip(np.dot(v1, v2) / (a*b), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _curl_score(pts, mcp, pip, dip):
    # 0 = straight, 1 = curled (maps PIP angle 180°->0, 60°->1)
    ang = _pip_angle(pts, mcp, pip, dip)
    return float(np.clip((180.0 - ang) / 120.0, 0.0, 1.0))

def _others_mostly_folded(pts, tol=0.45):
    curls = [
        _curl_score(pts, IX_MCP, IX_PIP, IX_DIP),
        _curl_score(pts, MI_MCP, MI_PIP, MI_DIP),
        _curl_score(pts, RI_MCP, RI_PIP, RI_DIP),
        _curl_score(pts, PI_MCP, PI_PIP, PI_DIP),
    ]
    return (sum(c >= tol for c in curls) >= 3) and (np.mean(curls) >= (tol - 0.05))

def _thumb_extended_and_up(pts):
    v1 = pts[TH_MCP] - pts[TH_IP]
    v2 = pts[TH_TIP] - pts[TH_IP]
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    ang = 180.0 if a == 0 or b == 0 else math.degrees(math.acos(np.clip(np.dot(v1, v2)/(a*b), -1.0, 1.0)))
    dir_vec = pts[TH_TIP] - pts[TH_MCP]
    n = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / n if n > 0 else np.array([0.0, 0.0])
    upness = -dir_vec[1]  # +1 up, -1 down
    return (ang > 150.0, upness, dir_vec)

def is_thumbs_up(landmark_list_xy):
    pts = np.asarray(landmark_list_xy, dtype=np.float32)
    s = _scale_from_points(pts)
    if not _others_mostly_folded(pts, tol=0.45):
        return False
    th_ext, upness, dir_vec = _thumb_extended_and_up(pts)
    if not th_ext:
        return False
    vertical_ok = (abs(upness) > 0.35) and (abs(upness) > abs(dir_vec[0]) * 0.8)
    if not (upness > 0 and vertical_ok):
        return False
    knuckle_y = 0.5 * (pts[IX_MCP][1] + pts[PI_MCP][1])
    margin = 0.06 * s
    tip_y = pts[TH_TIP][1]
    return tip_y < (knuckle_y - margin)

def _finger_extended(pts, mcp, pip, dip, tip, thres_deg=160):
    return _pip_angle(pts, mcp, pip, dip) > thres_deg

def is_open_palm(landmark_list_xy):
    """
    Open/Stop ✋: all four non-thumb fingers extended fairly straight,
    and a decent span between index and pinky tips.
    """
    pts = np.asarray(landmark_list_xy, dtype=np.float32)
    s = _scale_from_points(pts)

    idx = _finger_extended(pts, IX_MCP, IX_PIP, IX_DIP, IX_TIP)
    mid = _finger_extended(pts, MI_MCP, MI_PIP, MI_DIP, MI_TIP)
    rin = _finger_extended(pts, RI_MCP, RI_PIP, RI_DIP, RI_TIP)
    pin = _finger_extended(pts, PI_MCP, PI_PIP, PI_DIP, PI_TIP)

    if idx and mid and rin and pin:
        curls = [
            _curl_score(pts, IX_MCP, IX_PIP, IX_DIP),
            _curl_score(pts, MI_MCP, MI_PIP, MI_DIP),
            _curl_score(pts, RI_MCP, RI_PIP, RI_DIP),
            _curl_score(pts, PI_MCP, PI_PIP, PI_DIP),
        ]
        avg_curl = float(np.mean(curls))
        span = np.linalg.norm(pts[IX_TIP] - pts[PI_TIP]) / s
        return (avg_curl <= 0.20) and (span >= 0.28)
    return False

# ===================== Original helpers (unchanged drawing & IO) =====================
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((landmark_x, landmark_y))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

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
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        # Index
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
        # Middle
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
        # Ring
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
        # Little
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        r_big = 8 if index in (4, 8, 12, 16, 20) else 5
        cv.circle(image, (landmark[0], landmark[1]), r_big, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), r_big, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

# ===================== Start Gate =====================
class StartGate:
    """
    Arms on pause/launch. Requires a stable 👍 for N frames to fire once.
    """
    def __init__(self, stable_frames=START_STABLE_FRAMES):
        self.stable_needed = int(max(1, stable_frames))
        self.stable = 0
        self.armed = True

    def reset_and_arm(self):
        self.stable = 0
        self.armed = True

    def disarm(self):
        self.armed = False
        self.stable = 0

    def update(self, thumbs_now: bool) -> bool:
        """
        Returns True exactly once when a stable thumbs-up is achieved while armed.
        """
        if not self.armed:
            self.stable = 0
            return False

        if thumbs_now:
            self.stable += 1
            if self.stable >= self.stable_needed:
                self.disarm()
                return True
        else:
            # decay quickly to avoid accidental holds
            self.stable = max(0, self.stable - 2)
        return False

# ===================== Main (MULTI-HAND) =====================
def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe Hands with multiple hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=max(1, args.max_hands),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_histories = {"Left": deque(maxlen=history_length), "Right": deque(maxlen=history_length)}
    finger_gesture_histories = {"Left": deque(maxlen=history_length), "Right": deque(maxlen=history_length)}

    mode = 0

    # Start bobbing paused, interval mid-tempo
    bob = HeadBobber(host=HOST, port=PORT, path=OSC_PATH,
                     up=UP_ANGLE, down=DOWN_ANGLE, speed=SPEED, interval=1.0)

    # >>> Start gate (await thumbs-up)
    start_gate = StartGate(stable_frames=START_STABLE_FRAMES)

    # timebase for ramp
    last_t = time.time()

    try:
        while True:
            fps = cvFpsCalc.get()

            # Quit on q/ESC
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            number, mode = select_mode(key, mode)

            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # per-frame aggregate decisions
            want_stop = False
            want_go = False
            saw_point_any = False
            saw_spin_any = False
            thumbs_now_any = False  # >>> for start gate

            # track which hands were seen this frame
            seen_hands = set()

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"
                    seen_hands.add(hand_label)
                    # Ensure deques exist for this label (safety)
                    if hand_label not in point_histories:
                        point_histories[hand_label] = deque(maxlen=history_length)
                    if hand_label not in finger_gesture_histories:
                        finger_gesture_histories[hand_label] = deque(maxlen=history_length)

                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Hand sign classification (static)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    # Maintain PER-HAND point history (index tip if "Point" id==2)
                    if hand_sign_id == 2:
                        point_histories[hand_label].append(landmark_list[8])
                    else:
                        point_histories[hand_label].append([0, 0])

                    # Build preprocessed point history FOR THIS HAND
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_histories[hand_label]
                    )

                    # Finger gesture classification (temporal) per hand
                    finger_gesture_id = 0
                    if len(pre_processed_point_history_list) == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    finger_gesture_histories[hand_label].append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_histories[hand_label]).most_common()
                    finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]]

                    # Aggregate decisions
                    if hand_sign_text.strip().lower() in POINT_LABELS:
                        saw_point_any = True
                    if finger_gesture_text.strip().lower() in SPIN_KEYWORDS:
                        saw_spin_any = True

                    # Draw for this hand
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)

                    # STOP/GO from this hand
                    if is_open_palm(landmark_list):
                        hand_sign_text_draw = "Open Hand"
                        want_stop = True
                    elif is_thumbs_up(landmark_list):
                        hand_sign_text_draw = "Thumbs Up"
                        thumbs_now_any = True          # >>> feed start gate
                        if not start_gate.armed:
                            # Only acts as "GO" when not awaiting start
                            want_go = True
                    else:
                        hand_sign_text_draw = hand_sign_text

                    debug_image = draw_info_text(
                        debug_image, brect, handedness, hand_sign_text_draw, finger_gesture_text
                    )

            # For any hand NOT seen this frame, keep timeline moving with [0,0]
            for hand_label in ("Left", "Right"):
                if hand_label not in seen_hands:
                    point_histories[hand_label].append([0, 0])

            # >>> START GATE: while armed, ignore other GO signals and wait for stable 👍
            if start_gate.armed:
                triggered = start_gate.update(thumbs_now_any)
                if triggered:
                    on_start_playback()   # call your music start
                    bob.resume()          # and start bobbing
                # Status overlay
                need = max(0, start_gate.stable_needed - start_gate.stable)
                cv.putText(debug_image, f"Awaiting 👍 to start ({need} frames)",
                           (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv.LINE_AA)
            else:
                # Apply control: STOP has priority over GO when not awaiting start
                if want_stop:
                    bob.pause()
                    start_gate.reset_and_arm()  # >>> require another 👍 after stop
                    cv.putText(debug_image, "SHIMON: STOP (re-armed)", (10, 120),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv.LINE_AA)
                elif want_go or saw_spin_any:
                    bob.resume()
                    msg = "SHIMON: GO"
                    if saw_spin_any:
                        msg += " (Spin)"
                    cv.putText(debug_image, msg, (10, 120),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)

            # Smoothly ramp the interval (Point from ANY hand speeds up)
            now = time.time()
            dt = max(0.0, now - last_t)
            last_t = now

            if (not start_gate.armed) and (saw_point_any or saw_spin_any):
                bob.nudge_interval(-RAMP_FASTER_PER_S * dt)  # faster (shorter)
            elif not start_gate.armed:
                bob.nudge_interval(+RAMP_SLOWER_PER_S * dt)  # slower (longer)

            # HUD
            debug_image = draw_info(debug_image, fps, mode, number)
            state = "ON" if bob.running else "PAUSED"
            cv.putText(debug_image, f"Bobbing: {state}", (10, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(debug_image, f"Bobbing: {state}", (10, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(debug_image, f"Interval: {bob.interval:.2f}s", (10, 175),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(debug_image, f"Interval: {bob.interval:.2f}s", (10, 175),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

            # Trails
            debug_image = draw_point_history(debug_image, point_histories["Left"])
            debug_image = draw_point_history(debug_image, point_histories["Right"])

            cv.imshow('Hand Gesture Recognition + Shimon Control (Multi-hand + Thumbs-Up Start)', debug_image)

    finally:
        bob.shutdown()
        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
