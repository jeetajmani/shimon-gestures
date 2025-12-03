import time
import rtmidi
from pythonosc import udp_client
from cv import start_gestures_monitor
from cv import tempo_detection_enabled
import threading
import numpy as np
import mido
import copy
from collections import deque
import cv2  # <--- NEW: for optical flow

accepting_eye_contact = threading.Event()
shimon_turn = False
recording_done = threading.Event()

accepting_approval = threading.Event()
approval_result = None
approval_lock = threading.Lock()

accepting_tempo = threading.Event()
tempo = None
tempo_lock = threading.Lock()

# tempo nodding variables
# last 6 notes are used for beat detection
# --- NOD TEMPO STATE (for live nodding based on MIDI deltas) ---
midi_ioi_window = deque(maxlen=5)  # last 5 IOIs = last 6 notes
nod_bpm = None
nod_bpm_lock = threading.Lock()
last_event_ts = None  # time of last *valid* delta used for tempo
sad_gesture_stop = threading.Event()

part_1_flag = -1
phrase_num = 0

NOTE_LATENCY = 0.5

basepan_pos = None
neck_pos = None
headtile_pos = None

temperature = 0.75

free_play = False

SHIMON_ARM = udp_client.SimpleUDPClient("192.168.1.1", 9010)
SHIMON_HEAD = udp_client.SimpleUDPClient("192.168.1.1", 9000)
MAX_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7402)

def send_to_max(message, value):
    MAX_CLIENT.send_message(message, value)

# =========================
# OPTICAL FLOW MOTION INPUT
# =========================

# --- Tunables (copied / adapted from your optical-flow script) ---
RESIZE_FACTOR = 0.5

FLOW_NORMALIZE = 1.5        # magnitude that maps to 100
MAX_FLOW_DISPLAY = 100.0

BASELINE_FLOOR = 0.12       # dead-zone for tiny noise

USE_PERCENTILE = True
PERCENTILE = 90

COMPUTE_EVERY_N = 3         # compute flow 1 out of N frames

WINDOW_SIZE = 5             # smoothing window for displayed value

PRINT_INTERVAL_SEC = 1.0    # print motion once per second

# --- State for optical flow thread ---
flow_history = deque(maxlen=WINDOW_SIZE)
last_flow_value = 0.0
sum_since_print = 0.0
count_since_print = 0
last_print_ts = time.time()
frame_idx = 0

# This is the shared "how much the player is moving" signal: 0..1
motion_intensity = 0.0
motion_lock = threading.Lock()


def optical_flow_loop():
    """
    Background thread:
    - Reads from webcam
    - Computes dense optical flow every N frames
    - Produces a motion_intensity in [0, 1]
    which is used by nod_loop to scale how far up/down Shimon moves.
    """
    global last_flow_value, sum_since_print, count_since_print
    global last_print_ts, frame_idx, motion_intensity

    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    if not ret:
        print("[OPTICAL FLOW] Failed to grab initial frame")
        return

    prev_small = cv2.resize(prev_frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    print("[OPTICAL FLOW] Started. Using webcam 0 for motion intensity.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[OPTICAL FLOW] Frame grab failed, stopping.")
            break

        frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        computed_this_frame = False
        if frame_idx % COMPUTE_EVERY_N == 0:
            # --- Dense optical flow (Farneback) ---
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )

            # --- Motion measure ---
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if USE_PERCENTILE:
                flow_measure = np.percentile(mag, PERCENTILE)
            else:
                flow_measure = np.mean(mag)

            # --- Raise the floor (dead-zone) ---
            flow_measure = max(flow_measure - BASELINE_FLOOR, 0.0)

            # --- Normalize to 0–100 and smooth lightly ---
            flow_value = min((flow_measure / FLOW_NORMALIZE) * 100.0, MAX_FLOW_DISPLAY)
            flow_history.append(flow_value)
            last_flow_value = float(np.mean(flow_history))

            prev_gray = gray
            computed_this_frame = True

            # Accumulate for 1-second average print
            sum_since_print += last_flow_value
            count_since_print += 1

            # Map to [0, 1] for motion_intensity
            norm_intensity = min(last_flow_value / MAX_FLOW_DISPLAY, 1.0)

            with motion_lock:
                motion_intensity = norm_intensity

        # --- Print average once per second ---
        now = time.time()
        if now - last_print_ts >= PRINT_INTERVAL_SEC:
            if count_since_print > 0:
                avg_last_sec = sum_since_print / count_since_print
                print(f"[OPTICAL FLOW] avg (last {PRINT_INTERVAL_SEC:.0f}s): {avg_last_sec:.1f}/100")
            sum_since_print = 0.0
            count_since_print = 0
            last_print_ts = now

        frame_idx += 1

        # No cv2.imshow / waitKey here to keep this thread headless.
        # If you want a debug window, you can add imshow + waitKey
        # but GUI-from-thread can be finicky on some systems.

    cap.release()
    print("[OPTICAL FLOW] Stopped.")


## NEW nod at playing tempo helper
def normalize_ioi_to_beat(dt, min_bpm=60.0, max_bpm=150.0):
    """
    Map a raw inter-onset interval dt (seconds) to an approximate beat duration,
    assuming dt may be a subdivision (eighth, sixteenth, etc.).
    Scales by factors of 2 until BPM is in [min_bpm, max_bpm].
    """
    if dt <= 0:
        return None

    beat = dt
    bpm = 60.0 / beat

    # Too fast → treat as subdivision (double the beat)
    while bpm > max_bpm:
        beat *= 2.0
        bpm = 60.0 / beat

    # Too slow → treat as multiple beats (halve the beat)
    while bpm < min_bpm:
        beat /= 2.0
        bpm = 60.0 / beat

    return beat


def update_nod_tempo_from_delta(dt):
    """
    Called with the delta time between *this* event and the previous one.

    Uses ONLY the last 6 events (last 5 IOIs) to estimate tempo.
    Also updates last_event_ts so nod_loop can detect idle periods.
    """
    global nod_bpm, last_event_ts

    # Ignore junk intervals
    if not (0.05 < dt < 3.0):
        return

    midi_ioi_window.append(dt)

    # Need at least a few intervals before trusting tempo
    if len(midi_ioi_window) < 3:
        last_event_ts = time.time()
        return

    beat_estimates = []
    for d in midi_ioi_window:
        bt = normalize_ioi_to_beat(d, min_bpm=60.0, max_bpm=180.0)
        if bt is not None:
            beat_estimates.append(bt)

    if not beat_estimates:
        last_event_ts = time.time()
        return

    beat_sec = float(np.median(beat_estimates))
    bpm = 60.0 / beat_sec

    with nod_bpm_lock:
        nod_bpm = bpm
        last_event_ts = time.time()

    # Optional debug:
    # print(f"[TEMPO] Updated from last ~6 events: {bpm:.1f} BPM")


def nod_loop(idle_timeout=2.0):
    """
    Background loop: keeps nodding at the *current* tempo
    inferred from event deltas.

    - Tempo is recomputed from the last 6 events (via update_nod_tempo_from_delta).
    - If there are no events for > idle_timeout seconds, nodding pauses.

    NOW ALSO:
    - Uses motion_intensity from optical_flow_loop to scale how far up/down
      Shimon moves. More player motion -> bigger nod amplitude.
    """
    global nod_bpm, last_event_ts

    next_nod_time = None

    nod_up = True

    while True:
        now = time.time()

        with nod_bpm_lock:
            bpm = nod_bpm
            last_ts = last_event_ts

        # If we have no tempo yet or no recent events, don't nod
        if (
            bpm is None or
            last_ts is None or
            (now - last_ts) > idle_timeout
        ):
            next_nod_time = None
            time.sleep(0.05)
            continue

        beat_interval = 60.0 / bpm

        # schedule first nod
        if next_nod_time is None:
            next_nod_time = now + beat_interval
            time.sleep(0.01)
            continue

        if now >= next_nod_time:
            # Get motion-based amplitude factor in [0, 1]
            with motion_lock:
                intensity = motion_intensity

            # Map intensity to a neck amplitude: e.g. 0.05..0.25
            # (small nods when player is still, bigger when moving a lot)
            base_amp = 0.05
            extra_amp = 0.20
            amp = base_amp + extra_amp * intensity

            print(f"[NOD] ~{bpm:.1f} BPM, amp={amp:.3f}, mot={intensity:.2f}")

            if nod_up:
                send_gesture_to_shimon("NECK", amp, 10)
                nod_up = False
            else:
                send_gesture_to_shimon("NECK", -amp, 10)
                nod_up = True

            # Re-read bpm each time so nod spacing adapts if tempo changed
            with nod_bpm_lock:
                bpm = nod_bpm if nod_bpm is not None else bpm
            beat_interval = 60.0 / bpm
            next_nod_time += beat_interval
        else:
            dt = max(0.0, next_nod_time - now)
            time.sleep(min(0.02, dt))
# end of beat detection nodding logic

def send_note_to_shimon(note, velocity):
    SHIMON_ARM.send_message("/arm", [note, velocity])

def send_gesture_to_shimon(part, pos, vel):
    SHIMON_HEAD.send_message("/head-commands", [part, pos, vel])
    
def look_left():
    send_gesture_to_shimon("BASEPAN", -0.675, 5)
    send_gesture_to_shimon("NECK", -0.2, 10)
    send_gesture_to_shimon("HEADTILT", 0, 8)
    idle(1)
    
def look_forward():
    idle(0)
    send_gesture_to_shimon("BASEPAN", 0, 5)
    send_gesture_to_shimon("NECK", 0.0, 10)

def shimon_sad():
    """Modified to be interruptible"""
    sad_gesture_stop.clear()
    
    idle(0)
    send_gesture_to_shimon("NECK", -0.4, 10)
    send_gesture_to_shimon("HEADTILT", -1.0, 15)
    
    if sad_gesture_stop.wait(1.0):  # Interruptible sleep
        return
 
    for _ in range(100):
        send_gesture_to_shimon("BASEPAN", -1.0, 3)
        if sad_gesture_stop.wait(0.5):
            return
        send_gesture_to_shimon("BASEPAN", 0.0, 3)
        if sad_gesture_stop.wait(0.5):
            return
        send_gesture_to_shimon("BASEPAN", 1.0, 3)
        if sad_gesture_stop.wait(0.5):
            return
 
    if sad_gesture_stop.wait(0.7):
        return   

def shimon_nod():
    send_gesture_to_shimon("NECK", 0.3, 10)
    send_gesture_to_shimon("HEADTILT", -1, 20)
    time.sleep(0.3)
    
    send_gesture_to_shimon("NECK", -0.3, 10)
    send_gesture_to_shimon("HEADTILT", 0.4, 20)
    time.sleep(0.3)

def idle(status):   
    send_to_max("/idle", status)

## NEW TEMPERATURE FUNCTION
def process_midi_phrase_dict(events, temperature):
    """
    Randomly alters the notes in a MIDI phrase (list of dicts)
    according to a temperature parameter between 0 and 1.

    Only swaps notes that have non-zero velocity.
    """
    temperature = np.clip(temperature, 0, 1)

    # Collect indices of notes with non-zero velocity
    active_indices = [i for i, e in enumerate(events) if e["velocity"] > 0]
    n_events = len(active_indices)
    if n_events == 0:
        return events

    # Determine how many to modify (scaled by temperature)
    n_to_change = int(np.random.randint(0, max(1, int(n_events * temperature))))
    if n_to_change == 0:
        return events

    # Weighted probability (notes near center more likely to change)
    w = np.hanning(n_events) + 1e-6
    p = w / np.sum(w)

    # Choose indices among active notes
    change_indices = np.random.choice(active_indices, n_to_change, replace=False, p=p)

    # Unique playable pitches (non-zero velocity only)
    all_notes = np.array([e["note"] for e in events if e["velocity"] > 0])
    unique_notes = np.unique(all_notes)

    # Perform swaps
    for i in change_indices:
        old_note = events[i]["note"]
        
        # scale maximum step by temperature (0-12 semitones)
        max_step = max(1, int(np.ceil(12 * temperature)))
        steps = np.random.randint(1, max_step + 1)
        direction = np.random.choice([-1, 1])
        new_note = int(old_note + direction * steps)

        # keep note in a playable range (48..95) by octave-shifting if necessary
        new_note = ((new_note - 48) % 48) + 48

        events[i]["note"] = new_note
        print(f"Changed note {old_note} → {new_note} at index {i}")

    return events

def play_sequence(events, temperature):
    events = process_midi_phrase_dict(events, temperature)
    recording_done.clear()
    events[0]["delta"] = 0.01
    for event in events:
        time.sleep(event["delta"])
        
        if (event["velocity"] != 0):
            # *** NEW: update tempo from this event's delta ***
            update_nod_tempo_from_delta(event["delta"])

            send_note_to_shimon(event["note"], 120)
            print(f"Sent: {event}")
    
    sad_gesture_stop.set()
    accepting_approval.set()
    look_left()

def on_eye_contact():
    if (accepting_eye_contact.is_set() and not shimon_turn):
        print("Eye contact callback triggered")
        recording_done.set()
        accepting_eye_contact.clear()
    else:
        print('cant start yet')
        
    if free_play:
        send_to_max("/eyes", 1)
        
def on_approval_detected(direction):
    global approval_result
    if (accepting_approval.is_set() and not shimon_turn):
        print(f"Approval callback triggered: {direction}")
        with approval_lock:
            approval_result = direction
        accepting_approval.clear()
    else:
        print('cant look at approval~ yet')
        
    if free_play:
        print("FREEEEEEEE")
        with approval_lock:
            if approval_result is not None:
                result = approval_result
                if result == 0:
                    send_to_max("/eyes", 0)
                    accepting_approval.set()
                approval_result = None
        
def on_tempo_detected(detected_tempo):
    """Called by cv.py whenever a valid tempo burst is detected."""
    global tempo
    with tempo_lock:
        tempo = detected_tempo
    print(f"🎵 Detected nod tempo: {tempo:.1f} BPM")

        
def keyboard_phrase(port_index):
    global phrase_num
    global temperature
    temperature = 0.75
    print("KEYBOARD SEND")
    
    events = []
    start_time = time.time()
    last_time = start_time
    
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(port_index)
    
    i = 0
    while True:
        try: 
            if (part_1_flag == 1):
                msg = midi_in.get_message()
                if msg:
                    now = time.time()
                    delta = now - last_time
                    last_time = now
                    
                    event = {
                        "index": i,
                        "note": msg[0][1],
                        "velocity": msg[0][2],
                        "delta": delta
                    }
                    events.append(event)
                    print(event)
                    i += 1

                    # NEW: handle note for tempo detection
                    if event["velocity"] > 0:
                        update_nod_tempo_from_delta(event["delta"])
                    
                    if (len(events) >= 5):
                        print("NOW WE CAN SEE")
                        accepting_eye_contact.set()
                else:
                    time.sleep(0.001)
                    
                if (recording_done.is_set()):
                    print("ALL DONE")
                    events_original = copy.deepcopy(events)
                    look_forward()
                    play_sequence(events, temperature)
                    
                    while True:  # approval loop
                        global approval_result
                        print("WAITING FOR APPROVAL...")
                        approval_result = None  # reset before waiting
                        accepting_approval.set()

                        # Wait for approval signal
                        while True:
                            with approval_lock:
                                if approval_result is not None:
                                    result = approval_result
                                    approval_result = None
                                    break
                            time.sleep(0.05)

                        if result == 0:
                            print("!!DISLIKED - PLAYING AGAIN")
                            sad_thread = threading.Thread(target=shimon_sad, daemon=True)
                            sad_thread.start()
                            events_mod = copy.deepcopy(events_original)
                            play_sequence(events_mod, temperature/2)
                        elif result == 1:
                            print("!!LIKED IT")
                            shimon_nod()
                            break  # exit approval loop (nod detected)
                    
                    events.clear()
                    i = 0
                    recording_done.clear()
                    accepting_eye_contact.clear()
                    accepting_approval.clear()
                    print("READY FOR NEXT PHRASE")
                    break     
                           
            elif (part_1_flag == 0):
                while True:
                    accepting_eye_contact.set()
                    if (recording_done.is_set()):
                        phrase_num += 1
                        if (phrase_num == 1):
                            play_midi_track('CDL_1.mid')
                            break
                        elif (phrase_num == 2):
                            play_midi_track('CDL_2.mid')
                            break

                recording_done.clear()
                accepting_eye_contact.clear()
                print("READY FOR NEXT PHRASE")
                break
        except KeyboardInterrupt:
            look_forward()
            exit()
  
def tempo_detect():
    global tempo
    global free_play
    print("Waiting for tempo from nods...")

    # Make sure tempo mode is enabled on the CV side
    tempo_detection_enabled.set()

    # Clear any stale tempo
    with tempo_lock:
        tempo = None

    # --- FIRST TEMPO ---
    bpm = None
    while bpm is None:
        with tempo_lock:
            if tempo is not None:
                bpm = tempo
                tempo = None
        time.sleep(0.01)

    print(f"Tempo confirmed: {bpm:.1f} BPM — starting head movement")
    idle(0)
    free_play = True
    accepting_approval.set()
    try:
        send_to_max("/tempo", bpm)
        send_to_max("/status", 1)

        print("\n-------------")
        print("LIVE TEMPO MODE: Nod 3 times to set new tempo")
        print("-------------")

        # --- CONTINUOUS UPDATES ---
        while True:
            with tempo_lock:
                if tempo is not None:
                    bpm = tempo
                    tempo = None
                    send_to_max("/tempo", bpm)
                    print(f"🎛 LIVE TEMPO UPDATED → {bpm} BPM")
            time.sleep(0.01)

    except KeyboardInterrupt:
        send_to_max("/status", 0)
        look_forward()
    
def play_midi_track(input_filename, track_index=1):
    """Plays a MIDI file at its original tempo and sends notes to Shimon."""
    recording_done.clear()
    look_forward()

    mid = mido.MidiFile(input_filename)
    print(f"🎼 Playing track {track_index} from '{input_filename}' (original tempo)")

    # --- [1] Use mido's built-in playback iterator ---
    # It automatically converts tick times to seconds,
    # respecting tempo changes and time signatures.
    for msg in mid.play(meta_messages=True):  # meta_messages=True ensures tempo updates are respected
        if msg.type == "note_on" and msg.velocity > 0:
            note = msg.note
            velocity = msg.velocity

            # constrain playable range for Shimon
            if note < 48:
                note += 12
            elif note > 95:
                note -= 12

            send_note_to_shimon(note, velocity)
            print(f"🎹 Note {note} vel {velocity}")

    print("✅ Playback complete.")
    look_left()

    
def play_notes():
    """Plays notes right on each beat."""
    global tempo
    beat_interval = 60.0 / tempo
    t0 = time.monotonic()
    beat_count = 0
    while True:
        now = time.monotonic()
        next_beat = t0 + beat_count * beat_interval
        if now >= next_beat:
            note = int(np.random.randint(48, 96))
            send_note_to_shimon(note, 80)
            print(f"[{beat_count}] 🎹 Note {note} on beat")
            beat_count += 1

def move_neck_and_head():
    """Moves Shimon 500 ms after each beat to visually match audio."""
    global tempo
    beat_interval = 60.0 / tempo
    half_beat = beat_interval / 2.0

    t0 = time.monotonic()
    next_beat = t0
    beat_count = 0
    neck_direction = -1

    while True:
        now = time.monotonic()
        if now >= next_beat:
            # Downward motion
            neck_pos = 0.05 * neck_direction
            head_pos = -0.2 * neck_direction
            send_gesture_to_shimon("NECK", neck_pos, 15)
            send_gesture_to_shimon("HEADTILT", head_pos, 20)
            neck_direction *= -1

            beat_count += 1
            next_beat += beat_interval  # <— fixed here

if __name__ == "__main__":
    # List all available MIDI input ports
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    print("Available ports:", ports)

    # Ask user to choose a port
    port_index = int(input("Select input port number: "))
    part_1_flag = int(input("Select call-and-response (0) or turn-taking (1): "))
    if (part_1_flag == 0):
        print("------------------------------")
        print("------------ CALL AND RESPONSE")
        print("------------------------------")
    elif (part_1_flag == 1):
        print("------------------------------")
        print("------------------ TURN TAKING")
        print("------------------------------")

    start_gestures_monitor(on_eye_contact, on_approval_detected, on_tempo_detected)

    # NEW: start background nod thread (delta-based tempo)
    threading.Thread(target=nod_loop, daemon=True).start()

    # NEW: start background optical flow thread (motion -> nod amplitude)
    threading.Thread(target=optical_flow_loop, daemon=True).start()
    
    for i in range(2):
        look_left()
        keyboard_phrase(port_index)
        
    look_left()
    tempo_detection_enabled.set()
    accepting_tempo.set()
    print("------------------------------")
    print("---------- NOW DETECTING TEMPO")
    print("------------------------------")

    tempo_detect()
