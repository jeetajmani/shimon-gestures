import time
import rtmidi
from pythonosc import udp_client
from cv import start_gestures_monitor
from cv import tempo_detection_enabled
import threading
import numpy as np
import mido

accepting_eye_contact = threading.Event()
shimon_turn = False
recording_done = threading.Event()

accepting_approval = threading.Event()
approval_result = None
approval_lock = threading.Lock()

accepting_tempo = threading.Event()
tempo = None
tempo_lock = threading.Lock()

part_1_flag = -1
phrase_num = 0

NOTE_LATENCY = 0.5

def send_to_max(message, value, host="127.0.0.1", port=7402):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message(message, value)

def send_note_to_shimon(note, velocity, host="192.168.1.1", port=9010):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/arm", [note, velocity])

def send_gesture_to_shimon(part, pos, vel, host="192.168.1.1", port=9000):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/head-commands", [part, pos, vel])

def look_left():
    send_gesture_to_shimon("BASEPAN", -0.7, 5)
    send_gesture_to_shimon("NECK", -0.2, 10)
    send_gesture_to_shimon("HEADTILT", 0, 8)
    
def look_forward():
    send_gesture_to_shimon("BASEPAN", 0, 5)
    send_gesture_to_shimon("NECK", 0.2, 10)

def shimon_sad():
    send_gesture_to_shimon("NECK", -0.4, 7)
    send_gesture_to_shimon("HEADTILT", -1.2, 8)
    send_gesture_to_shimon("BASEPAN", 0, 5)

def shimon_nod():
    send_gesture_to_shimon("NECK", 0.3, 10)
    send_gesture_to_shimon("HEADTILT", -1, 20)
    time.sleep(0.3)
    
    send_gesture_to_shimon("NECK", -0.3, 10)
    send_gesture_to_shimon("HEADTILT", 0.4, 20)
    
# def process_midi_phrase_dict(events, temperature: float = 1.0):
#     """
#     Randomly alters the notes in a MIDI phrase (list of dicts)
#     according to a temperature parameter between 0 and 1.

#     Only swaps notes that have non-zero velocity.
#     """
#     temperature = np.clip(temperature, 0, 1)

#     # Collect indices of notes with non-zero velocity
#     active_indices = [i for i, e in enumerate(events) if e["velocity"] > 0]
#     n_events = len(active_indices)
#     if n_events == 0:
#         return events

#     # Determine how many to modify (scaled by temperature)
#     n_to_change = int(np.random.randint(0, max(1, int(n_events * temperature))))
#     if n_to_change == 0:
#         return events

#     # Weighted probability (notes near center more likely to change)
#     w = np.hanning(n_events) + 1e-6
#     p = w / np.sum(w)

#     # Choose indices among active notes
#     change_indices = np.random.choice(active_indices, n_to_change, replace=False, p=p)

#     # Unique playable pitches (non-zero velocity only)
#     all_notes = np.array([e["note"] for e in events if e["velocity"] > 0])
#     unique_notes = np.unique(all_notes)

#     # Perform swaps
#     for i in change_indices:
#         old_note = events[i]["note"]
#         new_note = int(np.random.choice(unique_notes))
#         events[i]["note"] = new_note
#         print(f"Changed note {old_note} → {new_note} at index {i}")

#     return events

## NEW TEMPERATURE FUNCTION
def process_midi_phrase_dict(events, temperature: float = 1.0):
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
        old_note = events[i]["note"]
        
        # scale maximum step by temperature (0-12 semitones)
        max_step = max(1, int(np.ceil(12 * temperature)))
        steps = np.random.randint(1, max_step + 1)
        direction = np.random.choice([-1, 1])
        new_note = int(old_note + direction * steps)

        # keep note in a playable range (48..95) by octave-shifting if necessary
        while new_note < 48:
            new_note += 12
        while new_note > 95:
            new_note -= 12

        events[i]["note"] = new_note
        print(f"Changed note {old_note} → {new_note} at index {i}")

    return events

def play_sequence(events, temperature=1.0):
    events = process_midi_phrase_dict(events, temperature=temperature)
    recording_done.clear()
    events[0]["delta"] = 0.01
    for event in events:
        time.sleep(event["delta"])
        
        if (event["velocity"] != 0):
            send_note_to_shimon(event["note"], 120)
            print(f"Sent: {event}")
    
    accepting_approval.set()
    look_left()

def on_eye_contact():
    if (accepting_eye_contact.is_set() and not shimon_turn):
        print("Eye contact callback triggered")
        recording_done.set()
        accepting_eye_contact.clear()
    else:
        print('cant start yet')
        
def on_approval_detected(direction):
    global approval_result
    if (accepting_approval.is_set() and not shimon_turn):
        print(f"Approval callback triggered: {direction}")
        with approval_lock:
            approval_result = direction
        accepting_approval.clear()
    else:
        print('cant look at approval~ yet')

def on_tempo_detected(detected_tempo):
    global tempo
    if (accepting_tempo.is_set()):
        with tempo_lock:
            tempo = detected_tempo
        print(f"🎵 Detected nod tempo: {tempo:.1f} BPM")
        accepting_tempo.clear()
    else:
        print('Cant detect tempo yet!')
        
def keyboard_phrase(port_index):
    global phrase_num
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
                    
                    if (len(events) >= 5):
                        print("NOW WE CAN SEE")
                        accepting_eye_contact.set()
                    
                if (recording_done.is_set()):
                    print("ALL DONE")

                    look_forward()
                    play_sequence(events, 1.0)
                    
                    while True:  # approval loop
                        global approval_result
                        print("WAITING FOR APPROVAL...")
                        with approval_lock:
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
                            shimon_sad()
                            play_sequence(events, temperature=0.5)
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
            exit()

  
def tempo_detect():
    global tempo
    print("Waiting for tempo from nods...")
    while True:
        with tempo_lock:
            if tempo is not None:
                bpm = tempo
                break
    
    print(f"Tempo confirmed: {bpm:.1f} BPM — starting head movement")
    tempo_detection_enabled.clear()
    
    try:
        send_to_max("/tempo", bpm)
        send_to_max("/status", 1)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        send_to_max("/status", 0)

def render_midi_at_tempo(input_filename, output_filename, track_index=1):
    """
    Extracts a single track from a MIDI file and writes a new file
    that plays at the specified BPM, ignoring internal tempo changes.

    Keeps the chosen track only, with its original delta timing.
    """
    global tempo
    
    mid = mido.MidiFile(input_filename)
    ppq = mid.ticks_per_beat
    seconds_per_beat = 60.0 / tempo
    us_per_beat = int(seconds_per_beat * 1_000_000)

    if track_index >= len(mid.tracks):
        raise IndexError(f"MIDI only has {len(mid.tracks)} tracks (0–{len(mid.tracks)-1})")

    print(f"🎼 Extracting Track {track_index} from '{input_filename}'")
    print(f"Target tempo: {tempo:.1f} BPM | PPQ = {ppq}")

    new_mid = mido.MidiFile(ticks_per_beat=ppq)
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)

    # Add your target tempo message at start
    new_track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))

    # Copy only messages from the selected track, ignoring any tempo events
    for msg in mid.tracks[track_index]:
        if msg.is_meta and msg.type == "set_tempo":
            continue
        new_track.append(msg.copy())

    new_mid.save(output_filename)
    print(f"✅ Saved: {output_filename}")
    
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

def shimon_synced():
    # note_thread = threading.Thread(target=play_notes, daemon=True)
    note_thread = threading.Thread(target=play_midi_track, args=("furelise.mid",), daemon=True)
    neck_thread = threading.Thread(target=move_neck_and_head, daemon=True)
    note_thread.start()
    time.sleep(0.5)
    neck_thread.start()
    
    while True:
        time.sleep(1)

# def shimon_move_to_tempo():
#     global tempo
#     beat_interval = 60.0 / tempo
#     half_beat = beat_interval / 2.0
    
#     t0 = time.monotonic()
#     next_beat = t0
#     beat_count = 0
#     neck_direction = 1
    
#     note_triggered = False
    
#     while True:
#         now = time.monotonic()
        
#         if not note_triggered and (next_beat - now) <= NOTE_LATENCY:
#             note = int(np.random.randint(48, 96))  # inclusive 48–95
#             send_note_to_shimon(note, 68)
#             print(f"Triggered note {note}")
#             # send_note_to_shimon(59, 68)
#             note_triggered = True
        
#         if now >= next_beat:
#             note_triggered = False
            
#             # Downward motion
#             neck_pos = 0.1 * neck_direction
#             send_gesture_to_shimon("NECK", neck_pos, 15)
#             neck_direction *= -1
            
#             send_gesture_to_shimon("HEADTILT", -0.2, 20)

#             # Schedule the 'up' event halfway through this beat
#             up_time = next_beat + half_beat

#             # Move head back up exactly halfway through the beat
#             while time.monotonic() < up_time:
#                 time.sleep(0.001)  # tiny wait to hold timing accuracy

#             send_gesture_to_shimon("HEADTILT", 0.2, 20)

#             beat_count += 1
#             next_beat = t0 + beat_count * beat_interval
    
if __name__ == "__main__":
    # List all available MIDI input ports
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    print("Available ports:", ports)
    
    # send_to_max("/tempo", 120)
    # send_to_max("/status", 1)

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
    
    look_forward()
    
    look_forward()
        
        
