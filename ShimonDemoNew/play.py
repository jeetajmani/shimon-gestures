import time
import rtmidi
from pythonosc import udp_client
from cv import start_gestures_monitor
import threading
import numpy as np

accepting_eye_contact = threading.Event()
shimon_turn = False
recording_done = threading.Event()
accepting_thumb = threading.Event()
thumb_result = None
thumb_lock = threading.Lock()

def send_note_to_shimon(note, velocity, host="192.168.1.1", port=9010):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/arm", [note, velocity])

def send_gesture_to_shimon(part, pos, vel, host="192.168.1.1", port=9000):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/head-commands", [part, pos, vel])

def look_left():
    send_gesture_to_shimon("BASEPAN", -0.7, 5)
    send_gesture_to_shimon("NECK", -0.2, 10)
    
def look_forward():
    send_gesture_to_shimon("BASEPAN", 0, 5)
    send_gesture_to_shimon("NECK", 0.2, 10)
    
def process_midi_phrase_dict(events, temperature: float = 1.0):
    """
    Randomly alters the notes in a MIDI phrase (list of dicts)
    according to a temperature parameter between 0 and 1.
    
    Each event is expected to be a dict with a 'note' key.
    """
    temperature = np.clip(temperature, 0, 1)

    # how many notes to modify
    n_events = len(events)
    n_to_change = int(np.random.randint(0, max(1, int(n_events * temperature))))

    if n_to_change == 0:
        return events

    # weighted probability (notes near center more likely to change)
    w = np.hanning(n_events) + 1e-6
    p = w / np.sum(w)

    # indices to change
    change_indices = np.random.choice(np.arange(n_events), n_to_change, replace=False, p=p)

    # get the unique set of note pitches present
    all_notes = np.array([e["note"] for e in events])
    unique_notes = np.unique(all_notes)

    # mutate notes
    for i in change_indices:
        old_note = events[i]["note"]
        new_note = int(np.random.choice(unique_notes))
        events[i]["note"] = new_note
        print(f"Changed note {old_note} → {new_note} at index {i}")

    return events

def play_sequence(events, temperature):
    events = process_midi_phrase_dict(events, temperature=temperature)
    recording_done.clear()
    look_forward()
    for event in events:
        if (event["index"] == 0):
            time.sleep(2)
        else: 
            time.sleep(event["delta"])
    
        send_note_to_shimon(event["note"], event["velocity"])
        
        print(f"Sent: {event}")

def on_eye_contact():
    if (accepting_eye_contact.is_set() and not shimon_turn):
        print("Eye contact callback triggered")
        recording_done.set()
        accepting_eye_contact.clear()
    else:
        print('cant start yet')
        
def on_thumb_detected(direction):
    global thumb_result
    if (accepting_thumb.is_set() and not shimon_turn):
        print(f"Thumb callback triggered: {direction}")
        with thumb_lock:
            thumb_result = direction
        accepting_thumb.clear()
    else:
        print('cant look at thumb yet')

def keyboard_phrase(port_index):
    print("KEYBOARD SEND")
    
    events = []
    start_time = time.time()
    last_time = start_time
    
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(port_index)
    
    i = 0
    while True:
        try: 
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
                play_sequence(events, 0)
                
                accepting_thumb.set()
                
                look_left()
                global thumb_result
                while True:
                    with thumb_lock:
                        if thumb_result is not None:
                            result = thumb_result
                            thumb_result = None
                            break
                        
                if (result == 0):
                    play_sequence(events, temperature=1.0)
                elif (result == 1):
                    print("liked it")
                        
                break
            
        except KeyboardInterrupt:
            exit()
                    
if __name__ == "__main__":
    # List all available MIDI input ports
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    print("Available ports:", ports)

    # Ask user to choose a port
    port_index = int(input("Select input port number: "))

    start_gestures_monitor(on_eye_contact, on_thumb_detected)
    
    for i in range(3):
        look_left()
    
        keyboard_phrase(port_index)
        
