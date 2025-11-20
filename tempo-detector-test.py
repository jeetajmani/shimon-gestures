import time
import threading
from collections import deque

import numpy as np
import keyboard  # pip install keyboard

# --- NOD TEMPO STATE (matches what we described for play.py) ---
midi_ioi_window = deque(maxlen=5)  # last 5 IOIs = last 6 notes
nod_bpm = None
nod_bpm_lock = threading.Lock()
last_event_ts = None  # time of last *valid* delta used for tempo


def normalize_ioi_to_beat(dt, min_bpm=60.0, max_bpm=180.0):
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

    print(f"[TEMPO] Updated from last ~6 events: {bpm:.1f} BPM")


def nod_loop(idle_timeout=2.0):
    """
    Background loop: keeps nodding at the *current* tempo
    inferred from event deltas.

    - Tempo is recomputed from the last 6 events (via update_nod_tempo_from_delta).
    - If there are no events for > idle_timeout seconds, nodding pauses.
    """
    global nod_bpm, last_event_ts

    next_nod_time = None

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
            print(f"[NOD] ~{bpm:.1f} BPM")
            # In the real robot code you would call: shimon_nod()

            # Re-read bpm each time so nod spacing adapts if tempo changed
            with nod_bpm_lock:
                bpm = nod_bpm if nod_bpm is not None else bpm
            beat_interval = 60.0 / bpm
            next_nod_time += beat_interval
        else:
            dt = max(0.0, next_nod_time - now)
            time.sleep(min(0.02, dt))


def main():
    print("========================================")
    print(" DELTA-BASED TEMPO TEST (last 6 events)")
    print(" - Press 'a' to simulate a MIDI note/event")
    print(" - Tempo is computed from the last 6 events")
    print(" - Nods printed at current tempo")
    print(" - If you stop for ~2s, nodding pauses")
    print(" - Press ESC to quit")
    print("========================================")

    # Start nodding thread
    t = threading.Thread(target=nod_loop, daemon=True)
    t.start()

    last_ts = None

    try:
        while True:
            if keyboard.is_pressed("esc"):
                print("Exiting...")
                break

            if keyboard.is_pressed("a"):
                now = time.time()
                print(f"\n[EVENT] 'a' at {now:.3f}")

                if last_ts is not None:
                    delta = now - last_ts
                    print(f"  delta = {delta:.3f} s")
                    update_nod_tempo_from_delta(delta)
                else:
                    print("  first event (no delta yet)")

                last_ts = now

                # debounce: avoid repeats while holding 'a'
                while keyboard.is_pressed("a"):
                    time.sleep(0.01)

            time.sleep(0.005)
    finally:
        # thread is daemon; it will exit when main ends
        pass


if __name__ == "__main__":
    main()
