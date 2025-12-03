def send_gesture_to_shimon(part, pos, vel, host="192.168.1.1", port=9000):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/head-commands", [part, pos, vel])

from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("192.168.1.1", 9000)

def disappointed_gesture(): 
    send_gesture_to_shimon("NECK", -0.4, 10)
    send_gesture_to_shimon("HEADTILT", -1.0, 15)
    time.sleep(1.0)

    for _ in range(2):
        send_gesture_to_shimon("NECKPAN", -1, 25)
        time.sleep(0.5)
        send_gesture_to_shimon("NECKPAN", 1, 25)
        time.sleep(0.5)

    send_gesture_to_shimon("BASEPAN", 0.0, 10)
    send_gesture_to_shimon("HEADTILT", -0.4, 10)
    send_gesture_to_shimon("NECK", 0.0, 10)
    #send_gesture_to_shimon("EYEBROW", 50, 0)
    time.sleep(0.7)

def head_nod():
    """Gentle nod (yes gesture)."""
    send_gesture_to_shimon("NECK", 0.3, 15)
    send_gesture_to_shimon("HEADTILT", 0.2, 20)
    time.sleep(0.5)

    send_gesture_to_shimon("NECK", -0.3, 15)
    send_gesture_to_shimon("HEADTILT", -0.4, 20)
    time.sleep(0.5)

    send_gesture_to_shimon("NECK", 0.0, 15)
    send_gesture_to_shimon("HEADTILT", 0.0, 20)
    time.sleep(0.3)

def surprised_expression():
    """Surprised reaction using eyebrows + mouth + slight head tilt."""
    send_gesture_to_shimon("EYEBROWS", 0.5, 15)
    send_gesture_to_shimon("MOUTH", 0.6, 15)
    send_gesture_to_shimon("HEADTILT", 0.4, 12)
    time.sleep(1.0)

    time.sleep(0.5)

    send_gesture_to_shimon("EYEBROWS", 0.1, 10)
    send_gesture_to_shimon("MOUTH", 0.1, 10)
    send_gesture_to_shimon("HEADTILT", 0.0, 10)
    time.sleep(0.7)


if __name__ == "__main__":
    disappointed_gesture()