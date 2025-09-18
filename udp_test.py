from pythonosc import udp_client

def send_to_shimon(host="192.168.1.1", port=9000):
    client = udp_client.SimpleUDPClient(host, port)
    client.send_message("/head-commands", ["NECK", 0.1, 3])

send_to_shimon()
