import json
import socket
import struct
import time
import argparse


SO_TIMESTAMPING = getattr(socket, "SO_TIMESTAMPING", 37)
SOF_TIMESTAMPING_TX_SOFTWARE = 1 << 1
SOF_TIMESTAMPING_RX_SOFTWARE = 1 << 3
SOF_TIMESTAMPING_SOFTWARE = 1 << 4
MSG_ERRQUEUE = getattr(socket, "MSG_ERRQUEUE", 0x2000)

TIMESTAMP_STRUCT = struct.Struct("@llllll")


def extract_timestamp_ns(ancdata):
    for level, message_type, data in ancdata:
        if (
            level == socket.SOL_SOCKET 
            and message_type == SO_TIMESTAMPING
            ):
            if len(data) < TIMESTAMP_STRUCT.size:
                continue

            values = TIMESTAMP_STRUCT.unpack_from(data)

            seconds = values[0]
            nanoseconds = values[1]

            if seconds != 0 or nanoseconds != 0:
                return seconds * 1_000_000_000 + nanoseconds

    raise RuntimeError("Kernel timestamp missing")


def read_tx_timestamp_ns(sock, timeout_seconds=1):

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            data, ancdata, flags, peer = sock.recvmsg(
                2048,
                socket.CMSG_SPACE(TIMESTAMP_STRUCT.size),
                MSG_ERRQUEUE,
            )
            return extract_timestamp_ns(ancdata)

        except BlockingIOError:
            time.sleep(0.001)

    raise RuntimeError("Timeout waiting for kernel TX timestamp")


def start_server(host, port):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    timestamp_flags = (
        SOF_TIMESTAMPING_TX_SOFTWARE |
        SOF_TIMESTAMPING_RX_SOFTWARE |
        SOF_TIMESTAMPING_SOFTWARE
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        SO_TIMESTAMPING,
        timestamp_flags, 
    )


    sock.bind((host, port))

    print(f"UDP server listening on {host}: {port}")
    print(f"Timestamp structure size: {TIMESTAMP_STRUCT.size}")

    try:
        while True:
            data, ancdata, flags, peer = sock.recvmsg(
                2048,
                socket.CMSG_SPACE(TIMESTAMP_STRUCT.size),
            )

            t2_ns = extract_timestamp_ns(ancdata)
            request = json.loads(data.decode("utf-8"))
            sequence = request["sequence"]


            response = {
                "type": "response",
                "sequence": sequence,
            }

            sock.sendto(
                json.dumps(response).encode("utf-8"),
                peer,
            )

            t3_ns = read_tx_timestamp_ns(sock)

            follow_up = {
                "type": "follow_up",
                "sequence": sequence,
                "t2_ns": t2_ns,
                "t3_ns": t3_ns,
            }

            sock.sendto(
                json.dumps(follow_up).encode("utf-8"),
                peer,
            )

            read_tx_timestamp_ns(sock)

            print(f"Handled sequence = {sequence}, \
                kernel t2={t2_ns}, \
                kernel t3={t3_ns}")


    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        sock.close()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")

    port = 31990
    if "cse-ai-6" in hostname:
        host = "10.67.91.123"
    elif "cse-ai-9" in hostname:
        host = "10.67.93.244"

    print(f"Host: {host}, Port: {port}")

    start_server(host, port)



    
