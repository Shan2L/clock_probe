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
            level == socket.SOL_SOCKET and
            message_type == SO_TIMESTAMPING
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


def measure_once(sock, sequence, host, port):

    sample_monotonic_ns = time.monotonic_ns()

    request = {
        "type": "request",
        "sequence": sequence,
    }

    request_data = json.dumps(request).encode("utf-8")

    sock.sendto(request_data, (host, port))
    t1_ns = read_tx_timestamp_ns(sock)

    data, response_ancdata, flags, peer = sock.recvmsg(
        2048,
        socket.CMSG_SPACE(TIMESTAMP_STRUCT.size),
    )

    response = json.loads(data.decode("utf-8"))

    if response["type"] != "response":
        raise RuntimeError("Unexpected response type")

    if response["sequence"] != sequence:
        raise RuntimeError("Unexpected sequence number")

    t4_ns = extract_timestamp_ns(response_ancdata)


    follow_up_data, follow_up_ancdata, flags, peer = sock.recvmsg(
        2048,
        socket.CMSG_SPACE(TIMESTAMP_STRUCT.size),
    )


    follow_up = json.loads(follow_up_data.decode("utf-8"))

    if follow_up["type"] != "follow_up":
        raise RuntimeError("Unexpected follow_up type")

    if follow_up["sequence"] != sequence:
        raise RuntimeError("Unexpected sequence number")

    t2_ns, t3_ns = follow_up["t2_ns"], follow_up["t3_ns"]

    offset_ns = ((t2_ns - t1_ns) + (t3_ns - t4_ns))/2
    rtt_ns = (t4_ns - t1_ns) - (t3_ns - t2_ns)

    return {
        "sequence": sequence,
        "monotonic_ns": sample_monotonic_ns,
        "t1_ns": t1_ns,
        "t2_ns": t2_ns,
        "t3_ns": t3_ns,
        "t4_ns": t4_ns,
        "offset_ns": offset_ns,
        "rtt_ns": rtt_ns,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--output", type=str, default="baseline.jsonl")


    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")

    port = 31990
    if "cse-ai-6" in hostname:
        host = "10.67.93.244"
    elif "cse-ai-9" in hostname:
        host = "10.67.91.123"

    print(f"Host: {host}, Port: {port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)

    timestamp_flags = (
        SOF_TIMESTAMPING_RX_SOFTWARE | 
        SOF_TIMESTAMPING_TX_SOFTWARE |
        SOF_TIMESTAMPING_SOFTWARE 
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        SO_TIMESTAMPING,
        timestamp_flags,
    )


    args = parser.parse_args()

    try:
        with open(args.output, 
                    "w", 
                    encoding="utf-8", 
                    buffering=1) as output_file:
            for sequence in range(1, args.sample_count+1):
                sample = measure_once(
                    sock, sequence, host, port
                )

                output_file.write(json.dumps(sample) + "\n")

                print(
                    f"sequence={sample['sequence']:03d} "
                    f"offset={sample['offset_ns'] / 1_000:.3f} us "
                    f"rtt={sample['rtt_ns'] / 1_000:.3f} us"
                )
                if sequence < args.sample_count:
                    time.sleep(0.1)

    finally:
        sock.close()