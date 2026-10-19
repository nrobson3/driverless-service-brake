"""Reference packer/parser for the service-brake CAN frames + a self-check of the .dbc.

The DBC is parsed with a few lines of regex (no third-party CAN library needed), every
message is checked for overlapping or out-of-frame signals, and a command frame is
round-tripped through pack -> CRC -> unpack.

Run:  python can_codec.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DBC = os.path.join(HERE, "can", "wfr_asb_service_brake.dbc")

SG = re.compile(r"SG_ (\w+) : (\d+)\|(\d+)@1([+-]) \(([-\d.]+),([-\d.]+)\) \[([-\d.]+)\|([-\d.]+)\]")
BO = re.compile(r"BO_ (\d+) (\w+): (\d+) (\w+)")


def parse(path=DBC):
    msgs, cur = {}, None
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        m = BO.match(line.strip())
        if m:
            cur = dict(id=int(m.group(1)), name=m.group(2), dlc=int(m.group(3)), sender=m.group(4), sig={})
            msgs[cur["name"]] = cur
            continue
        s = SG.search(line)
        if s and cur:
            cur["sig"][s.group(1)] = dict(start=int(s.group(2)), size=int(s.group(3)), signed=s.group(4) == "-",
                                          scale=float(s.group(5)), offset=float(s.group(6)),
                                          lo=float(s.group(7)), hi=float(s.group(8)))
    return msgs


def check(msgs):
    problems = []
    for m in msgs.values():
        used = {}
        for n, s in m["sig"].items():
            for b in range(s["start"], s["start"] + s["size"]):
                if b >= 8 * m["dlc"]:
                    problems.append(f"{m['name']}.{n} bit {b} outside the frame")
                if b in used:
                    problems.append(f"{m['name']}.{n} overlaps {used[b]} at bit {b}")
                used[b] = n
            raw_max = (2 ** (s["size"] - (1 if s["signed"] else 0))) - 1
            raw_min = -(2 ** (s["size"] - 1)) if s["signed"] else 0
            if s["hi"] > raw_max * s["scale"] + s["offset"] + 1e-9:
                problems.append(f"{m['name']}.{n} range {s['hi']} exceeds raw capacity")
            if s["lo"] < raw_min * s["scale"] + s["offset"] - 1e-9:
                problems.append(f"{m['name']}.{n} range {s['lo']} is below raw capacity")
    return problems


def crc8_j1850(data):
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ 0xFF


def pack(msg, values):
    """Frame bytes from physical values. A value outside the signal's DBC range
    raises ValueError: masking it into the field would send a valid-looking but
    wrong frame (a -1 bar request would read as 654 bar)."""
    word = 0
    for n, val in values.items():
        s = msg["sig"][n]
        if not s["lo"] - 1e-9 <= val <= s["hi"] + 1e-9:
            raise ValueError(f"{msg['name']}.{n} = {val} is outside its range [{s['lo']}, {s['hi']}]")
        raw = int(round((val - s["offset"]) / s["scale"]))
        raw &= (1 << s["size"]) - 1
        word |= raw << s["start"]
    data = bytearray(word.to_bytes(msg["dlc"], "little"))
    if "SB_CRC8" in msg["sig"]:
        data[7] = crc8_j1850(data[:7])
    return bytes(data)


def unpack(msg, data):
    """Physical values from frame bytes; a frame shorter than the DLC raises ValueError."""
    if len(data) < msg["dlc"]:
        raise ValueError(f"{msg['name']} needs {msg['dlc']} bytes, got {len(data)}")
    word = int.from_bytes(data[:msg["dlc"]], "little")
    out = {}
    for n, s in msg["sig"].items():
        raw = (word >> s["start"]) & ((1 << s["size"]) - 1)
        if s["signed"] and raw & (1 << (s["size"] - 1)):
            raw -= 1 << s["size"]
        out[n] = raw * s["scale"] + s["offset"]
    return out


if __name__ == "__main__":
    msgs = parse()
    probs = check(msgs)
    print("messages:", {m["name"]: hex(m["id"]) for m in msgs.values()})
    print("layout problems:", probs or "none")
    cmd = msgs["ASB_SB_Cmd"]
    frame = pack(cmd, {"SB_PressureReq": 42.37, "SB_ModeReq": 2, "SB_InitCheckReq": 0,
                       "SB_RampLimit": 600, "SB_AliveCounter": 7})
    back = unpack(cmd, frame)
    print("cmd frame:", frame.hex(" "), "->", {k: round(v, 2) for k, v in back.items()})
    print("CRC ok:", crc8_j1850(frame[:7]) == frame[7])
