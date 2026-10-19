"""Clearance-sweep poses and motion-study frames, generated from the mechanism model so they
can never drift from mechanism.json.

Run:  python make_poses.py   -> poses.json, motion_frames.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)
import mechanism as m                                                   # noqa: E402

M = json.load(open(os.path.join(OUT, "mechanism.json"), encoding="utf-8"))
TR, SW = M["theta_rest"], M["theta_sweep"]
P_PARK, P_REST, P_DES, P_STOP = M["phi_park"], M["phi_rest_engaged"], M["phi_design"], M["phi_stop"]
T_DES, T_STOP = M["theta_design"], M["theta_stop"]
T_OVR = TR + 20.0                     # override: driver pushes the pedal 20 deg from rest
P_BAR = round(m.DESIGN_PT["p_f_bar"])
GAP = M["min_manual_gap_mm"]


def engaged_phi(th):
    return m.crank_for_engaged(th)[0]


def poses():
    out = []
    for k in range(int(SW) + 1):
        th = TR + k
        out.append([f"manual: parked, pedal {th:+.0f}", th, P_PARK])
    for ph in np.linspace(P_PARK, P_REST, 5):
        out.append([f"dv_takeup: crank {ph:.1f}", TR, float(ph)])
    for th in np.linspace(TR, T_DES, 12):
        out.append([f"dv_apply: pedal {th:.2f}", float(th), engaged_phi(th)])
    for th in np.linspace(T_DES, TR + SW, 12):
        out.append([f"override: pedal {th:.2f}", float(th), P_DES])
    for ph in np.linspace(P_DES, P_PARK, 6):
        out.append([f"backoff: crank {ph:.1f}", T_OVR, float(ph)])
    out.append(["hard_stop", T_STOP, P_STOP])                           # lug on the toggle-side stop
    out.append(["park_stop", TR, P_PARK + 3.0])                         # lug on the park/homing stop
    return out


def ease(a, b, n):
    return [a + (b - a) * (1 - math.cos(math.pi * i / (n - 1))) / 2 for i in range(n)]


def frames():
    f = []
    cap = "MANUAL MODE - actuator parked, pedal completely free"
    f += [[th, P_PARK, cap] for th in ease(TR, TR + SW, 14)]
    f += [[th, P_PARK, cap] for th in ease(TR + SW, TR, 10)]
    cap = f"DRIVERLESS - crank takes up the {GAP:.1f} mm gap"
    f += [[TR, ph, cap] for ph in ease(P_PARK, P_REST, 6)]
    cap = f"DRIVERLESS - crank pulls the link, pedal applied to {P_BAR} bar"
    f += [[th, engaged_phi(th), cap] for th in ease(TR, T_DES, 14)]
    f += [[T_DES, P_DES, cap]] * 3
    cap = "OVERRIDE - driver brakes harder, crank pin slides in the slot"
    f += [[th, P_DES, cap] for th in ease(T_DES, T_OVR, 10)]
    cap = "OVERRIDE detected - actuator backs off to park"
    f += [[T_OVR, ph, cap] for ph in ease(P_DES, P_PARK, 8)]
    cap = "Driver releases - pedal returns, actuator stays parked"
    f += [[th, P_PARK, cap] for th in ease(T_OVR, TR, 8)]
    return f


if __name__ == "__main__":
    p, fr = poses(), frames()
    json.dump(p, open(os.path.join(OUT, "poses.json"), "w", encoding="utf-8"), indent=1)
    json.dump(fr, open(os.path.join(OUT, "motion_frames.json"), "w", encoding="utf-8"), indent=1)
    print(len(p), "poses,", len(fr), "frames; park gap", GAP, "mm; design", P_BAR, "bar")
