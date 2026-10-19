"""Concept D mechanism: motor crank + slotted pull-link on a pedal shoulder pin.

Single source of truth for the mechanism geometry. Writes mechanism.json (read by
the Fusion build script) and the kinematics plots used as motion evidence.

Frame (mm): origin on the pedal pivot axis, X forward, Y left/outboard (actuator
side), Z up. Pedal angle theta from vertical, + forward. Crank angle phi measured
in the X-Z plane from +X toward +Z.

Run:  python mechanism.py
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)
from params import BRAKES, DESIGN, PEDAL, v                     # noqa: E402
from sizing import area, brake_chain                            # noqa: E402

# ---------------------------------------------------------------- geometry (mm)
THETA_REST = -15.0          # pedal rest angle, deg from vertical (assumed reference pedal)
THETA_SWEEP = 25.0          # full manual sweep incl. over-travel to the BOTS trip (assumed)
R_PIN = 150.0               # pedal shoulder-pin radius from the pivot
MOTOR = (120.0, 130.0)      # crank axis (x, z)
R_CRANK = 20.0              # crank radius (crank-pin offset)
LINK = 146.5                # crank-pin centre to slot pull face (pin centre when engaged), rod end at
                            # mid-adjustment: the M10 thread gives +/-3 mm (12..18 mm engagement)
PARK_CLEAR = 6.5            # pin-to-pull-face gap when parked, pedal at rest: SET AT ASSEMBLY with the rod
                            # end against the real pedal rest (half-turn steps of 0.75 mm -> 6.5 +/- 0.4)
SLOT_END_CLEAR = 6.5        # gap kept at the near end of the slot at full manual sweep (F7: >= 5 mm)
ROD_END_ADJ = 3.0           # rod-end adjustment either side of nominal (mm)
FW_STOP_MARGIN = 4.0        # firmware soft limit: crank never commanded closer than stop + 4 deg
GAMMA_STOP = 35.0          # hard stop: crank never closer than this to toggle (deg). 35 not 20: halves the stall load (motor bearing s0 >= 2 at 18 N m)
MOTOR_PEAK_NM = 18.0        # actuator peak torque (AK80-9 class), used for the hard-stop load case

# soft-pedal sensitivity: +50 % hydraulic compliance
SOFT_FACTOR = 1.5

DESIGN_PT = brake_chain(v(DESIGN, "design_decel_g"))
BIAS_F = DESIGN_PT["bias_front"]
A_MF = area(v(BRAKES, "mc_bore_front_m"))
R_BB = v(PEDAL, "balance_bar_radius_m") * 1000.0
MARGIN = v(DESIGN, "actuator_margin")


def pin(theta_deg):
    t = math.radians(theta_deg)
    return np.array([R_PIN * math.sin(t), R_PIN * math.cos(t)])


def pressure_at(theta_deg, soft=1.0):
    s_mc = R_BB * math.radians(theta_deg - THETA_REST)
    return max(0.0, (s_mc - v(PEDAL, "mc_dead_stroke_mm")) / (v(PEDAL, "mc_stroke_per_bar_mm") * soft))


def theta_for_pressure(p_bar, soft=1.0):
    s_mc = v(PEDAL, "mc_dead_stroke_mm") + v(PEDAL, "mc_stroke_per_bar_mm") * soft * p_bar
    return THETA_REST + math.degrees(s_mc / R_BB)


def pedal_torque_Nm(theta_deg, soft=1.0, margin=True):
    p = pressure_at(theta_deg, soft)
    F_mc = p * 1e5 * A_MF / BIAS_F
    tau = F_mc * R_BB / 1000.0 + v(PEDAL, "return_spring_N") * v(PEDAL, "pad_radius_m")
    return tau * (MARGIN if margin else 1.0), p


def crank_for_engaged(theta_deg, link=LINK):
    """Crank angle (deg) and toggle margin gamma (deg) with the link engaged at pedal angle theta."""
    M = np.array(MOTOR)
    C = pin(theta_deg)
    d = M - C
    D = float(np.hypot(*d))
    q = (link * link - D * D - R_CRANK * R_CRANK) / (2 * R_CRANK)
    if abs(q) >= D:
        return None
    beta = math.atan2(d[1], d[0])
    gamma = math.acos(q / D)
    phi = beta + gamma
    return math.degrees(phi), math.degrees(gamma)


def crank_pin(phi_deg):
    p = math.radians(phi_deg)
    return np.array(MOTOR) + R_CRANK * np.array([math.cos(p), math.sin(p)])


def forces(theta_deg, phi_deg, tau_pedal_Nm):
    """Link force (N) and motor torque (N m) to hold pedal torque tau at (theta, phi)."""
    C = pin(theta_deg) / 1000.0
    K = crank_pin(phi_deg) / 1000.0
    M = np.array(MOTOR) / 1000.0
    u = (K - C) / np.linalg.norm(K - C)
    lever = -(C[0] * u[1] - C[1] * u[0])           # m, + means a pull applies the brake
    F = tau_pedal_Nm / lever
    r = K - M
    tau_m = abs(r[0] * u[1] - r[1] * u[0]) * F
    return F, tau_m, lever


def gamma_to_theta(gamma_target):
    """Pedal angle at which the engaged crank reaches toggle margin gamma_target (bisection)."""
    lo, hi = THETA_REST, THETA_REST + 40.0
    for _ in range(80):
        mid = (lo + hi) / 2
        r = crank_for_engaged(mid)
        if r is None or r[1] < gamma_target:
            hi = mid
        else:
            lo = mid
    return lo


def stop_case_for_link(link, phi_stop):
    """Hard-stop load case for a given (adjusted) link length: the stop is a fixed crank angle."""
    lo, hi = THETA_REST - 5.0, THETA_REST + 40.0
    for _ in range(100):
        mid = (lo + hi) / 2
        r = crank_for_engaged(mid, link)
        if r is None or r[0] < phi_stop:
            hi = mid
        else:
            lo = mid
    phi, gam = crank_for_engaged(lo, link)
    F1, tau1, lever = forces(lo, phi, 1.0)
    F = MOTOR_PEAK_NM / tau1 * F1
    return dict(theta=round(lo, 2), gamma=round(gam, 2), link_force_N=round(F, 0), pedal_torque_Nm=round(F * lever, 0))


def adjustment_study(phi_park, phi_stop, slot_len):
    """F7 tolerance: the two parked gaps share the slot length, so the rod end is set at assembly.
    Reports the gaps over the adjustment range and pedal-rest error, and the stop case at both ends."""
    M = np.array(MOTOR)
    K = crank_pin(phi_park)
    rows = []
    for d_rest in (-1.0, 0.0, 1.0):                      # pedal rest angle error (deg)
        C0, C1 = pin(THETA_REST + d_rest), pin(THETA_REST + d_rest + THETA_SWEEP)
        # rod end set so the park gap is PARK_CLEAR at this real rest angle (to the nearest half turn)
        L_ideal = PARK_CLEAR + float(np.linalg.norm(K - C0))
        L_set = LINK + round((L_ideal - LINK) / 0.75) * 0.75
        gap = L_set - float(np.linalg.norm(K - C0))
        near = float(np.linalg.norm(K - C1)) - (L_set - slot_len)
        rows.append(dict(pedal_rest_error_deg=d_rest, link_set_mm=round(L_set, 2), park_gap_mm=round(gap, 2),
                         slot_end_gap_mm=round(near, 2), adjustment_used_mm=round(L_set - LINK, 2)))
    return dict(adjustment=rows, rod_end_adjust_mm=ROD_END_ADJ,
                stop_case_link_short=stop_case_for_link(LINK - ROD_END_ADJ, phi_stop),
                stop_case_link_long=stop_case_for_link(LINK + ROD_END_ADJ, phi_stop))


def main():
    out = {}
    th_design = theta_for_pressure(DESIGN_PT["p_f_bar"])
    th_soft = theta_for_pressure(DESIGN_PT["p_f_bar"], SOFT_FACTOR)
    th_stop = gamma_to_theta(GAMMA_STOP)

    # engaged range: rest -> design point
    rows = []
    for th in np.linspace(THETA_REST, th_design, 81):
        phi, gam = crank_for_engaged(th)
        tau_p, p = pedal_torque_Nm(th)
        F, tau_m, lever = forces(th, phi, tau_p)
        rows.append((th - THETA_REST, p, tau_p, F, tau_m, phi, gam, lever))
    rows = np.array(rows)

    # soft-pedal case: same pressure needs more travel
    soft_rows = []
    for th in np.linspace(THETA_REST, min(th_soft, th_stop), 81):
        phi, gam = crank_for_engaged(th)
        tau_p, p = pedal_torque_Nm(th, SOFT_FACTOR)
        F, tau_m, lever = forces(th, phi, tau_p)
        soft_rows.append((th - THETA_REST, p, tau_p, F, tau_m, phi, gam))
    soft_rows = np.array(soft_rows)
    p_soft_at_stop = pressure_at(th_stop, SOFT_FACTOR)                  # reach at the mechanical stop
    # reach within the firmware soft limit (crank angle >= phi_stop + FW_STOP_MARGIN)
    lo, hi = THETA_REST, th_stop
    phi_lim = crank_for_engaged(th_stop)[0] + FW_STOP_MARGIN
    for _ in range(80):
        mid = (lo + hi) / 2
        if crank_for_engaged(mid)[0] >= phi_lim:
            lo = mid
        else:
            hi = mid
    p_soft_fw = pressure_at(lo, SOFT_FACTOR)
    p_nom_fw = pressure_at(lo, 1.0)

    # hard-stop load case: motor at peak torque with the crank at the stop
    phi_stop, _ = crank_for_engaged(th_stop)
    F1, tau1, lever_stop = forces(th_stop, phi_stop, 1.0)        # per N m at the pedal
    F_stop = MOTOR_PEAK_NM / tau1 * F1                          # link force when motor = peak
    tau_pedal_stop = F_stop * lever_stop

    # parked position: rotate the crank back until the pull face is PARK_CLEAR behind the rest pin
    M = np.array(MOTOR)
    C_rest = pin(THETA_REST)
    # find phi_park (crank swung further from toggle) with |K - C_rest| = LINK - PARK_CLEAR
    target = LINK - PARK_CLEAR
    d = M - C_rest
    D = float(np.hypot(*d))
    q = (target * target - D * D - R_CRANK * R_CRANK) / (2 * R_CRANK)
    beta = math.atan2(d[1], d[0])
    phi_park = math.degrees(beta + math.acos(q / D))
    K_park = crank_pin(phi_park)
    C_full = pin(THETA_REST + THETA_SWEEP)
    park_gap_max = LINK - (float(np.linalg.norm(M - C_rest)) - R_CRANK)
    near_end = float(np.linalg.norm(K_park - C_full))           # pin centre to crank-pin centre at full sweep
    slot_len = math.ceil((LINK - near_end + SLOT_END_CLEAR) * 10.0) / 10.0   # slot travel, rounded UP
    # gap between parked link and pin over the full manual sweep (along the link)
    sweep_gap = []
    for th in np.linspace(THETA_REST, THETA_REST + THETA_SWEEP, 51):
        dist = float(np.linalg.norm(K_park - pin(th)))
        sweep_gap.append((th - THETA_REST, LINK - dist, dist - (LINK - slot_len)))
    sweep_gap = np.array(sweep_gap)

    out.update(
        frame="mm; origin pedal pivot; X fwd, Y outboard, Z up; angles deg",
        theta_rest=THETA_REST, theta_sweep=THETA_SWEEP, r_pin=R_PIN, motor_xz=list(MOTOR),
        r_crank=R_CRANK, link=LINK, slot_travel=round(slot_len, 1),
        phi_park=round(phi_park, 2), phi_rest_engaged=round(rows[0, 5], 2),
        phi_design=round(rows[-1, 5], 2), phi_stop=round(phi_stop, 2),
        theta_design=round(th_design, 2), theta_soft=round(th_soft, 2), theta_stop=round(th_stop, 2),
        gamma_design=round(rows[-1, 6], 1), gamma_stop=GAMMA_STOP,
        motor_torque_max_Nm=round(rows[:, 4].max(), 2), link_force_max_N=round(rows[:, 3].max(), 0),
        soft_pressure_at_mech_stop_bar=round(p_soft_at_stop, 1),
        soft_pressure_within_fw_limit_bar=round(p_soft_fw, 1),
        nominal_pressure_within_fw_limit_bar=round(p_nom_fw, 1),
        soft_motor_torque_max_Nm=round(soft_rows[:, 4].max(), 2),
        stop_link_force_N=round(F_stop, 0), stop_pedal_torque_Nm=round(tau_pedal_stop, 0),
        min_manual_gap_mm=round(sweep_gap[:, 1].min(), 2),
        park_gap_geometric_max_mm=round(park_gap_max, 2),
        min_slot_end_gap_mm=round(sweep_gap[:, 2].min(), 2),
        crank_sweep_rest_to_design=round(abs(rows[-1, 5] - rows[0, 5]), 1),
        crank_sweep_park_to_stop=round(abs(phi_stop - phi_park), 1),
    )
    out.update(adjustment_study(phi_park, phi_stop, slot_len))
    with open(os.path.join(OUT, "mechanism.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for k, val in out.items():
        print(f"{k}: {val}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(rows[:, 0], rows[:, 4], label="nominal pedal")
    ax[0].plot(soft_rows[:, 0], soft_rows[:, 4], "--", label="soft pedal (+50 % compliance)")
    ax[0].axhline(MOTOR_PEAK_NM, color="tab:red", lw=1, label="motor peak torque")
    ax[0].set_xlabel("pedal rotation from rest [deg]")
    ax[0].set_ylabel("motor torque incl. 1.25 margin [N m]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].plot(rows[:, 0], rows[:, 1], label="nominal")
    ax[1].plot(soft_rows[:, 0], soft_rows[:, 1], "--", label="soft pedal")
    ax[1].axvline(th_stop - THETA_REST, color="tab:red", lw=1, label="crank hard stop")
    ax[1].set_xlabel("pedal rotation from rest [deg]")
    ax[1].set_ylabel("front line pressure [bar]")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    ax[2].plot(sweep_gap[:, 0], sweep_gap[:, 1], label="pin to pull face (must stay > 0)")
    ax[2].plot(sweep_gap[:, 0], sweep_gap[:, 2], label="pin to slot near end (must stay > 0)")
    ax[2].set_xlabel("pedal rotation from rest, actuator PARKED [deg]")
    ax[2].set_ylabel("gap [mm]")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mechanism_kinematics.png"), dpi=150)


if __name__ == "__main__":
    main()
