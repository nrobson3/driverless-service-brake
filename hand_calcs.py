"""Hand calculations for the parts the 2D FE does not cover, plus cross-checks (rev C).

Load split at the crank (rev C: crank pin pressed into BOTH plates, so the crank is one rigid body
on two supports - the motor's output bearing and the 6201 in the bearing carrier):
  * How the link force splits between the two supports depends on their stiffness, so each support
    is checked against the share that is worst FOR IT:
      - 6201 / inner plate / inner wall: the pin is fixed in both plates, so the inner plate can take
        up to half of the RADIAL component; the TANGENTIAL part needs a torque about the pin axis that
        the R7 press fit can only give 1.3-5 N m (< 250 N at r 20), so it goes to the motor side.
        R_in = sqrt((F_r / 2)^2 + 250^2).
      - motor output bearing: the softest inner support (the slip-fit model of rev B: pin as a
        propped cantilever, inner plate takes 5/16 F_r) -> all F_t + 11/16 F_r.

Cases: worst = motor at its 18 N m peak against the toggle-side hard stop; the actuator's own
current limit (14 N m, set in the CubeMars driver); and the 1.6 g design point incl. the 1.25 margin.

Run:  python hand_calcs.py   -> hand_calcs.md, hand_calcs.json
"""
import json
import math
import os

import numpy as np

import mechanism as mech

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)
M = json.load(open(os.path.join(OUT, "mechanism.json")))
FE = json.load(open(os.path.join(OUT, "fea_summary.json")))

F_STOP = M["stop_link_force_N"]
F_DES = M["link_force_max_N"]
LIMIT = 14.0 / 18.0                       # actuator current limit / peak torque
SY = {"6061-T6": 276.0, "12.9 alloy steel": 1080.0, "8.8 steel": 640.0, "17-4PH H900": 1170.0,
      "17-4PH H1150": 724.0, "dowel": 1000.0}
C0_6201, C0_AK80 = 3100.0, 2810.0        # static radial ratings, N (SKF 6201; CubeMars AK80-9 listing)
T_PRESS_MAX = 5.0                         # N m: R7/h6 press fit of the 8 mm pin in 5 mm of 6061 (upper bound)
rows = []


def add(item, case, formula, value, unit, allow, allow_txt, need=2.0):
    value = float(value)
    fos = float(allow / value) if value > 0 else float("inf")
    rows.append(dict(item=item, case=case, formula=formula, value=round(value, 1), unit=unit,
                     allowable=allow_txt, FOS=round(fos, 2), need=need, ok=bool(fos >= need)))


def zc(d):                                   # section modulus of a round bar
    return math.pi * d ** 3 / 32.0


def split(theta, phi, F):
    """Radial / tangential components of the link force on the crank pin (relative to the crank arm)."""
    C = mech.pin(theta)
    K = mech.crank_pin(phi)
    Mx = np.array(mech.MOTOR)
    u = (C - K) / np.linalg.norm(C - K)                       # force on the pin, toward the pedal
    r = (K - Mx) / np.linalg.norm(K - Mx)                     # crank arm direction (outward)
    Fr = abs(F * (u @ r))
    Ft = abs(F * (u[0] * r[1] - u[1] * r[0]))
    return Fr, Ft


Fr_s, Ft_s = split(M["theta_stop"], M["phi_stop"], F_STOP)
Fr_d, Ft_d = split(M["theta_design"], M["phi_design"], F_DES)
motor_s = math.hypot(11 / 16 * Fr_s, Ft_s)
motor_d = math.hypot(11 / 16 * Fr_d, Ft_d)
t_cap = T_PRESS_MAX / 0.020
inner_s = math.hypot(Fr_s / 2, min(Ft_s, t_cap))
inner_d = math.hypot(Fr_d / 2, min(Ft_d, t_cap))

# stop lug / collar geometry (as wfr_cad.py rev C)
STOP_R, COLLAR_R, LUG_HW = 37.0, 6.0, 9.0
r_contact = math.sqrt(STOP_R ** 2 - (LUG_HW + COLLAR_R) ** 2)
F_lug = 18.0 / (r_contact / 1000.0)

# 1 crank pin: stepped 8 h6, 17-4PH H900. Conservative: propped cantilever from the outer plate
L, a = 15.0, 7.5
Mfix = math.hypot(3 / 16 * Fr_s * L, Ft_s * a)
add("Crank pin dia 8 h6 (17-4PH H900)", "hard stop", "M = sqrt((3/16 Fr L)^2 + (Ft a)^2) (propped-cantilever bound); sigma = M / Z",
    Mfix / zc(8), "MPa", SY["17-4PH H900"], "Sy = 1170 MPa")
add("Crank pin dia 8 h6 (17-4PH H900)", "hard stop", "tau = motor-side force / A", motor_s / (math.pi * 16), "MPa",
    0.577 * SY["17-4PH H900"], "0.577 Sy = 675 MPa")
add("Crank-pin bore, 6061 plate (bearing)", "hard stop", "sigma_br = motor-side force / (8 x 5)", motor_s / 40.0, "MPa",
    1.5 * SY["6061-T6"], "1.5 Sy bearing = 414 MPa")

# 2 pedal shoulder screw ISO 7379 dia 10 (M8 thread): cantilever 7 mm to the rod-end centre
add("Pedal shoulder screw dia 10 (ISO 7379, 12.9)", "hard stop", "sigma = F x 7 mm / Z", F_STOP * 7 / zc(10), "MPa",
    SY["12.9 alloy steel"], "Sy = 1080 MPa")

# 3 link: rails in tension, slot end in bearing, M10 thread in the boss (12 mm MINIMUM engagement:
#   the witness hole), against the stop force at the long end of the rod-end adjustment
F_adj = max(F_STOP, M["stop_case_link_long"]["link_force_N"], M["stop_case_link_short"]["link_force_N"])
add("Pull-link rails", "hard stop", "sigma = F / (2 x 6.75 x 8)", F_adj / (2 * 6.75 * 8), "MPa", SY["6061-T6"],
    "Sy = 276 MPa")
add("Pull-link slot end on the hardened sleeve (bearing)", "hard stop", "sigma_br = F / (d t), d = 12, t = 8",
    F_adj / (12 * 8), "MPa", 1.5 * SY["6061-T6"], "1.5 Sy bearing = 414 MPa")
As = math.pi * 10 * 12 * 0.58
add("Pull-link M10 internal thread, 12 mm min. engagement", "hard stop, adjustment extreme", "tau = F / (pi d Le 0.58)",
    F_adj / As, "MPa", 0.577 * SY["6061-T6"], "0.577 Sy = 159 MPa")

# 4 bearings
add("Outboard bearing 6201-2RS (C0 3.1 kN)", "hard stop", "R_in = sqrt((Fr/2)^2 + min(Ft, 250)^2)", inner_s, "N",
    C0_6201, "C0 = 3100 N")
add("Motor output bearing AK80-9 (C0r 2.81 kN)", "hard stop, 18 N m (no limit)", "sqrt((11/16 Fr)^2 + Ft^2)",
    motor_s, "N", C0_AK80, "C0r = 2810 N", need=2.0)
add("Motor output bearing AK80-9 (C0r 2.81 kN)", "hard stop, 14 N m driver limit", "same x 14/18",
    motor_s * LIMIT, "N", C0_AK80, "C0r = 2810 N", need=2.0)
add("Motor output bearing AK80-9 (C0r 2.81 kN)", "design point (1.6 g x 1.25)", "same", motor_d, "N", C0_AK80,
    "C0r = 2810 N")

# 5 flange joint: outer crank plate to motor flange, 6x M3 hex-head 12.9 on PCD 30 under a keyed lock ring. The pilot takes
#   the radial load; conservatively the screws are checked carrying it too.
Fb = motor_s / 6 + 18000.0 / (6 * 15.0)
add("Flange screws 6x M3 hex ISO 4017 (12.9), lock ring", "hard stop", "F/6 + T/(6 r), tau = Fb / 5.03 mm^2", Fb / 5.03, "MPa",
    0.577 * SY["12.9 alloy steel"], "0.577 Sy = 623 MPa")

# 6 stub shaft 12 mm (17-4PH H1150): inner-bearing load, 12.5 mm from the 6201 centre to the plate mid-plane
add("Stub shaft dia 12 (17-4PH H1150)", "hard stop", "sigma = R_in x 12.5 / Z", inner_s * 12.5 / zc(12), "MPa",
    SY["17-4PH H1150"], "Sy = 724 MPa")

# 7 yoke inner wall: in-plane cantilever, 60 x 10 section, inner-bearing load at 152 mm
add("Yoke inner wall (beam check of the FE)", "hard stop", "sigma = R_in 152 / (10 x 60^2 / 6)",
    inner_s * 152 / (10 * 60 ** 2 / 6), "MPa", SY["6061-T6"], "Sy = 276 MPa")
fe_wall = FE["yoke_inner_plate"]
fe_wall_scaled = fe_wall["peak_corrected_MPa"] * inner_s / fe_wall["bore_load_N"]
add("Yoke inner wall, FE peak scaled to R_in", "hard stop", f"FE {fe_wall['peak_corrected_MPa']:.1f} MPa at "
    f"{fe_wall['bore_load_N']:.0f} N x R_in / {fe_wall['bore_load_N']:.0f}", fe_wall_scaled, "MPa", SY["6061-T6"],
    "Sy = 276 MPa")

# 8 bearing carrier: 3x M4 8.8 clamp (friction) + 2x 4 mm dowels match-reamed after alignment
P_i = 0.75 * 600.0 * 8.78
add("Carrier clamp, 3x M4 8.8 (friction only, before dowelling)", "hard stop", "mu 0.15 x 3 x 0.75 x 600 x 8.78 / R_in",
    inner_s, "N", 0.15 * 3 * P_i, f"slip at {0.15 * 3 * P_i:.0f} N", need=1.0)
add("Carrier dowels 2x 4 m6 (shear)", "hard stop", "tau = R_in / (2 x pi 2^2)", inner_s / (2 * math.pi * 4), "MPa",
    0.577 * SY["dowel"], "577 MPa")
add("Carrier dowel holes, 5 mm flange (bearing)", "hard stop", "sigma_br = R_in / 2 / (4 x 5)", inner_s / 2 / 20.0,
    "MPa", 1.5 * SY["6061-T6"], "1.5 Sy bearing = 414 MPa")

# 9 yoke outer wall + gussets: motor-side load offset 42.5 mm from the wall -> torsion taken by the two
#   gussets as a couple (84 mm apart). Gusset root 45 x 8; also the section through the 16 mm hole.
T = motor_s * 42.5
Fg = T / 84.0
add("Yoke gusset 8 mm, root 45 mm (torsion couple)", "hard stop", "Fg = (F_motor x 42.5) / 84; sigma = Fg 152 / (8 x 45^2/6)",
    Fg * 152 / (8 * 45 ** 2 / 6), "MPa", SY["6061-T6"], "Sy = 276 MPa")
b_h = 45.0 * (1 - 18.5 / 122.0)
Z_h = 8 * (b_h ** 3 - 16.0 ** 3) / 12 / (b_h / 2)
add("Yoke gusset, section through the 16 mm lightening hole", "hard stop",
    f"width {b_h:.1f} - 16 hole; sigma = Fg 133.5 / Z_net", Fg * 133.5 / Z_h, "MPa", SY["6061-T6"], "Sy = 276 MPa")
add("Yoke outer wall in-plane bending (root, solid)", "hard stop", "sigma = F_motor 152 / (8 x 90^2 / 6)",
    motor_s * 152 / (8 * 90 ** 2 / 6), "MPa", SY["6061-T6"], "Sy = 276 MPa")
Z_w = 8 * (90 ** 3 - 36 ** 3) / 12 / 45.0
add("Yoke outer wall through the 36 mm window", "hard stop", "sigma = F_motor 132 / (8 (90^3 - 36^3) / 12 / 45)",
    motor_s * 132 / Z_w, "MPa", SY["6061-T6"], "Sy = 276 MPa")

# 10 hard stops (link slack: the whole 18 N m goes into the toggle-side stop)
E_star = 1 / ((1 - 0.3 ** 2) / 200000 + (1 - 0.33 ** 2) / 69000)
p0 = math.sqrt(F_lug * E_star / (math.pi * 4.5 * COLLAR_R))
add("Toggle stop: steel collar 12 on the 6061 lug (Hertz, 4.5 mm line)", "18 N m into the stop",
    f"F = 18 / {r_contact:.1f} mm = {F_lug:.0f} N; p0 = sqrt(F E* / (pi L R))", p0, "MPa",
    4 * SY["6061-T6"] / math.sqrt(3), "shakedown limit 4k = 637 MPa", need=1.0)
add("Toggle stop, same at the 14 N m driver limit", "14 N m into the stop", "p0 x sqrt(14/18)", p0 * math.sqrt(LIMIT),
    "MPa", 4 * SY["6061-T6"] / math.sqrt(3), "shakedown limit 4k = 637 MPa", need=1.0)
P_i6 = 0.75 * 600.0 * 20.1
add("Stop stud M6 8.8 + nyloc: collar clamp vs prying", "18 N m into the stop",
    "M = F x 7.5 mm vs preload x collar r 6 x 0.5", F_lug * 7.5, "N mm", P_i6 * 6.0 * 0.5, f"{P_i6 * 3:.0f} N mm")
add("Stop stud M6 8.8: shear by clamp friction", "18 N m into the stop", "F vs mu 0.15 x 0.75 x 600 x 20.1", F_lug, "N",
    0.15 * P_i6, f"slip at {0.15 * P_i6:.0f} N")
fe_lug = FE["crank_plate_inner_stop_lug"]
add("Inner crank plate stop lug (FE, R3 roots + notch fill)", "18 N m into the stop", "2D FE x 1.12 CST correction",
    fe_lug["peak_corrected_MPa"], "MPa", SY["6061-T6"], "Sy = 276 MPa")

# 11 pedal: bending moment in the arm at the balance bar, actuator worst case vs the 2 kN rule case
Mrule = 2000 * (225 - 45) / 1000.0
Mact = F_adj * (150 - 45) / 1000.0
add("Pedal-arm moment at the balance bar: actuator worst case vs 2 kN rule case", "hard stop, adjustment extreme",
    "M_act = F (150-45) vs M_rule = 2000 (225-45)", Mact, "N m", Mrule, f"rule case {Mrule:.0f} N m (ratio >= 1)", need=1.0)

# 12 motor torque budget: an everyday 0.8 g stop with the FIXED 61 % balance bar
from params import BRAKES, CAR, G, v                     # noqa: E402


def tyre_force_per_bar(axle):
    """Axle braking force per bar of line pressure (both wheels)."""
    d = v(BRAKES, f"caliper_{axle}_piston_d_m")
    n = v(BRAKES, f"caliper_{axle}_pistons_per_side")
    clamp = n * math.pi * d * d / 4 * 1e5
    return 2 * clamp * 2 * v(BRAKES, "pad_mu") * v(BRAKES, f"disc_reff_{axle}_m") / v(CAR, "tyre_radius_m")


b_f = mech.BIAS_F                                         # fixed balance-bar split (equal MC bores)
need = 0.8 * v(CAR, "mass_kg") * G
p_front_fixed = need / (tyre_force_per_bar("front") + tyre_force_per_bar("rear") * (1 - b_f) / b_f)
th08 = mech.theta_for_pressure(p_front_fixed)
phi08, _ = mech.crank_for_engaged(th08)
tau08, _ = mech.pedal_torque_Nm(th08, margin=False)
_, tm08, _ = mech.forces(th08, phi08, tau08)
motor = dict(peak_needed_incl_margin=M["motor_torque_max_Nm"], soft_pedal_max=M["soft_motor_torque_max_Nm"],
             everyday_0p8g=round(tm08, 2), everyday_0p8g_front_bar=round(p_front_fixed, 1),
             ak80_9_rated=9.0, ak80_9_peak=18.0, driver_limit=14.0)
loads = dict(stop_Fr=round(Fr_s), stop_Ft=round(Ft_s), stop_motor_side=round(motor_s), stop_inner=round(inner_s),
             design_motor_side=round(motor_d), design_inner=round(inner_d), stop_lug_N=round(F_lug),
             stop_contact_r_mm=round(r_contact, 1), link_force_adjust_extreme_N=round(F_adj))

out = {"rows": rows, "motor_Nm": motor, "crank_loads_N": loads}
with open(os.path.join(OUT, "hand_calcs.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

lines = ["# Hand calculations, rev C (worst case = motor at 18 N m peak against the toggle-side hard stop, "
         f"link force {F_STOP:.0f} N nominal, {F_adj:.0f} N at the rod-end adjustment extreme; design point {F_DES:.0f} N)\n",
         f"Crank-pin load split at the stop: F_r {Fr_s:.0f} N, F_t {Ft_s:.0f} N -> motor side {motor_s:.0f} N "
         f"(soft inner support), inner support {inner_s:.0f} N (stiff inner support, press-fit torque limit) "
         f"(design point: {motor_d:.0f} / {inner_d:.0f} N). Link slack: {F_lug:.0f} N on the toggle collar "
         f"at r {r_contact:.1f} mm.\n",
         "| Item | Case | Calculation | Value | Allowable | FOS |", "|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['item']} | {r['case']} | {r['formula']} | {r['value']} {r['unit']} | {r['allowable']} | "
                 f"{r['FOS']} (need {r['need']:g}){'' if r['ok'] else ' LOW'} |")
lines.append("")
lines.append(f"Motor torque: {motor['peak_needed_incl_margin']} N m at the 1.6 g design point incl. the 1.25 margin "
             f"(driver limit 14, AK80-9 peak 18 N m); {motor['everyday_0p8g']} N m for an everyday 0.8 g stop "
             f"({motor['everyday_0p8g_front_bar']} bar front with the fixed balance bar; AK80-9 rated 9 N m).")
with open(os.path.join(OUT, "hand_calcs.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
