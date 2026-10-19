"""First-cut sizing: decel -> line pressure -> master-cylinder force -> pedal torque.

Run:  python sizing.py      (writes sizing_results.md and sizing_curves.png to out/)

Hydraulic load only: mechanism.py adds the pedal return spring (about 9 N m at the
pivot) when it sizes the link and motor.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import BRAKES, CAR, DESIGN, G, PEDAL, v

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)


def area(d):
    return math.pi * d * d / 4.0


def brake_chain(decel_g, pad_mu=None):
    """Ideal-balance brake demand at a given deceleration (both axles at the limit together)."""
    m, h, L = v(CAR, "mass_kg"), v(CAR, "cg_height_m"), v(CAR, "wheelbase_m")
    wf, R = v(CAR, "front_static_frac"), v(CAR, "tyre_radius_m")
    mu = v(BRAKES, "pad_mu") if pad_mu is None else pad_mu
    W = m * G
    dW = m * decel_g * G * h / L                     # longitudinal load transfer
    Fz_f, Fz_r = wf * W + dW, (1 - wf) * W - dW
    Fx_f, Fx_r = decel_g * Fz_f, decel_g * Fz_r      # braking force per axle
    T_f, T_r = Fx_f * R / 2, Fx_r * R / 2            # torque per wheel
    clamp_f = T_f / (2 * mu * v(BRAKES, "disc_reff_front_m"))
    clamp_r = T_r / (2 * mu * v(BRAKES, "disc_reff_rear_m"))
    A_cf = v(BRAKES, "caliper_front_pistons_per_side") * area(v(BRAKES, "caliper_front_piston_d_m"))
    A_cr = v(BRAKES, "caliper_rear_pistons_per_side") * area(v(BRAKES, "caliper_rear_piston_d_m"))
    p_f, p_r = clamp_f / A_cf, clamp_r / A_cr        # Pa
    F_mc = p_f * area(v(BRAKES, "mc_bore_front_m")) + p_r * area(v(BRAKES, "mc_bore_rear_m"))
    T_pivot = F_mc * v(PEDAL, "balance_bar_radius_m")
    F_pad = T_pivot / v(PEDAL, "pad_radius_m")
    bias_front = p_f * area(v(BRAKES, "mc_bore_front_m")) / F_mc if F_mc > 0 else float("nan")
    return dict(decel_g=decel_g, Fz_f=Fz_f, Fz_r=Fz_r, p_f_bar=p_f / 1e5, p_r_bar=p_r / 1e5,
                F_mc=F_mc, T_pivot=T_pivot, F_pad=F_pad, bias_front=bias_front)


def pedal_rotation_deg(p_bar):
    """Pedal rotation needed to build pressure p (front circuit governs)."""
    s_mc = v(PEDAL, "mc_dead_stroke_mm") + v(PEDAL, "mc_stroke_per_bar_mm") * p_bar
    return math.degrees(s_mc / 1000.0 / v(PEDAL, "balance_bar_radius_m"))


def main():
    a_d = v(DESIGN, "design_decel_g")
    margin = v(DESIGN, "actuator_margin")
    d = brake_chain(a_d)
    warm = brake_chain(a_d, pad_mu=0.45)
    rot = pedal_rotation_deg(d["p_f_bar"])
    t_full = v(DESIGN, "rise_time_s") / 0.8            # 10-90 % covers ~80 % of the stroke
    rot_rate = math.radians(rot) / t_full

    lines = []
    lines.append("# First-cut sizing (assumed reference car)\n")
    lines.append("All inputs: `params.py`. Ideal brake balance assumed (both axles reach the limit together).\n")
    lines.append("| Quantity | Design point | Same, warm pads (mu 0.45) |")
    lines.append("|---|---|---|")
    lines.append(f"| Deceleration | {a_d:.2f} g | {a_d:.2f} g |")
    lines.append(f"| Front / rear axle load | {d['Fz_f']:.0f} / {d['Fz_r']:.0f} N | same |")
    lines.append(f"| Line pressure front / rear | {d['p_f_bar']:.1f} / {d['p_r_bar']:.1f} bar | "
                 f"{warm['p_f_bar']:.1f} / {warm['p_r_bar']:.1f} bar |")
    lines.append(f"| Front brake bias (MC force share) | {100*d['bias_front']:.0f} % | same |")
    lines.append(f"| Balance-bar force | {d['F_mc']:.0f} N | {warm['F_mc']:.0f} N |")
    lines.append(f"| Torque about pedal pivot | {d['T_pivot']:.1f} N m | {warm['T_pivot']:.1f} N m |")
    lines.append(f"| Equivalent foot force at pad | {d['F_pad']:.0f} N | {warm['F_pad']:.0f} N |")
    lines.append(f"| Pedal rotation to reach it | {rot:.1f} deg | |")
    lines.append("")
    spring = v(PEDAL, "return_spring_N") * v(PEDAL, "pad_radius_m")
    lines.append(f"**Actuator requirement at the pedal pivot (hydraulic only):** {d['T_pivot']*margin:.0f} N m peak "
                 f"(design point x {margin} margin); with the {spring:.0f} N m return spring it is "
                 f"{(d['T_pivot'] + spring)*margin:.0f} N m, which mechanism.py uses. Pedal rotation {rot:.1f} deg in about "
                 f"{t_full*1000:.0f} ms -> mean pedal speed {rot_rate:.2f} rad/s.\n")
    lines.append("Pick-up force on a pedal pin at radius r from the pivot (hydraulic only):\n")
    lines.append("| Pin radius | Force at design point | With margin |")
    lines.append("|---|---|---|")
    for r in (0.08, 0.10, 0.12, 0.15, 0.18):
        F = d["T_pivot"] / r
        lines.append(f"| {r*1000:.0f} mm | {F:.0f} N | {F*margin:.0f} N |")
    lines.append("")
    lines.append("Sensitivity (design pad mu):\n")
    lines.append("| Decel | Front bar | Rear bar | Pivot torque | Pad force |")
    lines.append("|---|---|---|---|---|")
    for a in (0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
        r_ = brake_chain(a)
        lines.append(f"| {a:.1f} g | {r_['p_f_bar']:.1f} | {r_['p_r_bar']:.1f} | "
                     f"{r_['T_pivot']:.1f} N m | {r_['F_pad']:.0f} N |")
    with open(os.path.join(OUT, "sizing_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    acc = [i / 50 for i in range(0, 91)]
    rows = [brake_chain(a) for a in acc]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(acc, [r_["p_f_bar"] for r_ in rows], label="front")
    ax[0].plot(acc, [r_["p_r_bar"] for r_ in rows], label="rear")
    ax[0].set_xlabel("deceleration [g]")
    ax[0].set_ylabel("line pressure [bar]")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(acc, [r_["T_pivot"] for r_ in rows], color="k")
    ax[1].axhline(d["T_pivot"] * margin, ls="--", color="tab:red",
                  label=f"actuator requirement {d['T_pivot']*margin:.0f} N m")
    ax[1].set_xlabel("deceleration [g]")
    ax[1].set_ylabel("torque about pedal pivot [N m]")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sizing_curves.png"), dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
