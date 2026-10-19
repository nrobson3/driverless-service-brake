"""Closed-loop simulation of the service-brake actuator (concept D).

Plant (2 DOF): crank angle phi (motor output) and pedal angle theta, coupled by the
slotted link as a ONE-WAY spring (tension only, lost motion otherwise). Pedal sees the
hydraulic reaction p(theta), the return spring, damping and an optional driver torque.
Controller (1 kHz, the ASB node): pressure feed-forward from the static model + PI on the
measured front pressure, torque limit, soft angle limit, and the state machine from the
CAN spec (PARKED / READY / ACTIVE / OVERRIDE / SAFE_STOP / FAULT).

Run:  python control_sim.py   -> control_*.png, control_results.json
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
import mechanism as mech                                  # noqa: E402

M = json.load(open(os.path.join(OUT, "mechanism.json")))
D2R = math.pi / 180.0
L = M["link"] / 1000.0
TH_REST, PHI_PARK, PHI_STOP = M["theta_rest"] * D2R, M["phi_park"] * D2R, M["phi_stop"] * D2R

# ---------------------------------------------------------------- plant parameters (assumed)
J_CRANK = 6.5e-3        # kg m^2 at the output: AK80-9 rotor reflected through 9:1 (estimate) + crank
B_CRANK = 0.02          # N m s/rad viscous
TC_CRANK = 0.15         # N m Coulomb (gearbox)
J_PEDAL = 4.5e-3        # kg m^2 about the pivot
C_PEDAL = 0.6           # N m s/rad (pads/fluid)
K_LINK = 4.0e6          # N/m tension stiffness of link + pins + bushes
C_LINK = 400.0          # N s/m
K_STOP = 2.0e4          # N m/rad hard stops
T_MAX = 18.0            # N m AK80-9 peak
T_LIMIT = 14.0          # N m firmware limit (just above the 13.0 N m design need)
TAU_I = 0.001           # s current-loop lag
P_NOISE = 0.3           # bar rms sensor noise (the PSS-140 accuracy band, +/-1 % FS = 1.4 bar, is separate)
TAU_SENSOR = 0.002      # s
DT = 2.0e-5
CTRL_DT = 1.0e-3


def pin_xy(th):
    return np.array([M["r_pin"] * math.sin(th), M["r_pin"] * math.cos(th)]) / 1000.0


def crank_xy(ph):
    return (np.array(M["motor_xz"]) + M["r_crank"] * np.array([math.cos(ph), math.sin(ph)])) / 1000.0


def gap(ph, th):
    """Link stretch g = |K - C| - L (m); g > 0 means the link is pulling."""
    return float(np.linalg.norm(crank_xy(ph) - pin_xy(th))) - L


def hyd_torque(th):
    tau, p = mech.pedal_torque_Nm(th / D2R, margin=False)
    spring = 40.0 * 0.225
    return tau - spring, p, spring


# feed-forward table: crank torque that holds pressure p (static model, no margin)
FF_P = np.linspace(0, 75, 151)
FF_T, FF_PHI = [], []
for p in FF_P:
    th = mech.theta_for_pressure(max(p, 0.01))
    ph, _ = mech.crank_for_engaged(th)
    tau_p, _ = mech.pedal_torque_Nm(th, margin=False)
    _, tm, _ = mech.forces(th, ph, tau_p)
    FF_T.append(tm)
    FF_PHI.append(ph * D2R)
FF_T, FF_PHI = np.array(FF_T), np.array(FF_PHI)
PHI_READY = mech.crank_for_engaged(M["theta_rest"])[0] * D2R + 1.5 * D2R    # ~0.5 mm short of contact


def p_from_phi(phi):
    """Pressure the static model predicts if the link is engaged at crank angle phi."""
    if phi >= FF_PHI[0]:
        return 0.0
    return float(np.interp(-phi, -FF_PHI, FF_P))


class Controller:
    """Cascade: pressure reference -> crank-angle reference (static model, plus a slow
    pressure-feedback trim) -> PD position loop with torque feed-forward.
    Override = measured pressure higher than the crank angle can explain (the driver
    has pushed the pedal ahead of the link) while also above the command."""

    def __init__(self, kp_pos=40.0, kd_pos=0.8, ki_trim=60.0):
        self.kp, self.kd, self.ki = kp_pos, kd_pos, ki_trim
        self.state = "READY"
        self.trim = 0.0
        self.p_ref = 0.0
        self.over_t = 0.0
        self.last_cmd_t = 0.0

    def _track(self, p_target, p_meas, phi, phidot):
        self.p_ref += float(np.clip(p_target - self.p_ref, -600 * CTRL_DT, 600 * CTRL_DT))   # 600 bar/s ramp
        if self.p_ref > 1.0:
            self.trim = float(np.clip(self.trim + self.ki * (self.p_ref - p_meas) * CTRL_DT / 100.0, -10, 10))
        pr = max(0.0, self.p_ref + self.trim)
        phi_ref = float(np.interp(pr, FF_P, FF_PHI)) if pr > 0.05 else PHI_READY
        phi_ref = max(phi_ref, PHI_STOP + 4 * D2R)                # soft limit before the hard stop
        ff = float(np.interp(pr, FF_P, FF_T))
        u = ff + self.kp * (phi - phi_ref) + self.kd * phidot     # brake-positive (phi decreases to brake)
        return float(np.clip(u, -T_LIMIT, T_LIMIT))

    def step(self, t, p_meas, phi, phidot, cmd, cmd_valid, powered):
        if not powered:
            self.state = "FAULT"
            return 0.0
        if cmd_valid:
            self.last_cmd_t = t
        age = t - self.last_cmd_t
        if self.state in ("READY", "ACTIVE") and age > 0.100:
            self.state = "SAFE_STOP"
        if self.state == "READY" and cmd > 0.5:
            self.state = "ACTIVE"
        if self.state == "ACTIVE" and cmd <= 0.5 and self.p_ref < 0.5:
            self.state = "READY"
            self.trim = 0.0                                       # no stale trim into the next apply
        if self.state == "OVERRIDE" and cmd < 0.5:                # re-armed only by a zero command
            self.state = "READY"
            self.trim = 0.0
        if self.state == "ACTIVE":
            ahead = p_meas - p_from_phi(phi)
            self.over_t = self.over_t + CTRL_DT if (ahead > 5.0 and p_meas > cmd + 2.0) else 0.0
            if self.over_t > 0.020:
                self.state, self.p_ref, self.trim = "OVERRIDE", 0.0, 0.0
        if self.state == "ACTIVE":
            return self._track(cmd, p_meas, phi, phidot)
        if self.state == "SAFE_STOP":
            return self._track(35.0, p_meas, phi, phidot)
        # READY holds ~0.5 mm short of contact; OVERRIDE backs the crank off to PARK
        goal = PHI_READY if self.state == "READY" else PHI_PARK
        u = self.kp * 0.3 * (phi - goal) + self.kd * 0.5 * phidot
        return float(np.clip(u, -4.0, 4.0))


def simulate(t_end, cmd_fn, driver_fn=lambda t: 0.0, valid_fn=lambda t: True, power_fn=lambda t: True, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = Controller()
    ph, phd = PHI_READY, 0.0
    th, thd = TH_REST, 0.0
    tq = 0.0
    p_filt = 0.0
    u = 0.0
    log = {k: [] for k in ("t", "cmd", "p", "p_meas", "torque", "phi", "theta", "tension", "state")}
    n = int(t_end / DT)
    next_ctrl = 0.0
    for k in range(n):
        t = k * DT
        tau_h, p, spring = hyd_torque(th)
        g = gap(ph, th)
        if g > 0:
            eps = 1e-7
            dg_dph = (gap(ph + eps, th) - g) / eps
            dg_dth = (gap(ph, th + eps) - g) / eps
            gdot = dg_dph * phd + dg_dth * thd
            T = max(0.0, K_LINK * g + C_LINK * gdot)
        else:
            dg_dph = dg_dth = 0.0
            T = 0.0
        if t >= next_ctrl:
            p_meas = p_filt + rng.normal(0, P_NOISE)
            u = ctrl.step(t, p_meas, ph, phd, cmd_fn(t), valid_fn(t), power_fn(t))
            next_ctrl += CTRL_DT
            log["t"].append(t); log["cmd"].append(cmd_fn(t)); log["p"].append(p); log["p_meas"].append(p_meas)
            log["torque"].append(tq); log["phi"].append(ph / D2R); log["theta"].append(th / D2R - M["theta_rest"])
            log["tension"].append(T); log["state"].append(ctrl.state)
        powered = power_fn(t)
        tq += (((u if powered else 0.0)) - tq) * DT / TAU_I
        # crank: brake torque u rotates phi DOWN (toward toggle)
        stop = 0.0
        if ph < PHI_STOP:
            stop = K_STOP * (PHI_STOP - ph)
        if ph > PHI_PARK + 3 * D2R:
            stop = -K_STOP * (ph - PHI_PARK - 3 * D2R)
        fric = -TC_CRANK * math.tanh(phd / 0.05)
        phdd = (-tq - B_CRANK * phd + fric + stop - T * dg_dph) / J_CRANK
        # pedal: driver + link pull forward; hydraulics + return spring back; rest stop
        pstop = (K_STOP * (TH_REST - th) - 8.0 * thd) if th < TH_REST else 0.0   # damped rest stop
        thdd = (driver_fn(t) - tau_h - spring - C_PEDAL * thd - T * dg_dth + pstop) / J_PEDAL
        phd += phdd * DT
        ph += phd * DT
        thd += thdd * DT
        th += thd * DT
        p_filt += (p - p_filt) * DT / TAU_SENSOR
    return {k: np.array(v) if k != "state" else v for k, v in log.items()}


def metrics_step(log, t0, target):
    t, p = log["t"], log["p"]
    after = t >= t0
    tt, pp = t[after], p[after]
    if not (pp >= 0.9 * target).any():
        raise RuntimeError("pressure never reached 90 % of the target")
    t10 = tt[np.argmax(pp >= 0.1 * target)]
    t90 = tt[np.argmax(pp >= 0.9 * target)]
    settled = pp[tt > tt[0] + 0.4]
    return dict(rise_10_90_ms=round((t90 - t10) * 1000, 1), delay_to_10pct_ms=round((t10 - t0) * 1000, 1),
                overshoot_pct=round(100 * (pp.max() - target) / target, 1),
                steady_error_bar=round(float(np.mean(settled) - target), 2),
                peak_torque_Nm=round(float(np.max(np.abs(log["torque"]))), 2))


def plot(log, title, path, extra=None):
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    ax[0].plot(log["t"], log["cmd"], "k--", lw=1, label="CAN command")
    ax[0].plot(log["t"], log["p"], lw=1.6, label="front line pressure")
    ax[0].set_ylabel("bar")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=.3)
    ax[0].set_title(title, fontsize=10)
    ax[1].plot(log["t"], log["torque"], color="tab:red", lw=1.2, label="motor torque (brake +)")
    ax[1].plot(log["t"], log["tension"] / 100.0, color="tab:green", lw=1.0, label="link tension / 100 [N]")
    ax[1].set_ylabel("N m")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=.3)
    states = ["READY", "ACTIVE", "OVERRIDE", "SAFE_STOP", "FAULT"]
    ax[2].plot(log["t"], [states.index(s) for s in log["state"]], drawstyle="steps-post", color="tab:purple")
    ax[2].set_yticks(range(len(states)))
    ax[2].set_yticklabels(states, fontsize=8)
    ax[2].set_xlabel("time [s]")
    ax[2].grid(alpha=.3)
    if extra:
        for x, txt in extra:
            ax[0].axvline(x, color="grey", lw=0.8, ls=":")
            ax[0].annotate(txt, (x, ax[0].get_ylim()[1] * 0.92), fontsize=8, rotation=0, xytext=(4, 0),
                           textcoords="offset points")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    res = {}
    # 1) step to 40 bar from READY
    log = simulate(0.8, lambda t: 40.0 if t >= 0.1 else 0.0)
    res["step_40bar"] = metrics_step(log, 0.1, 40.0)
    plot(log, "Step 0 -> 40 bar from READY", os.path.join(OUT, "control_step.png"))

    # 2) design-point step to 59 bar
    log = simulate(0.8, lambda t: 59.0 if t >= 0.1 else 0.0)
    res["step_59bar"] = metrics_step(log, 0.1, 59.0)

    # 3) driver override: command 30 bar, driver brakes to ~50 bar at 0.5 s, releases at 1.1 s, command off at 1.4 s
    th50 = mech.theta_for_pressure(50.0) * D2R
    tau_drv = hyd_torque(th50)[0] + hyd_torque(th50)[2]

    def driver(t):
        if 0.5 <= t < 1.1:
            return tau_drv * min(1.0, (t - 0.5) / 0.08)
        return 0.0
    cmd3 = lambda t: 30.0 if 0.1 <= t < 1.4 else 0.0
    log = simulate(1.7, cmd3, driver_fn=driver)
    st = log["state"]
    t_det = next((log["t"][i] for i in range(len(st)) if st[i] == "OVERRIDE"), None)
    res["override"] = dict(driver_applies_s=0.5, override_detected_s=None if t_det is None else round(float(t_det), 3),
                           max_pressure_bar=round(float(log["p"].max()), 1),
                           link_tension_after_detection_N=round(float(log["tension"][log["t"] > (t_det or 0) + 0.05].max()), 1),
                           pressure_after_release_bar=round(float(log["p"][(log["t"] > 1.3) & (log["t"] < 1.39)].mean()), 2))
    plot(log, "Driver brakes harder than the 30 bar command (lost motion + back-off)",
         os.path.join(OUT, "control_override.png"), [(0.5, "driver presses"), (1.1, "driver releases"), (1.4, "command 0")])

    # 4) CAN command stops at 0.5 s while braking at 25 bar
    log = simulate(1.2, lambda t: 25.0 if t >= 0.1 else 0.0, valid_fn=lambda t: t < 0.5)
    st = log["state"]
    t_ss = next((log["t"][i] for i in range(len(st)) if st[i] == "SAFE_STOP"), None)
    res["command_timeout"] = dict(last_command_s=0.5, safe_stop_entered_s=None if t_ss is None else round(float(t_ss), 3),
                                  pressure_end_bar=round(float(log["p"][-50:].mean()), 1))
    plot(log, "CAN command lost at 0.5 s: hold 100 ms, then ramp to the 35 bar safe-stop pressure",
         os.path.join(OUT, "control_timeout.png"), [(0.5, "last valid command")])

    # 5) actuator power lost at 0.5 s while at 40 bar -> must release (EBS takes over)
    log = simulate(1.0, lambda t: 40.0 if t >= 0.1 else 0.0, power_fn=lambda t: t < 0.5)
    p = log["p"]
    t_rel = next((log["t"][i] for i in range(len(p)) if log["t"][i] > 0.5 and p[i] < 2.0), None)
    res["power_loss"] = dict(power_off_s=0.5, pressure_below_2bar_s=None if t_rel is None else round(float(t_rel), 3))
    plot(log, "Actuator power lost at 0.5 s: back-drivable drive releases the pedal",
         os.path.join(OUT, "control_powerloss.png"), [(0.5, "power off")])

    with open(os.path.join(OUT, "control_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
