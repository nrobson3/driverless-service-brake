"""Run the 2D FE checks: solver validation, then link / inner crank plate / inner yoke plate
at the worst load case (motor peak torque against the toggle-side hard stop), with a
mesh-convergence pair for each. Writes fea_*.png and fea_summary.json.

Run:  python fea_run.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import fe2d
from shapely.geometry import Point, Polygon, box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)
MECH = json.load(open(os.path.join(OUT, "mechanism.json")))
AL6061 = dict(E=68900.0, nu=0.33, Sy=276.0, name="6061-T6")      # MPa
F_STOP = MECH["stop_link_force_N"]          # motor peak torque at the toggle-side stop
F_DESIGN = MECH["link_force_max_N"]         # design point incl. 1.25 margin
LUG_HALF_W, LUG_LEN, STOP_R, SLEEVE_R = 9.0, 36.0, 37.0, 6.0   # as wfr_cad.py (rev C)
_DELTA = math.degrees(math.asin((LUG_HALF_W + SLEEVE_R) / STOP_R))   # stop lug half-width + collar radius
R_CONTACT = math.sqrt(STOP_R ** 2 - (LUG_HALF_W + SLEEVE_R) ** 2)    # contact point along the lug axis
F_LUG = 18.0 / (R_CONTACT / 1000.0)        # motor peak torque reacted entirely by the toggle stop (link slack)
MECH_LUG_ANGLE = 115.0 - _DELTA - (MECH["phi_park"] + 3.0)  # crank-local lug direction (as in wfr_cad.py)
MECH_TOGGLE_ANG = MECH["phi_stop"] + MECH_LUG_ANGLE - _DELTA


def link_dir_world():
    """Unit vector of the force the link applies to the crank pin at the hard stop (toward the pedal)."""
    t = math.radians(MECH["theta_stop"])
    C = np.array([MECH["r_pin"] * math.sin(t), MECH["r_pin"] * math.cos(t)])
    p = math.radians(MECH["phi_stop"])
    K = np.array(MECH["motor_xz"]) + MECH["r_crank"] * np.array([math.cos(p), math.sin(p)])
    d = C - K
    return d / np.linalg.norm(d)


def outline(poly):
    """Exterior ring of a shapely polygon as a CCW vertex list (no repeated last point).
    The mesher's inside tests use matplotlib Path radii, whose sign flips with ring
    orientation, so a clockwise ring would chew a saw-tooth into the boundary."""
    poly = orient(poly, sign=1.0)
    xs, ys = poly.exterior.coords.xy
    return list(zip(xs[:-1], ys[:-1]))


def rounded_end_bar(x0, x1, h, r):
    """Bar x0..x1, height +/-h, far-end corners rounded r (link side view)."""
    return unary_union([sbox(x0, -h, x1 - r, h), sbox(x1 - r, -h + r, x1, h - r),
                        Point(x1 - r, h - r).buffer(r, 64), Point(x1 - r, -h + r).buffer(r, 64)])


def crank_inner_outline(lug_deg, r_int=3.0):
    """Trimmed inner crank plate: hub r13 + web to the r9 pin boss at x 20 + 18-wide stop-lug arm,
    internal corners R3 (a morphological closing = the CAD's concave fillets)."""
    a = math.radians(lug_deg)
    ux, uy = math.cos(a), math.sin(a)
    hw, L = LUG_HALF_W, LUG_LEN
    arm = Polygon([(hw * uy, -hw * ux), (L * ux + hw * uy, L * uy - hw * ux),
                   (L * ux - hw * uy, L * uy + hw * ux), (-hw * uy, hw * ux)])
    # notch fill between the pin-boss web and the lug (as wfr_cad.lug_notch_fill)
    C = (22.0 * ux - hw * uy, 22.0 * uy + hw * ux)
    dx, dy = C[0] - 20.0, C[1]
    d_ = math.hypot(dx, dy)
    beta, alpha = math.atan2(dy, dx), math.acos(9.0 / d_)
    T = max(((20.0 + 9.0 * math.cos(beta + sg * alpha), 9.0 * math.sin(beta + sg * alpha)) for sg in (1, -1)),
            key=lambda q: q[0])
    fill = Polygon([(10.0, -5.0), T, C])
    shape = unary_union([Point(0, 0).buffer(13, 64), sbox(0, -9, 20, 9), Point(20, 0).buffer(9, 64),
                         arm, Point(L * ux, L * uy).buffer(hw, 64), fill])
    return shape.buffer(r_int, 64).buffer(-r_int, 64)


def boundary_patch_load(P, poly, center, half_len, force):
    """Uniform load on the outer-boundary nodes within half_len of `center` (a contact patch)."""
    ring = poly.exterior
    on = [i for i in range(len(P)) if math.hypot(P[i, 0] - center[0], P[i, 1] - center[1]) < half_len
          and ring.distance(Point(P[i, 0], P[i, 1])) < 1e-6]
    return {int(i): (force[0] / len(on), force[1] / len(on)) for i in on}


def merge_loads(*sets):
    out = {}
    for ld in sets:
        for k, (fx, fy) in ld.items():
            ox, oy = out.get(k, (0.0, 0.0))
            out[k] = (ox + fx, oy + fy)
    return out


def inner_wall_outline(mx, mz, top, bore_r, slot_w):
    """Inner yoke wall (world X-Z): 60 wide from z -22 to top, with the carrier bore and the
    open-top slot cut from the outline (they break the top edge, so they are boundary, not holes)."""
    wall = sbox(mx - 30, -22.0, mx + 30, top)
    notch = unary_union([Point(mx, mz).buffer(bore_r, 256), sbox(mx - slot_w / 2, mz, mx + slot_w / 2, top + 1),
                         sbox(mx + slot_w / 2, mz + 26.0, mx + 31, top + 1)])      # trimmed right horn
    return wall.difference(notch)


def plot(P, T, vm, title, path, holes_txt=""):
    tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    tc = ax.tripcolor(tri, facecolors=vm, cmap="turbo", edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mm")
    cb = fig.colorbar(tc, ax=ax, shrink=0.85)
    cb.set_label("von Mises stress [MPa]")
    i = int(np.argmax(vm))
    c = P[T[i]].mean(axis=0)
    ax.plot(c[0], c[1], "k+", ms=12)
    ax.annotate(f"max {vm[i]:.0f} MPa", c, xytext=(10, 10), textcoords="offset points", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


CST_CORRECTION = 2.51 / 2.24      # validation: linear triangles under-read the hole-edge peak by ~11 % at h = 0.5


def run_case(name, outer, holes, fixed_fn, load_fn, t, h_far, h_near, mat=AL6061, region=None, refine=(), png=None):
    """Max von Mises is reported raw and 'engineering' (excluding elements touching supports or
    the assumed pin-contact patch). If `region(centroids) -> mask` is given, the governing value
    is the max inside that region (a converged location such as a bending edge), and the fine
    mesh is re-run with a second random seed to show the scatter."""
    out = {}
    for tag, hn in (("coarse", h_near * 2), ("fine", h_near)):
        P, T = fe2d.mesh(outer, holes, h_far, hn, refine=refine)
        fixed = fixed_fn(P)
        loads = load_fn(P)
        u, vm, A, R = fe2d.solve(P, T, mat["E"], mat["nu"], t, fixed, loads)
        applied = np.array(list(loads.values())).sum(axis=0)
        reac = np.array([R[0::2][fixed].sum(), R[1::2][fixed].sum()])
        touching = np.isin(T, fixed).any(axis=1)
        loaded = np.isin(T, np.array(list(loads.keys()))).any(axis=1)
        vm_eng = vm[~touching & ~loaded]          # outside supports AND outside the assumed contact patch
        cents = P[T].mean(axis=1)
        reg = float(vm[region(cents) & ~loaded & ~touching].max()) if region else None
        out[tag] = dict(nodes=len(P), elements=len(T), max_vm_raw=float(vm.max()),
                        max_vm=float(vm_eng.max()), contact_patch_max=float(vm[loaded].max()),
                        region_max=reg,
                        max_disp_mm=float(np.abs(u).max()), equilibrium_error_N=float(np.linalg.norm(applied + reac)))
        if tag == "fine":
            vm_plot = vm.copy()
            vm_plot[touching] = np.nan
            plot(P, T, np.nan_to_num(vm_plot, nan=0.0), f"{name}: {mat['name']}, t = {t} mm, {len(T)} elements",
                 os.path.join(OUT, png or f"fea_{name.split(' ')[0].lower()}.png"))
    key = "region_max" if region else "max_vm"
    if region:
        P, T = fe2d.mesh(outer, holes, h_far, h_near, seed=7, refine=refine)
        u, vm, A, R = fe2d.solve(P, T, mat["E"], mat["nu"], t, fixed_fn(P), load_fn(P))
        ld = np.isin(T, np.array(list(load_fn(P).keys()))).any(axis=1)
        tc = np.isin(T, fixed_fn(P)).any(axis=1)
        out["fine_seed7_region_max"] = float(vm[region(P[T].mean(axis=1)) & ~ld & ~tc].max())
    out["governing"] = key
    out["mesh_change_pct"] = 100 * (out["fine"][key] - out["coarse"][key]) / out["fine"][key]
    out["peak_corrected_MPa"] = out["fine"][key] * CST_CORRECTION
    out["FOS_yield"] = mat["Sy"] / out["peak_corrected_MPa"]
    out["material"] = mat["name"]
    return out


def validation():
    """Plate with a central hole in tension, d/W = 0.2: Peterson Kt(net) = 2.51."""
    W, Lp, d, t, sig = 100.0, 300.0, 20.0, 1.0, 10.0
    outer = [(-Lp / 2, -W / 2), (Lp / 2, -W / 2), (Lp / 2, W / 2), (-Lp / 2, W / 2)]
    hole = fe2d.Circle(0, 0, d / 2)
    res = {}
    for hn in (1.0, 0.5):
        P, T = fe2d.mesh(outer, [hole], 8.0, hn, grade=6.0)
        left = np.where(np.abs(P[:, 0] + Lp / 2) < 1e-6)[0]
        right = np.where(np.abs(P[:, 0] - Lp / 2) < 1e-6)[0]
        # whole left edge clamped (far from the hole, so it does not affect Kt); uniform traction on the right
        ys = P[right, 1]
        order = np.argsort(ys)
        right, ys = right[order], ys[order]
        trib = np.zeros(len(right))
        trib[1:] += np.diff(ys) / 2
        trib[:-1] += np.diff(ys) / 2
        loads = {int(n): (sig * t * w, 0.0) for n, w in zip(right, trib)}
        fixed = left
        u, vm, A, R = fe2d.solve(P, T, AL6061["E"], AL6061["nu"], t, fixed, loads)
        # sigma_x at the hole edge (top), from element stresses near (0, d/2)
        cents = P[T].mean(axis=1)
        near = np.where((np.abs(cents[:, 0]) < hn) & (cents[:, 1] > d / 2) & (cents[:, 1] < d / 2 + 1.5 * hn))[0]
        kt_net = vm[near].max() / (sig * W / (W - d))
        res[f"h={hn}"] = round(float(kt_net), 2)
    res["peterson_kt_net"] = 2.51
    return res


def main():
    summary = {"load_cases": {"design_point_N": F_DESIGN, "hard_stop_N": F_STOP,
                              "note": "hard stop = motor at 18 N m peak with the crank at the toggle-side stop"}}
    summary["validation_plate_with_hole"] = validation()

    d = link_dir_world()                       # force on crank pin / yoke, toward the pedal
    L, S, R = MECH["link"], MECH["slot_travel"], 6.25

    # 1) slotted pull-link, link frame: x from the pedal pin toward the crank, y = height
    x0, x1 = 54.0, L + R + 10.0                            # boss ends at x 54 (rev C)
    outer = outline(rounded_end_bar(x0, x1, 13.0, 6.0))       # R6 far-end corners as built
    slot = fe2d.Slot(L - S, 0, L, 0, R)
    summary["pull_link"] = run_case(
        "Pull-link (slot end loaded by the crank bush)", outer, [slot],
        lambda P: np.where(np.abs(P[:, 0] - x0) < 1e-6)[0],
        lambda P: fe2d.bearing_load(P, slot, (1.0, 0.0), F_STOP), t=8.0, h_far=3.0, h_near=0.5,
        region=lambda c: c[:, 0] >= L + R + 2.0)            # outer half of the end bridge (bending edge)

    # 2) inner crank plate, crank frame (pin on +x). It turns freely on its bearing, so it can
    #    only carry the RADIAL part of its half of the pin load; the tangential part goes to the
    #    motor-side plate through the pin. Balanced bearing loads at pin hole and bore, plus a
    #    two-node constraint that only removes rigid-body motion (reactions ~ 0).
    phi = math.radians(MECH["phi_stop"])
    dl = np.array([d[0] * math.cos(-phi) - d[1] * math.sin(-phi), d[0] * math.sin(-phi) + d[1] * math.cos(-phi)])
    radial = np.array([1.0, 0.0])
    # rev C: pin pressed into both plates -> the inner plate can take up to HALF the radial component;
    # the tangential part needs a torque the R7 press fit cannot give (< 250 N at r 20)
    f_rad = 0.5 * F_STOP * abs(dl @ radial)
    f_tan = F_STOP * abs(dl[0] * radial[1] - dl[1] * radial[0])
    R_IN = math.hypot(f_rad, min(f_tan, 250.0))            # inner-support (6201 / wall) load bound
    pull = -radial if dl @ radial < 0 else radial          # pin pushes toward (or away from) the axis
    bore, pinh = fe2d.Circle(0, 0, 6.0), fe2d.Circle(20.0, 0, 4.0)
    outer = outline(crank_inner_outline(MECH_LUG_ANGLE))   # trimmed plate as built (hub, pin boss, lug arm)

    def crank_loads(P):
        ld = fe2d.bearing_load(P, pinh, pull, f_rad)
        for k, (fx, fy) in fe2d.bearing_load(P, bore, -pull, f_rad).items():
            ox, oy = ld.get(k, (0.0, 0.0))
            ld[k] = (ox + fx, oy + fy)
        return ld

    def crank_fixed(P):
        far = np.argmin(P[:, 0])                            # hub rim, away from both holes
        top = np.argmax(P[:, 1])
        return np.array([far, top])

    summary["crank_plate_inner"] = run_case(
        "Crank plate (inner, half the radial pin load)", outer, [bore, pinh],
        crank_fixed, crank_loads, t=5.0, h_far=2.0, h_near=0.35,
        region=lambda c: (c[:, 0] > 8.0) & (c[:, 0] < 14.0) & (np.abs(c[:, 1]) < 3.0))   # ligament
    summary["crank_plate_inner"]["radial_load_N"] = f_rad

    # 2b) inner crank plate, STOP-LUG case: link slack (driver braking, actuator faulted to peak torque),
    #     18 N m pushes the lug onto the toggle-side collar. Stop force normal to the lug face at
    #     R_CONTACT; the crank pin reacts the moment (tangential); the stub bore takes the rest.
    a = math.radians(MECH_LUG_ANGLE)
    u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    pc = R_CONTACT * u - LUG_HALF_W * n                   # toggle collar sits on the -n side of the lug
    Fs = F_LUG * n
    Fp = np.array([0.0, -F_LUG * R_CONTACT / 20.0])        # pin at (20, 0): moment balance about the axis
    Fb = -(Fs + Fp)
    poly = crank_inner_outline(MECH_LUG_ANGLE)
    outer_l = outline(poly)
    roots = [(9.4, 9.0), tuple(9.38 * u - LUG_HALF_W * n), tuple(22.0 * u + LUG_HALF_W * n), tuple(pc),
             tuple(14.0 * u + LUG_HALF_W * n)]

    def lug_loads(P):
        return merge_loads(boundary_patch_load(P, orient(poly, 1.0), pc, 1.6, Fs),
                           fe2d.bearing_load(P, pinh, Fp, float(np.linalg.norm(Fp))),
                           fe2d.bearing_load(P, bore, Fb, float(np.linalg.norm(Fb))))

    def lug_root(c):                                       # lug root + the filled notch corner at s 22
        s_ = c @ u
        return (s_ > 6.0) & (s_ < 27.0) & (np.abs(c @ n) > 4.0)

    summary["crank_plate_inner_stop_lug"] = run_case(
        "Crank plate (inner) stop lug: 18 N m into the toggle stop", outer_l, [bore, pinh],
        crank_fixed, lug_loads, t=5.0, h_far=2.0, h_near=0.3, region=lug_root, refine=roots,
        png="fea_crank_lug.png")
    summary["crank_plate_inner_stop_lug"].update(stop_force_N=F_LUG, pin_force_N=float(np.linalg.norm(Fp)),
                                                 bore_force_N=float(np.linalg.norm(Fb)), contact_r_mm=R_CONTACT)

    # 3) yoke inner wall (world X-Z) as built: carrier bore + open-top slot in the outline,
    #    lightening window, two stop-screw holes and three carrier-screw holes; the carrier
    #    spigot bears on the bore with the inner-bearing load; root edge clamped.
    mx, mz = MECH["motor_xz"]
    top = mz + 46.0
    outer = outline(inner_wall_outline(mx, mz, top, 18.5, 14.0))      # 37 mm bore (rev C)
    bore_arc = fe2d.Circle(mx, mz, 18.5)                   # used only to find the loaded bore nodes
    holes = [fe2d.Slot(mx, 15.0, mx, 69.0, 13.0)]          # lightening window 26 x 80
    stud_holes = []
    for ang in (115.0, MECH_TOGGLE_ANG):                   # M6 through-studs
        a_ = math.radians(ang)
        stud_holes.append(fe2d.Circle(mx + 37 * math.cos(a_), mz + 37 * math.sin(a_), 3.2))
    holes += stud_holes
    for ang in (210.0, 270.0, 330.0):                      # M4 carrier screws (tapped)
        a_ = math.radians(ang)
        holes.append(fe2d.Circle(mx + 25.5 * math.cos(a_), mz + 25.5 * math.sin(a_), 1.65))
    for ang in (240.0, 300.0):                             # 4 mm carrier dowels
        a_ = math.radians(ang)
        holes.append(fe2d.Circle(mx + 25.5 * math.cos(a_), mz + 25.5 * math.sin(a_), 2.0))
    summary["yoke_inner_plate"] = run_case(
        "Yoke inner wall (carrier bore, inner-bearing load)", outer, holes,
        lambda P: np.where(np.abs(P[:, 1] + 22.0) < 1e-6)[0],
        lambda P: fe2d.bearing_load(P, bore_arc, d, R_IN), t=10.0, h_far=3.0, h_near=0.45)
    summary["yoke_inner_plate"]["bore_load_N"] = R_IN

    # 3b) same wall, link-slack stop case: the lug pushes the toggle stud, the stub pushes the bore
    rot = lambda v_: np.array([v_[0] * math.cos(phi) - v_[1] * math.sin(phi),        # noqa: E731
                               v_[0] * math.sin(phi) + v_[1] * math.cos(phi)])
    f_stud, f_bore = rot(-Fs), rot(-Fb)                    # forces ON the yoke (world frame)
    summary["yoke_inner_plate_link_slack"] = run_case(
        "Yoke inner wall (link slack: toggle stud + bearing)", outer, holes,
        lambda P: np.where(np.abs(P[:, 1] + 22.0) < 1e-6)[0],
        lambda P: merge_loads(fe2d.bearing_load(P, stud_holes[1], f_stud, float(np.linalg.norm(f_stud))),
                              fe2d.bearing_load(P, bore_arc, f_bore, float(np.linalg.norm(f_bore)))),
        t=10.0, h_far=3.0, h_near=0.45, png="fea_yoke_slack.png")
    summary["yoke_inner_plate_link_slack"].update(stud_force_N=float(np.linalg.norm(f_stud)),
                                                  bore_force_N=float(np.linalg.norm(f_bore)))

    for k in ("pull_link", "crank_plate_inner", "yoke_inner_plate"):
        r = summary[k]
        r["max_vm_design_point"] = r["peak_corrected_MPa"] * F_DESIGN / F_STOP
        r["FOS_design_point"] = AL6061["Sy"] / r["max_vm_design_point"]
    with open(os.path.join(OUT, "fea_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
