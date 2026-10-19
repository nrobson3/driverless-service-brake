"""Fusion build for the WFR driverless service brake (concept D) on a REFERENCE pedal box.

Executed inside Fusion through a short loader (MCP). Everything is built from
TemporaryBRepManager primitives into base features, so the model is fully
reproducible from this file + out/mechanism.json (run mechanism.py first).

Frame (mm): origin on the pedal pivot axis, X forward, Y outboard (actuator
side), Z up. Fusion works in cm; helpers convert.

Entry points (called by the loader):
    build(run_id)                 -> (re)builds every component in the WFR document
    set_pose(theta_deg, phi_deg)  -> moves pedal, crank and link
    clearance_report(...)         -> min distances / interference for actuator parts
"""
import json
import math
import os

import adsk.core
import adsk.fusion

# repository root: set BRAKE_REPO if the loader runs this file without __file__
BASE = os.environ.get("BRAKE_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECH = json.load(open(os.path.join(BASE, "out", "mechanism.json"), encoding="utf-8"))

APP = adsk.core.Application.get()
TM = adsk.fusion.TemporaryBRepManager.get()
P3 = adsk.core.Point3D.create
V3 = adsk.core.Vector3D.create

# ------------------------------------------------------------------ parameters (mm)
THETA_REST = MECH["theta_rest"]
R_PIN = MECH["r_pin"]
MX, MZ = MECH["motor_xz"]
R_CRANK = MECH["r_crank"]
LINK = MECH["link"]
SLOT = MECH["slot_travel"]

PIVOT_Z_ABOVE_FLOOR = 30.0
FLOOR = -PIVOT_Z_ABOVE_FLOOR            # top of the pedal-box base plate
MC_Y = 34.0                             # master cylinders at y = +/-34
BB_R = 45.0                             # balance bar radius on the pedal
MC_Z = BB_R * math.cos(math.radians(THETA_REST))
BB_X = BB_R * math.sin(math.radians(THETA_REST))

# actuator envelope: CubeMars AK80-9 class (update from datasheet)
MOTOR_R = 49.0
MOTOR_Y0, MOTOR_Y1 = 21.0, 59.5         # output face .. back face
FLANGE_R, FLANGE_Y0 = 26.0, 21.0
PILOT_R, PILOT_DEPTH = 10.0, 1.5         # HOLD: output-flange centring spigot (confirm on the CubeMars drawing)
LOCK_T = 0.8                             # flange-screw lock ring thickness (pocket depth)

# link / crank layout in Y
LINK_Y0, LINK_Y1 = 7.0, 15.0
CRANK_IN = (1.0, 6.0)                   # inner crank plate
CRANK_OUT = (16.0, 21.0)                # outer crank plate (bolted to motor flange)
YOKE_IN = (-14.0, -4.0)                 # inner yoke plate (outboard bearing), 10 mm
YOKE_OUT = (MOTOR_Y1, MOTOR_Y1 + 8.0)   # outer yoke plate (motor stator mount)

DOC_ATTR = ("wfr", "doc")


def cm(x):
    return x / 10.0


def pt(x, y, z):
    return P3(cm(x), cm(y), cm(z))


# ------------------------------------------------------------------ primitives
def box(x0, x1, y0, y1, z0, z1):
    c = pt((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    obb = adsk.core.OrientedBoundingBox3D.create(c, V3(1, 0, 0), V3(0, 1, 0),
                                                 cm(abs(x1 - x0)), cm(abs(y1 - y0)), cm(abs(z1 - z0)))
    return TM.createBox(obb)


def obox(center, ldir, wdir, length, width, height):
    """Oriented box; center (mm), ldir/wdir unit vectors (tuples), sizes in mm."""
    obb = adsk.core.OrientedBoundingBox3D.create(pt(*center), V3(*ldir), V3(*wdir),
                                                 cm(length), cm(width), cm(height))
    return TM.createBox(obb)


def cyl(a, b, r):
    return TM.createCylinderOrCone(pt(*a), cm(r), pt(*b), cm(r))


def union(a, *others):
    for o in others:
        TM.booleanOperation(a, o, adsk.fusion.BooleanTypes.UnionBooleanType)
    return a


def cut(a, *tools):
    for t in tools:
        TM.booleanOperation(a, t, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return a


def tube(a, b, ro, ri):
    return cut(cyl(a, b, ro), cyl(a, b, ri))


def wedge_yz(x0, x1, y_corner, z_corner, y_far, z_far):
    """Right-triangle plate in a Y-Z plane, thickness x0..x1; right angle at
    (y_corner, z_corner), legs to (y_far, z_corner) and (y_corner, z_far)."""
    b = box(x0, x1, min(y_corner, y_far), max(y_corner, y_far), min(z_corner, z_far), max(z_corner, z_far))
    dy, dz = y_far - y_corner, z_far - z_corner
    hyp = math.hypot(dy, dz)
    t = (0.0, -dy / hyp, dz / hyp)                     # along the hypotenuse (y_far,zc)->(yc,z_far)
    n = (0.0, dz / hyp, dy / hyp)                      # normal pointing away from the right angle
    if (n[1] * (y_far - y_corner) + n[2] * 0.0) < 0:
        n = (0.0, -n[1], -n[2])
    mid = ((x0 + x1) / 2, (y_corner + y_far) / 2, (z_corner + z_far) / 2)
    big = 4 * hyp
    c = (mid[0], mid[1] + n[1] * big / 2, mid[2] + n[2] * big / 2)
    tool = obox(c, (1, 0, 0), t, 4 * abs(x1 - x0) + 10, big, big)
    return cut(b, tool)


def hex_prism(center, axis, af, length):
    """Regular hexagon prism (across-flats af) centred at `center`, extruded `length` along
    axis 'x' | 'y'. Built as the intersection of three rotated slabs."""
    cx, cy, cz = center
    out = None
    for k in range(3):
        a = math.radians(60 * k)
        if axis == "y":
            ldir, wdir = (math.cos(a), 0.0, math.sin(a)), (0.0, 1.0, 0.0)
            slab = obox(center, ldir, wdir, af, length, af * 2.0)
        else:
            ldir, wdir = (0.0, math.cos(a), math.sin(a)), (1.0, 0.0, 0.0)
            slab = obox(center, ldir, wdir, af, length, af * 2.0)
        if out is None:
            out = slab
        else:
            TM.booleanOperation(out, slab, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return out


def tri_prism_y(pts, y0, y1):
    """Triangular prism: vertices pts = [(x, z)] x3 in an X-Z section, extruded y0..y1. Built as the
    intersection of three half-space slabs (no sketch needed)."""
    big = 1000.0
    out = None
    for i in range(3):
        a, b_, c = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
        L = math.hypot(b_[0] - a[0], b_[1] - a[1])
        t = ((b_[0] - a[0]) / L, (b_[1] - a[1]) / L)
        nrm = (-t[1], t[0])
        if (c[0] - a[0]) * nrm[0] + (c[1] - a[1]) * nrm[1] < 0:
            nrm = (-nrm[0], -nrm[1])                       # toward the third vertex
        mid = ((a[0] + b_[0]) / 2 + nrm[0] * big / 2, (a[1] + b_[1]) / 2 + nrm[1] * big / 2)
        slab = obox((mid[0], (y0 + y1) / 2, mid[1]), (t[0], 0.0, t[1]), (0.0, 1.0, 0.0), big, y1 - y0, big)
        if out is None:
            out = slab
        else:
            TM.booleanOperation(out, slab, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return out


def lug_notch_fill(lug_deg, hw, s_c=22.0, r_boss=9.0):
    """Crank-frame triangle (hub at the origin, pin boss at (R_CRANK, 0)) that fills the narrow notch
    between the pin-boss web and the lug: from inside the web to the boss tangent point to a point
    on the lug's web-side face at s_c. FE showed the R3 notch there governed the stop-lug case."""
    a = math.radians(lug_deg)
    ux, uz = math.cos(a), math.sin(a)
    C = (s_c * ux - hw * uz, s_c * uz + hw * ux)           # +n side (n = u rotated +90 deg)
    dx, dz = C[0] - R_CRANK, C[1]
    d = math.hypot(dx, dz)
    beta, alpha = math.atan2(dz, dx), math.acos(r_boss / d)
    T = max(((R_CRANK + r_boss * math.cos(beta + sg * alpha), r_boss * math.sin(beta + sg * alpha)) for sg in (1, -1)),
            key=lambda q: q[0])                            # outer tangent point on the boss
    return [(R_CRANK * 0.5, -5.0), T, C]


def stadium_y(x0, x1, z0, z1, y0, y1):
    """Through-cut tool for a rounded slot/window in a plate normal to Y (full-radius ends on the
    short sides)."""
    w, h = x1 - x0, z1 - z0
    if h >= w:
        r = w / 2
        return union(box(x0, x1, y0, y1, z0 + r, z1 - r),
                     cyl(((x0 + x1) / 2, y0, z0 + r), ((x0 + x1) / 2, y1, z0 + r), r),
                     cyl(((x0 + x1) / 2, y0, z1 - r), ((x0 + x1) / 2, y1, z1 - r), r))
    r = h / 2
    return union(box(x0 + r, x1 - r, y0, y1, z0, z1),
                 cyl((x0 + r, y0, (z0 + z1) / 2), (x0 + r, y1, (z0 + z1) / 2), r),
                 cyl((x1 - r, y0, (z0 + z1) / 2), (x1 - r, y1, (z0 + z1) / 2), r))


def rounded_rect_z(x0, x1, y0, y1, z0, z1, r):
    """Through-cut tool for a rounded rectangle in a plate normal to Z."""
    t = union(box(x0 + r, x1 - r, y0, y1, z0, z1), box(x0, x1, y0 + r, y1 - r, z0, z1))
    for cx in (x0 + r, x1 - r):
        for cy_ in (y0 + r, y1 - r):
            t = union(t, cyl((cx, cy_, z0), (cx, cy_, z1), r))
    return t


def socket_head_y(x, z, y_face, d_head, h_head, af_socket, outward):
    """Socket-head cap screw head sitting on a face at y_face, extending `outward` (+1/-1)."""
    y_end = y_face + outward * h_head
    head = cyl((x, min(y_face, y_end), z), (x, max(y_face, y_end), z), d_head / 2)
    sock_depth = 0.6 * h_head
    ys = y_end - outward * sock_depth / 2
    return cut(head, hex_prism((x, ys, z), "y", af_socket, sock_depth + 0.2))


# ------------------------------------------------------------------ document handling
def find_doc():
    for i in range(APP.documents.count):
        d = APP.documents.item(i)
        try:
            des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
        except Exception:
            continue
        if des and des.attributes.itemByName(*DOC_ATTR):
            return d, des
    return None, None


def get_doc(create=True):
    d, des = find_doc()
    if d is None and create:
        d = APP.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
        des.attributes.add(DOC_ATTR[0], DOC_ATTR[1], "WFR Service Brake - Reference Pedal Box + Actuator")
    if d is not None:
        d.activate()
    return d, des


def add_component(root, name, bodies, part_number=None):
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = name
    if part_number:
        comp.partNumber = part_number
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    for _, tb in bodies:
        comp.bRepBodies.add(tb, bf)
    bf.finishEdit()
    for i, (bname, _) in enumerate(bodies):
        comp.bRepBodies.item(i).name = bname
    return occ


# ------------------------------------------------------------------ geometry
def pedal_box_bodies():
    b = []
    b.append(("Base plate 6 mm", box(-130, 250, -120, 140, FLOOR - 6, FLOOR)))
    for s in (1, -1):
        b.append((f"Pivot clevis cheek {'L' if s > 0 else 'R'}",
                  cut(box(-20, 20, s * 15, s * 21, FLOOR, 15), cyl((0, s * 14, 0), (0, s * 22, 0), 4.2))))
    b.append(("Pivot bolt M8", cyl((0, -25, 0), (0, 25, 0), 4.0)))
    fw = box(40, 46, -75, 75, FLOOR, 85)
    for y in (-MC_Y, MC_Y):
        fw = cut(fw, cyl((39, y, MC_Z), (47, y, MC_Z), 6.0))
    b.append(("Master-cylinder bulkhead", fw))
    for y, tag in ((-MC_Y, "front"), (MC_Y, "rear")):
        b.append((f"Master cylinder ({tag}) 5-8 in", union(cyl((46, y, MC_Z), (146, y, MC_Z), 16.0),
                                                            box(46, 51, y - 22, y + 22, MC_Z - 12, MC_Z + 12))))
        b.append((f"Pushrod ({tag})", cyl((2, y, MC_Z), (46, y, MC_Z), 4.0)))
        b.append((f"Outlet fitting ({tag})", cyl((146, y, MC_Z), (158, y, MC_Z), 6.0)))
        b.append((f"Sensor tee ({tag})", box(158, 178, y - 10, y + 10, MC_Z - 10, MC_Z + 10)))
        # sensor clocked 45 deg up and inboard so the plug and its cable bend clear the floor
        sp = lambda k: (168, y - k / math.sqrt(2.0), MC_Z + k / math.sqrt(2.0))      # noqa: E731
        b.append((f"Brake-pressure sensor ({tag} circuit)",
                  union(cyl(sp(10), sp(25), 11.0), cyl(sp(25), sp(52), 9.5))))
        b.append((f"Brake line ({tag})", cyl((178, y, MC_Z), (240, y, MC_Z), 2.4)))
    b.append(("Brake over-travel switch (BOTS)", box(30, 50, -10, 10, 85, 105)))
    return b


def pedal_bodies():
    """Built vertical (theta = 0); the occurrence transform applies the pedal angle."""
    arm = box(-13, 13, -5, 5, 0, 250)
    arm = union(arm, cyl((0, -5, 0), (0, 5, 0), 15.0))
    arm = union(arm, cyl((0, -10, BB_R), (0, 10, BB_R), 11.0))
    arm = cut(arm, cyl((0, -6, 0), (0, 6, 0), 4.2))                       # pivot bore
    arm = cut(arm, cyl((0, -6, R_PIN), (0, 6, R_PIN), 5.1))               # actuator pick-up hole
    arm = cut(arm, cyl((0, -6, 120.0), (0, 6, 120.0), 5.1))               # spare hole reserved for EBS
    b = [("Brake pedal arm (reference)", arm)]
    b.append(("Pivot sleeve", tube((0, -15, 0), (0, 15, 0), 9.0, 4.2)))
    b.append(("Foot pad", box(-21, -13, -40, 40, 195, 255)))
    bb = cyl((0, -40, BB_R), (0, 40, BB_R), 5.0)
    for y in (-MC_Y, MC_Y):
        bb = union(bb, box(-2, 4, y - 6, y + 6, BB_R - 6, BB_R + 6))
    b.append(("Balance bar + clevises", bb))
    return b


def foot_keepout():
    """Driver foot envelope at the pedal rest position (world frame): sole from the
    ball of the foot on the pad down-back at 50 deg for 200 mm, 90 mm thick, 110 wide."""
    t = math.radians(THETA_REST)
    bx = -21 * math.cos(t) + 225 * math.sin(t)          # pad face centre (ball of foot)
    bz = 21 * math.sin(t) + 225 * math.cos(t)
    a = math.radians(50.0)
    d = (-math.cos(a), 0.0, -math.sin(a))                # from ball toward heel
    n = (-math.sin(a), 0.0, math.cos(a))                 # sole normal, into the foot
    L, T = 270.0, 90.0
    c = (bx + d[0] * (L / 2 - 70) + n[0] * T / 2, 0.0, bz + d[2] * (L / 2 - 70) + n[2] * T / 2)
    return obox(c, d, (0, 1, 0), L, 110.0, T)


def pin_bodies():
    """ISO 7379 shoulder screw 10 x 25 (M8) with a 3.2 mm spacer under the head, washers and an
    M8 nyloc - pedal frame (theta = 0). Stack: spacer | pedal arm | washer | rod-end ball | washer
    = 25.2 mm on a 25 mm shoulder, so the nut clamps the stack (rod-end ball included), not the step."""
    z = R_PIN
    screw = socket_head_y(0, z, -8.2, 16.0, 10.0, 8.0, -1)                  # head y -18.2..-8.2
    screw = union(screw, cyl((0, -8.2, z), (0, 16.8, z), 5.0), cyl((0, 16.8, z), (0, 30, z), 4.0))
    nut = cut(hex_prism((0, 21, z), "y", 13.0, 8.0), cyl((0, 16, z), (0, 26, z), 4.0))
    return [
        ("Shoulder screw ISO 7379 10x25 (M8, 12.9)", screw),
        ("Spacer 3.2 mm", tube((0, -8.2, z), (0, -5, z), 8.0, 5.1)),
        ("Washer 10 (inner)", tube((0, 5, z), (0, 6, z), 9.0, 5.1)),
        ("Washer 10 (outer)", tube((0, 16, z), (0, 17, z), 9.0, 5.1)),
        ("Nyloc nut M8", nut),
    ]


def actuator_bodies():
    body = cyl((MX, FLANGE_Y0 + 3, MZ), (MX, MOTOR_Y1, MZ), MOTOR_R)
    # output flange with its centring spigot (HOLD: diameter from the CubeMars drawing); the outer
    # crank plate pilots on it, which makes the crank coaxial with the motor output
    flange = union(cyl((MX, FLANGE_Y0, MZ), (MX, FLANGE_Y0 + 3, MZ), FLANGE_R),
                   cyl((MX, FLANGE_Y0 - PILOT_DEPTH, MZ), (MX, FLANGE_Y0, MZ), PILOT_R - 0.02))
    # AK80-9 KV100 has board connectors (XT30PW-M power, 4-pin CAN): plugs go in through the 24 mm
    # access hole AFTER the motor is bolted in, so nothing rigid has to pass the wall at drop-in
    plugs = union(cyl((MX, MOTOR_Y1, MZ), (MX, MOTOR_Y1 + 12, MZ), 6.0),
                  cyl((MX, MOTOR_Y1 + 12, MZ), (MX, MOTOR_Y1 + 30, MZ), 3.0))
    return [("CAN servo actuator (CubeMars AK80-9 envelope)", body), ("Actuator output flange", flange),
            ("Power + CAN plugs (fitted after mounting)", plugs)]


# ---- one-piece yoke layout (mm) ----
INNER_TOP = MZ + 46.0                   # inner wall raised so the park/homing stop sits in metal
OUTER_TOP = MZ + 45.0                   # outer wall top is a R45 arch about the motor axis
RIGHT_HORN_TOP = MZ + 26.0              # right horn carries nothing: trimmed to just above the bore
G_OUT = YOKE_OUT[1] + 45.0              # 45 mm gusset root
BASE_Y1 = G_OUT + 2.0
BORE_D = 37.0                           # wall bore: the 36 mm carrier spigot floats 0.5 mm until aligned
SPIGOT_D = 36.0                         # carrier spigot (aligned to the crank stub at assembly, then doweled)
SLOT_W = 14.0                           # open-top slot: the crank's 12 mm stub shaft drops in from above
CARRIER_PCD, CARRIER_ANGLES = 51.0, (210.0, 270.0, 330.0)   # 5.3 mm ligament to the 37 mm bore
DOWEL_ANGLES = (240.0, 300.0)          # 2x 4 mm dowels, reamed through carrier + wall after alignment
GUSSET_TIP_CUT = 8.0                   # acute top tip of each gusset cut back (durability, handling)
FLOOR_HOLES = ((82.0, -4.0), (158.0, -4.0), (95.0, BASE_Y1 - 10.0), (145.0, BASE_Y1 - 10.0))
MOTOR_PCD = 60.0                        # HOLD: stator pattern from the CubeMars drawing

# ---- hard stops: lug on the inner crank plate against two through-studs (M6 + nyloc) ----
# toggle side = hardened steel collar 12 (rigid, so the toggle margin is exact); park side = 8 mm
# collar + 2 mm EPDM sleeve (homing bumper, glycol-fluid compatible). Both 12 mm OD at the lug.
STOP_R, STOP_SLEEVE_R, LUG_HALF_W = 37.0, 6.0, 9.0
LUG_LEN = 36.0                          # lug flat runs past the stop contact (r 33.8) before the tip radius
STOP_DELTA = math.degrees(math.asin((LUG_HALF_W + STOP_SLEEVE_R) / STOP_R))
PARK_STOP_ANG = 115.0                   # in the wall's left horn, clear of the open-top slot
LUG_ANGLE = PARK_STOP_ANG - STOP_DELTA - (MECH["phi_park"] + 3.0)      # crank-local lug direction
TOGGLE_STOP_ANG = MECH["phi_stop"] + LUG_ANGLE - STOP_DELTA


def stop_xz(ang):
    a = math.radians(ang)
    return MX + STOP_R * math.cos(a), MZ + STOP_R * math.sin(a)


def yoke_bodies():
    """One-piece machined yoke (6061-T6): base, inner wall (bearing carrier seat with an
    open-top slot), outer wall (motor mount) and two gussets. Fillets and edge breaks are
    added afterwards as Fusion fillet features (finish_parts)."""
    x0, x1 = MX - 45, MX + 45
    base = box(x0, x1, YOKE_IN[0], BASE_Y1, FLOOR, FLOOR + 8)
    inner = box(MX - 30, MX + 30, YOKE_IN[0], YOKE_IN[1], FLOOR + 8, INNER_TOP)
    outer = box(x0, x1, YOKE_OUT[0], YOKE_OUT[1], FLOOR + 8, OUTER_TOP)
    yoke = union(base, inner, outer)
    for xa in (x0, x1 - 8):
        g = wedge_yz(xa, xa + 8, YOKE_OUT[1], FLOOR + 8, G_OUT, MZ - 30)
        g = cut(g, box(xa - 1, xa + 9, YOKE_OUT[1] - 1, G_OUT + 1, MZ - 30 - GUSSET_TIP_CUT, MZ - 29))
        yoke = union(yoke, g)
        # lightening hole at the gusset's incircle centre (legs 45 x 122 -> r_in 18.5)
        yoke = cut(yoke, cyl((xa - 1, YOKE_OUT[1] + 18.5, FLOOR + 8 + 18.5), (xa + 9, YOKE_OUT[1] + 18.5, FLOOR + 8 + 18.5), 8.0))
    # outer wall: R45 arch about the motor axis (removes the unloaded top corners)
    yoke = cut(yoke, cut(box(x0 - 1, x1 + 1, YOKE_OUT[0] - 1, YOKE_OUT[1] + 1, MZ, OUTER_TOP + 1),
                         cyl((MX, YOKE_OUT[0] - 2, MZ), (MX, YOKE_OUT[1] + 2, MZ), 45.0)))
    # inner wall: trim the right horn above the bore
    yoke = cut(yoke, box(MX + SLOT_W / 2, MX + 31, YOKE_IN[0] - 1, YOKE_IN[1] + 1, RIGHT_HORN_TOP, INNER_TOP + 1))
    # inner wall: carrier bore + open-top slot + lightening window + stop-screw + carrier-screw holes
    yoke = cut(yoke, cyl((MX, YOKE_IN[0] - 1, MZ), (MX, YOKE_IN[1] + 1, MZ), BORE_D / 2),
               box(MX - SLOT_W / 2, MX + SLOT_W / 2, YOKE_IN[0] - 1, YOKE_IN[1] + 1, MZ, INNER_TOP + 1),
               stadium_y(MX - 13, MX + 13, FLOOR + 32, FLOOR + 112, YOKE_IN[0] - 1, YOKE_IN[1] + 1))
    # 1x45 chamfers on the bore (both faces) - lead-in for the carrier spigot
    yoke = cut(yoke, TM.createCylinderOrCone(pt(MX, YOKE_IN[1] - 1.0, MZ), cm(BORE_D / 2), pt(MX, YOKE_IN[1] + 0.01, MZ), cm(BORE_D / 2 + 1.01)),
               TM.createCylinderOrCone(pt(MX, YOKE_IN[0] - 0.01, MZ), cm(BORE_D / 2 + 1.01), pt(MX, YOKE_IN[0] + 1.0, MZ), cm(BORE_D / 2)))
    for ang in (PARK_STOP_ANG, TOGGLE_STOP_ANG):
        sx, sz = stop_xz(ang)
        yoke = cut(yoke, cyl((sx, YOKE_IN[0] - 1, sz), (sx, YOKE_IN[1] + 1, sz), 3.2))      # M6 clearance
    for ang in DOWEL_ANGLES:
        a = math.radians(ang)
        dx_, dz_ = MX + CARRIER_PCD / 2 * math.cos(a), MZ + CARRIER_PCD / 2 * math.sin(a)
        yoke = cut(yoke, cyl((dx_, YOKE_IN[0] - 1, dz_), (dx_, YOKE_IN[0] + 8, dz_), 2.0))
    for ang in CARRIER_ANGLES:
        a = math.radians(ang)
        yoke = cut(yoke, cyl((MX + CARRIER_PCD / 2 * math.cos(a), YOKE_IN[0] - 1, MZ + CARRIER_PCD / 2 * math.sin(a)),
                             (MX + CARRIER_PCD / 2 * math.cos(a), YOKE_IN[0] + 8, MZ + CARRIER_PCD / 2 * math.sin(a)), 1.65))
    # outer wall: cable pass, motor screws, lightening window near the bending neutral axis
    yoke = cut(yoke, cyl((MX, YOKE_OUT[0] - 1, MZ), (MX, YOKE_OUT[1] + 1, MZ), 12.0),
               TM.createCylinderOrCone(pt(MX, YOKE_OUT[0] - 0.01, MZ), cm(13.01), pt(MX, YOKE_OUT[0] + 1.0, MZ), cm(12.0)),
               TM.createCylinderOrCone(pt(MX, YOKE_OUT[1] - 1.0, MZ), cm(12.0), pt(MX, YOKE_OUT[1] + 0.01, MZ), cm(13.01)),
               stadium_y(MX - 18, MX + 18, FLOOR + 28, FLOOR + 104, YOKE_OUT[0] - 1, YOKE_OUT[1] + 1))
    for k in range(6):
        a = math.radians(30 + 60 * k)
        hx, hz = MX + MOTOR_PCD / 2 * math.cos(a), MZ + MOTOR_PCD / 2 * math.sin(a)
        yoke = cut(yoke, cyl((hx, YOKE_OUT[0] - 1, hz), (hx, YOKE_OUT[1] + 1, hz), 1.7))
    # base: central lightening window + floor holes
    yoke = cut(yoke, rounded_rect_z(MX - 27, MX + 27, 9, 49, FLOOR - 1, FLOOR + 9, 8.0),
               rounded_rect_z(MX - 16, MX + 16, 75, 96, FLOOR - 1, FLOOR + 9, 6.0))
    for hx, hy in FLOOR_HOLES:
        yoke = cut(yoke, cyl((hx, hy, FLOOR - 1), (hx, hy, FLOOR + 9), 3.3))
    b = [("Actuator yoke (one-piece, 6061-T6)", yoke)]

    # bearing carrier: flange on the inboard face + spigot in the wall bore; bearing seated on its lip
    yi0 = YOKE_IN[0]
    carrier = union(tube((MX, yi0 - 5, MZ), (MX, yi0, MZ), 29.5, 14.0), tube((MX, yi0, MZ), (MX, YOKE_IN[1], MZ), SPIGOT_D / 2, 16.0))
    for ang in CARRIER_ANGLES:                  # 5 mm holes for M4: +/-0.5 mm of alignment float
        a = math.radians(ang)
        carrier = cut(carrier, cyl((MX + CARRIER_PCD / 2 * math.cos(a), yi0 - 6, MZ + CARRIER_PCD / 2 * math.sin(a)),
                                   (MX + CARRIER_PCD / 2 * math.cos(a), yi0 + 1, MZ + CARRIER_PCD / 2 * math.sin(a)), 2.5))
    for ang in DOWEL_ANGLES:
        a = math.radians(ang)
        carrier = cut(carrier, cyl((MX + CARRIER_PCD / 2 * math.cos(a), yi0 - 6, MZ + CARRIER_PCD / 2 * math.sin(a)),
                                   (MX + CARRIER_PCD / 2 * math.cos(a), yi0 + 1, MZ + CARRIER_PCD / 2 * math.sin(a)), 2.0))
    b.append(("Bearing carrier (6061-T6)", carrier))
    for k, ang in enumerate(DOWEL_ANGLES):
        a = math.radians(ang)
        dx_, dz_ = MX + CARRIER_PCD / 2 * math.cos(a), MZ + CARRIER_PCD / 2 * math.sin(a)
        b.append((f"Carrier dowel 4 m6 x 12 {k + 1}", cyl((dx_, yi0 - 5, dz_), (dx_, yi0 + 7, dz_), 2.0)))
    b.append(("Bearing 6201-2RS (12x32x10)", tube((MX, yi0, MZ), (MX, YOKE_IN[1], MZ), 16.0, 6.0)))
    for k, ang in enumerate(CARRIER_ANGLES):
        a = math.radians(ang)
        cx_, cz_ = MX + CARRIER_PCD / 2 * math.cos(a), MZ + CARRIER_PCD / 2 * math.sin(a)
        b.append((f"Carrier screw M4 SHCS drilled head {k + 1}", union(socket_head_y(cx_, cz_, yi0 - 5, 7.0, 4.0, 3.0, -1),
                                                           cyl((cx_, yi0 - 5, cz_), (cx_, yi0 + 6, cz_), 2.0))))
    # hard stops: turned stud = collar (crank side, clamped on the wall face) + M6 thread through the
    # wall, washer + nyloc on the inboard face (positive locking). Fitted before the crank drops in.
    y_end = CRANK_IN[1] - 0.5
    for tag, ang, collar_r in (("park/homing", PARK_STOP_ANG, 4.0), ("toggle-side", TOGGLE_STOP_ANG, STOP_SLEEVE_R)):
        sx, sz = stop_xz(ang)
        stud = union(cyl((sx, YOKE_IN[1], sz), (sx, y_end, sz), collar_r), cyl((sx, yi0 - 10.1, sz), (sx, YOKE_IN[1], sz), 3.0))
        stud = cut(stud, hex_prism((sx, y_end - 1.5, sz), "y", 3.0 if collar_r < 5 else 4.0, 3.2))   # hex socket to hold it
        name = ("Hard-stop stud M6, steel collar 12 (toggle-side)" if collar_r > 5 else
                "Hard-stop stud M6, collar 8 (park/homing)")
        b.append((name, stud))
        if collar_r < 5:
            b.append((f"EPDM bumper sleeve ({tag})", tube((sx, YOKE_IN[1], sz), (sx, y_end, sz), STOP_SLEEVE_R, collar_r)))
        b.append((f"Washer M6 ({tag} stop)", tube((sx, yi0 - 1.6, sz), (sx, yi0, sz), 6.0, 3.2)))
        b.append((f"Nyloc nut M6 ({tag} stop)", cut(hex_prism((sx, yi0 - 4.6, sz), "y", 10.0, 6.0),
                                                    cyl((sx, yi0 - 8, sz), (sx, yi0 - 1, sz), 3.0))))
    # motor screws (outboard face) and floor screws
    for k in range(6):
        a = math.radians(30 + 60 * k)
        hx, hz = MX + MOTOR_PCD / 2 * math.cos(a), MZ + MOTOR_PCD / 2 * math.sin(a)
        b.append((f"Motor screw M3 SHCS drilled head {k + 1}", socket_head_y(hx, hz, YOKE_OUT[1], 5.5, 3.0, 2.5, +1)))
    for k, (hx, hy) in enumerate(FLOOR_HOLES):
        head = cut(cyl((hx, hy, FLOOR + 9), (hx, hy, FLOOR + 15), 5.0), cyl((hx, hy, FLOOR + 11.5), (hx, hy, FLOOR + 15.1), 2.5))
        b.append((f"Floor screw M6 SHCS {k + 1}", head))
        b.append((f"Floor washer M6 {k + 1}", tube((hx, hy, FLOOR + 8.0), (hx, hy, FLOOR + 9.0), 6.0, 3.2)))
    return b


def crank_bodies():
    """Built at phi = 0 (crank pin on +X of the motor axis). Plates are trimmed to what carries
    load: a hub, the pin boss and (inner plate) the stop-lug arm."""
    kx = MX + R_CRANK
    yi = (CRANK_IN[0], CRANK_IN[1])
    la = math.radians(LUG_ANGLE)
    ux, uz = math.cos(la), math.sin(la)
    lug_len = LUG_LEN                                    # arm reaches r = LUG_LEN + LUG_HALF_W
    inner = union(cyl((MX, yi[0], MZ), (MX, yi[1], MZ), 13.0),
                  box(MX, kx, yi[0], yi[1], MZ - 9.0, MZ + 9.0),
                  cyl((kx, yi[0], MZ), (kx, yi[1], MZ), 9.0),
                  obox((MX + ux * lug_len / 2, (yi[0] + yi[1]) / 2, MZ + uz * lug_len / 2), (ux, 0, uz), (0, 1, 0),
                       lug_len, yi[1] - yi[0], 2 * LUG_HALF_W),
                  cyl((MX + ux * lug_len, yi[0], MZ + uz * lug_len), (MX + ux * lug_len, yi[1], MZ + uz * lug_len), LUG_HALF_W))
    fill = [(MX + x_, z_ + MZ) for x_, z_ in lug_notch_fill(LUG_ANGLE, LUG_HALF_W)]
    inner = union(inner, tri_prism_y(fill, yi[0], yi[1]))
    inner = cut(inner, cyl((MX, yi[0] - 1, MZ), (MX, yi[1] + 1, MZ), 6.0), cyl((kx, yi[0] - 1, MZ), (kx, yi[1] + 1, MZ), 4.0))
    yo = (CRANK_OUT[0], CRANK_OUT[1])
    outer = union(cyl((MX, yo[0], MZ), (MX, yo[1], MZ), 21.0), box(MX, kx, yo[0], yo[1], MZ - 9.0, MZ + 9.0),
                  cyl((kx, yo[0], MZ), (kx, yo[1], MZ), 9.0))
    outer = cut(outer, cyl((kx, yo[0] - 1, MZ), (kx, yo[1] + 1, MZ), 4.0),
                cyl((kx, yo[1] - 2.0, MZ), (kx, yo[1] + 1, MZ), 5.6),                     # pin-head counterbore
                cyl((MX, yo[1] - PILOT_DEPTH, MZ), (MX, yo[1] + 1, MZ), PILOT_R))         # pilot on the flange spigot
    # 6x M3 hex-head (ISO 4017) to the motor flange on PCD 30, heads flush in 6.8 counterbores on the link
    # side, LOCKED by a 0.8 mm stainless ring: hex cut-outs over the heads, keyed on the crank pin (cannot
    # turn), held down by the PTFE washer. Positive locking that does not rely on screw tension (T10).
    csk = []
    ring = union(tube((MX, yo[0], MZ), (MX, yo[0] + LOCK_T, MZ), 20.0, 9.5),
                 cyl((kx, yo[0], MZ), (kx, yo[0] + LOCK_T, MZ), 8.0))
    pocket = union(tube((MX, yo[0] - 1, MZ), (MX, yo[0] + LOCK_T, MZ), 20.2, 9.3),
                   cyl((kx, yo[0] - 1, MZ), (kx, yo[0] + LOCK_T, MZ), 8.2))
    outer = cut(outer, pocket)
    ring = cut(ring, cyl((kx, yo[0] - 1, MZ), (kx, yo[0] + 2, MZ), 4.05))
    for i in range(6):
        t = math.radians(30 + 60 * i)
        hx, hz = MX + 15 * math.cos(t), MZ + 15 * math.sin(t)
        outer = cut(outer, cyl((hx, yo[0] - 1, hz), (hx, yo[1] + 1, hz), 1.7),
                    cyl((hx, yo[0] - 1, hz), (hx, yo[0] + 2.0, hz), 3.4))
        ring = cut(ring, hex_prism((hx, yo[0] + LOCK_T / 2, hz), "y", 5.6, LOCK_T + 1.0))
        head = hex_prism((hx, yo[0] + 1.0, hz), "y", 5.5, 2.0)
        csk.append((f"Flange screw M3 hex 12.9 {i + 1}",
                    union(head, cyl((hx, yo[0] + 2.0, hz), (hx, FLANGE_Y0 + 2.5, hz), 1.5))))
    csk.append(("Screw lock ring 0.8 (301 stainless)", ring))
    # stub shaft: 12 p6 press + Loctite 638 into the inner plate against the collar; collar stops 1 mm
    # short of the wall so the unit drops into the U with axial play; the 6201 inner ring floats (12 g6)
    shaft = union(cyl((MX, YOKE_IN[0] - 3, MZ), (MX, CRANK_IN[1], MZ), 6.0), cyl((MX, YOKE_IN[1] + 1.0, MZ), (MX, CRANK_IN[0], MZ), 8.0))
    # crank pin: stepped 8 h6 (17-4PH H900), R7 press in BOTH plates (pressed on a fixture that holds
    # the stub coaxial with the outer-plate pilot); head sunk in the outer plate, trapped by the flange
    pin_ = union(cyl((kx, CRANK_IN[0], MZ), (kx, yo[1] - 2.0, MZ), 4.0),
                 cyl((kx, yo[1] - 2.0, MZ), (kx, yo[1] - 0.2, MZ), 5.5))
    # link axial stack: plate gap 10.0 = 0.8 washer + 8.4 hardened sleeve + 0.8 washer (clamped);
    # link 8.0 -0.05/-0.10 on the sleeve -> 0.45-0.50 mm end float
    w0, w1 = CRANK_IN[1] + 0.8, CRANK_OUT[0] - 0.8
    return [
        ("Crank plate inner 5 mm (6061-T6, with stop lug)", inner),
        ("Crank plate outer 5 mm (6061-T6)", outer),
        ("Stub shaft 12 with collar (17-4PH)", shaft),
        ("Shaft circlip DIN 471-12", tube((MX, YOKE_IN[0] - 1.3, MZ), (MX, YOKE_IN[0] - 0.3, MZ), 8.5, 5.75)),
        ("Crank pin 8 h6 stepped (17-4PH, head trapped by the flange)", pin_),
        ("Crank sleeve 12 h6 x 8.4 (hardened, clamped)", tube((kx, w0, MZ), (kx, w1, MZ), 6.0, 4.0)),
        ("Washer crank inner (PTFE 0.8)", tube((kx, CRANK_IN[1], MZ), (kx, w0, MZ), 8.0, 4.0)),
        ("Washer crank outer (PTFE 0.8)", tube((kx, w1, MZ), (kx, CRANK_OUT[0], MZ), 8.0, 4.0)),
    ] + csk


SLOT_R = 6.25          # slot half-width for the 12 mm crank sleeve (0.25 mm running gap)
BOSS_X0, BOSS_X1 = 28.0, 54.0          # threaded boss (link frame): M10 18 deep, witness hole at 12


def link_bodies():
    """Built along +X from the PEDAL-pin centre at the link-frame origin. The rod end
    sits on the pedal shoulder bolt; the slot at the far end rides on the crank bush.
    Pull face (slot far end) at x = LINK: the crank pin sits there when engaged."""
    yc = (LINK_Y0 + LINK_Y1) / 2
    # rod end: housing ring + spherical ball (10 mm bore) + neck + male M10 shank (13 mm engagement)
    ball = TM.createSphere(pt(0, yc, 0), cm(9.5))
    TM.booleanOperation(ball, box(-12, 12, LINK_Y0 - 1, LINK_Y1 + 1, -12, 12), adsk.fusion.BooleanTypes.IntersectionBooleanType)
    ball = cut(ball, cyl((0, LINK_Y0 - 2, 0), (0, LINK_Y1 + 2, 0), 5.0))
    housing = union(tube((0, LINK_Y0, 0), (0, LINK_Y1, 0), 12.0, 9.0), box(10, 16, LINK_Y0, LINK_Y1, -6.5, 6.5))
    housing = cut(housing, TM.createSphere(pt(0, yc, 0), cm(9.5)))
    eye = union(housing, ball)
    shank = cyl((15, yc, 0), (43, yc, 0), 5.0)           # 15 mm engaged at nominal, +/-3 mm adjustment
    nut = cut(hex_prism((25, yc, 0), "x", 17.0, 6.0), cyl((21, yc, 0), (29, yc, 0), 5.0))
    boss = box(BOSS_X0, BOSS_X1, yc - 8, yc + 8, -13, 13)   # 16 x 26 x 26 boss carries the M10 thread
    far = LINK + SLOT_R + 10.0                          # 10 mm end beam past the pull face
    bar = union(box(BOSS_X1, far, LINK_Y0, LINK_Y1, -13, 13), boss)
    slot = union(box(LINK - SLOT, LINK, LINK_Y0 - 1, LINK_Y1 + 1, -SLOT_R, SLOT_R),
                 cyl((LINK - SLOT, LINK_Y0 - 1, 0), (LINK - SLOT, LINK_Y1 + 1, 0), SLOT_R),
                 cyl((LINK, LINK_Y0 - 1, 0), (LINK, LINK_Y1 + 1, 0), SLOT_R))
    bar = cut(bar, slot)
    bar = cut(bar, cyl((BOSS_X0 - 1, yc, 0), (BOSS_X0 + 18, yc, 0), 4.25),          # M10 tap drill 8.5, 18 deep
              TM.createCylinderOrCone(pt(BOSS_X0 - 0.01, yc, 0), cm(5.51), pt(BOSS_X0 + 1.26, yc, 0), cm(4.24)),  # 90 deg csk
              cyl((BOSS_X0 + 12, yc - 9, 0), (BOSS_X0 + 12, yc + 9, 0), 1.0))       # 2 mm witness hole: 12 mm min engagement
    return [("Rod end M10 (spherical, on pedal pin)", eye), ("Rod end shank M10", shank), ("Jam nut M10", nut),
            ("Slotted pull-link 8 mm (6061-T6)", bar)]


def keepout_bodies():
    return [("KEEP-OUT future EBS (shuttle valves / 2nd MC / cylinder)", box(185, 290, -75, 55, FLOOR, 75)),
            ("KEEP-OUT driver foot envelope (rest)", foot_keepout())]


# ------------------------------------------------------------------ poses
def rot_y(angle_deg, about_x=0.0, about_z=0.0):
    m = adsk.core.Matrix3D.create()
    m.setToRotation(math.radians(angle_deg), V3(0, 1, 0), pt(about_x, 0, about_z))
    return m


def pedal_pin_world(theta_deg):
    t = math.radians(theta_deg)
    return R_PIN * math.sin(t), R_PIN * math.cos(t)


def crank_pin_world(phi_deg):
    p = math.radians(phi_deg)
    return MX + R_CRANK * math.cos(p), MZ + R_CRANK * math.sin(p)


def occ_by_name(root, name):
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i)
        if o.component.name == name:
            return o
    return None


NAMES = dict(
    box="Reference pedal box", pedal="Reference brake pedal", pin="Pedal pick-up pin",
    act="CAN servo actuator", yoke="Actuator yoke bracket", crank="Crank assembly",
    link="Slotted pull-link", keep="Keep-out volumes",
)


def set_pose(theta_deg, phi_deg, capture=False):
    """Move the mechanism. Parametric designs revert uncaptured moves on the next
    compute, so pass capture=True for a pose that must persist (adds a Position
    snapshot); frame-by-frame capture leaves it False."""
    d, des = get_doc(create=False)
    root = des.rootComponent
    occs = [occ_by_name(root, NAMES[k]) for k in ("pedal", "pin", "crank", "link")]
    m_pedal = rot_y(theta_deg)
    # crank: built with the pin at +X (phi = 0); rotating by phi from +X toward +Z is -phi about +Y
    m_crank = rot_y(-phi_deg, MX, MZ)
    kx, kz = crank_pin_world(phi_deg)
    cx, cz = pedal_pin_world(theta_deg)
    # link frame: origin on the pedal pin, +X pointing at the crank pin
    alpha = math.degrees(math.atan2(kz - cz, kx - cx))
    m_link = rot_y(-alpha)
    m_link.translation = V3(cm(cx), 0, cm(cz))
    for o, m in zip(occs, (m_pedal, m_pedal, m_crank, m_link)):
        o.transform2 = m
    if capture and des.snapshots.hasPendingSnapshot:
        des.snapshots.add()
    dist = math.hypot(cx - kx, cz - kz)
    return {"theta": theta_deg, "phi": phi_deg, "pin_to_crankpin_mm": round(dist, 2),
            "engaged": abs(dist - LINK) < 0.05, "in_slot": LINK - SLOT - 0.01 <= dist <= LINK + 0.01}


def set_camera(target_mm, eye_mm, up=(0, 0, 1), ortho=False, extents_cm=None):
    vp = APP.activeViewport
    cam = vp.camera
    cam.cameraType = (adsk.core.CameraTypes.OrthographicCameraType if ortho
                      else adsk.core.CameraTypes.PerspectiveCameraType)
    cam.target = pt(*target_mm)
    cam.eye = pt(*eye_mm)
    cam.upVector = V3(*up)
    cam.isSmoothTransition = False
    if extents_cm and ortho:
        cam.viewExtents = extents_cm
    vp.camera = cam
    adsk.doEvents()
    vp.refresh()


def show(key, body_prefixes=None, visible=True):
    """Toggle a whole component (body_prefixes None) or selected bodies."""
    d, des = get_doc(create=False)
    o = occ_by_name(des.rootComponent, NAMES[key])
    if body_prefixes is None:
        o.isLightBulbOn = visible
        return
    for i in range(o.component.bRepBodies.count):
        b = o.component.bRepBodies.item(i)
        if any(b.name.startswith(p) for p in body_prefixes):
            b.isLightBulbOn = visible


# ------------------------------------------------------------------ clearance sweep
# Pairs that touch by design (bolted, pinned or running fits) are not reported as clashes.
BY_DESIGN = [
    ("Shoulder screw", "Rod end M10"), ("Washer 10", "Rod end M10"),
    ("Stub shaft", "Bearing 6201"), ("Shaft circlip", "Bearing 6201"),
    ("Crank plate outer", "Actuator output flange"), ("Flange screw", "Actuator output flange"),
    ("Screw lock ring", "Washer crank outer"),
    ("Foot pad", "KEEP-OUT driver"), ("Brake pedal arm", "KEEP-OUT driver"),   # the foot is ON the pad
]
WATCH = [("Slotted pull-link", "Crank sleeve")]     # running gap we want to see, not a clash
# The crank lug and the stop studs are NOT excluded: they are measured at every pose (touching only
# at the two stop poses), so a near-miss elsewhere in the sweep shows up.


def _bodies(root, key):
    o = occ_by_name(root, NAMES[key])
    return [o.bRepBodies.item(i) for i in range(o.bRepBodies.count)]


def _bbox_gap_mm(a, b):
    ba, bb = a.boundingBox, b.boundingBox
    gap = 0.0
    for lo1, hi1, lo2, hi2 in ((ba.minPoint.x, ba.maxPoint.x, bb.minPoint.x, bb.maxPoint.x),
                               (ba.minPoint.y, ba.maxPoint.y, bb.minPoint.y, bb.maxPoint.y),
                               (ba.minPoint.z, ba.maxPoint.z, bb.minPoint.z, bb.maxPoint.z)):
        gap = max(gap, lo2 - hi1, lo1 - hi2)
    return gap * 10.0


def _is_by_design(n1, n2):
    for a, b in BY_DESIGN:
        if (n1.startswith(a) and n2.startswith(b)) or (n1.startswith(b) and n2.startswith(a)):
            return True
    return False


def _overlap_volume_mm3(a, b):
    ta, tb = TM.copy(a), TM.copy(b)
    try:
        TM.booleanOperation(ta, tb, adsk.fusion.BooleanTypes.IntersectionBooleanType)
        return ta.volume * 1000.0
    except Exception:
        return 0.0


def clearance_at(root, groups_a, groups_b, near_mm=25.0):
    mm = APP.measureManager
    rows = []
    A = [b for g in groups_a for b in _bodies(root, g)]
    B = [b for g in groups_b for b in _bodies(root, g)]
    for a in A:
        for b in B:
            if a.assemblyContext == b.assemblyContext:
                continue
            if _is_by_design(a.name, b.name):
                continue
            if _bbox_gap_mm(a, b) > near_mm:
                continue
            dmm = mm.measureMinimumDistance(a, b).value * 10.0
            vol = _overlap_volume_mm3(a, b) if dmm < 1e-3 else 0.0
            rows.append((a.name, b.name, round(dmm, 2), round(vol, 3)))
    return rows


def clearance_sweep(out_path, poses):
    """poses: list of (label, theta, phi). Writes the minimum gap per body pair per scenario."""
    d, des = get_doc(create=False)
    root = des.rootComponent
    worst = {}
    log = []
    for label, th, ph in poses:
        state = set_pose(th, ph)
        rows = clearance_at(root, ["crank", "link"], ["pedal", "pin", "box", "yoke", "act", "keep"])
        rows += clearance_at(root, ["link"], ["crank"])
        rows += clearance_at(root, ["pedal", "pin"], ["yoke", "act", "keep", "crank"])
        for n1, n2, dmm, vol in rows:
            k = (label.split(":")[0], n1, n2)
            if k not in worst or dmm < worst[k][0] or vol > worst[k][1]:
                worst[k] = (dmm, vol, th, ph)
        log.append({"pose": label, "theta": th, "phi": ph, "state": state,
                    "min_gap_mm": min([r[2] for r in rows], default=None),
                    "clash": [r for r in rows if r[3] > 1e-3]})
    out = {"poses": log,
           "pairs": [{"scenario": k[0], "a": k[1], "b": k[2], "min_gap_mm": v[0], "overlap_mm3": v[1],
                      "at_theta": v[2], "at_phi": v[3]} for k, v in sorted(worst.items(), key=lambda kv: kv[1][0])]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def capture_frames(frames, out_dir, w=1280, h=720, token="cap"):
    """frames: list of (theta, phi, caption). Saves PNGs + captions.json; logs progress."""
    os.makedirs(out_dir, exist_ok=True)
    vp = APP.activeViewport
    caps = []
    log = os.path.join(out_dir, "_progress.txt")
    for i, (th, ph, cap) in enumerate(frames):
        state = set_pose(th, ph)
        adsk.doEvents()
        vp.refresh()
        fn = os.path.join(out_dir, f"f{i:03d}.png")
        vp.saveAsImageFile(fn, w, h)
        caps.append({"file": os.path.basename(fn), "caption": cap, **state})
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"{token} {i + 1}/{len(frames)}")
    with open(os.path.join(out_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=1)
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"{token} DONE {len(frames)}")


# ------------------------------------------------------------------ STEP export
def _export_comp(des, comp, path):
    em = des.exportManager
    ok = em.execute(em.createSTEPExportOptions(path, comp))
    txt = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
    return {"file": os.path.basename(path), "ok": bool(ok), "bytes": len(txt),
            "solids": txt.count("MANIFOLD_SOLID_BREP") + txt.count("BREP_WITH_VOIDS")}


def _temp_component(root, name, part_number, tbodies):
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = name
    comp.partNumber = part_number
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    for _, tb in tbodies:
        comp.bRepBodies.add(tb, bf)
    bf.finishEdit()
    for i, (bn, _) in enumerate(tbodies):
        comp.bRepBodies.item(i).name = bn
    return occ


def show_all():
    """Hidden bodies are silently left out of a component STEP export - unhide everything first."""
    d, des = get_doc(create=False)
    root = des.rootComponent
    for key in NAMES:
        o = occ_by_name(root, NAMES[key])
        if o is None:
            continue
        o.isLightBulbOn = True
        for i in range(o.component.bRepBodies.count):
            o.component.bRepBodies.item(i).isLightBulbOn = True


def export_steps(out_dir, theta_deg=None, phi_deg=None):
    os.makedirs(out_dir, exist_ok=True)
    d, des = get_doc(create=False)
    root = des.rootComponent
    show_all()
    res = []
    # 1) designed parts, each in its own component frame
    for key, fn in (("yoke", "WFR-SB-140 Actuator yoke bracket.step"), ("crank", "WFR-SB-150 Crank assembly.step"),
                    ("link", "WFR-SB-160 Slotted pull-link.step"), ("pin", "WFR-SB-120 Pedal pick-up pin.step")):
        o = occ_by_name(root, NAMES[key])
        res.append(_export_comp(des, o.component, os.path.join(out_dir, fn)))
    # 2) whole assembly at one pose, as a multi-body part in the pedal-box frame
    th = THETA_REST if theta_deg is None else theta_deg
    ph = MECH["phi_park"] if phi_deg is None else phi_deg
    set_pose(th, ph)
    copies = []
    for key in ("box", "pedal", "pin", "act", "yoke", "crank", "link"):
        o = occ_by_name(root, NAMES[key])
        for i in range(o.bRepBodies.count):
            b = o.bRepBodies.item(i)
            copies.append((f"{NAMES[key]} - {b.name}", TM.copy(b)))
    occ = _temp_component(root, "WFR-SB-000 Assembly (rest pose)", "WFR-SB-000", copies)
    res.append(_export_comp(des, occ.component, os.path.join(out_dir, "WFR-SB-000 Assembly rest pose (multibody).step")))
    occ.deleteMe()
    # 3) FEA-ready yoke: the one-piece machined yoke on its own (single solid for SimulationXpress)
    o = occ_by_name(root, NAMES["yoke"])
    yb = TM.copy(_body(o.component, "Actuator yoke"))
    occ = _temp_component(root, "FEA yoke single body", "WFR-SB-140-FEA", [("Yoke single body", yb)])
    res.append(_export_comp(des, occ.component, os.path.join(out_dir, "FEA - WFR-SB-140 yoke single body.step")))
    res.append({"fea_yoke_lumps": yb.lumps.count if hasattr(yb, "lumps") else None})
    occ.deleteMe()
    with open(os.path.join(out_dir, "_export_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


GOLD = "Paint - Metallic (Yellow)"      # WFR gold: every NEW machined part
LOOK_RULES = [
    # (component key, body-name prefix or "*", appearance) - first match wins
    ("keep", "KEEP-OUT driver", "Plastic - Translucent Matte (Blue)"),
    ("keep", "*", "Plastic - Translucent Matte (Green)"),
    ("box", "Brake-pressure sensor", "Plastic - Matte (Black)"),
    ("box", "Brake over-travel", "Plastic - Matte (Red)"),
    ("box", "Brake line", "Stainless Steel - Satin"),
    ("box", "Master cylinder", "Aluminum - Satin"),
    ("box", "*", "Aluminum - Anodized Rough (Grey)"),
    ("pedal", "*", "Aluminum - Anodized Rough (Grey)"),
    ("pin", "*", "Coating - Black Oxide"),
    ("act", "Power", "Plastic - Matte (Black)"),
    ("act", "*", "Paint - Metallic (Black)"),
    ("yoke", "Actuator yoke", GOLD),
    ("yoke", "Bearing carrier", GOLD),
    ("yoke", "EPDM", "Plastic - Matte (Red)"),
    ("yoke", "Carrier dowel", "Steel - Satin"),
    ("yoke", "Bearing", "Steel - Satin"),
    ("yoke", "Floor washer", "Steel - Satin"),
    ("yoke", "*", "Coating - Black Oxide"),
    ("crank", "Crank plate", GOLD),
    ("crank", "Flange screw", "Coating - Black Oxide"),
    ("crank", "Screw lock ring", "Stainless Steel - Satin"),
    ("crank", "Washer crank", "Plastic - Translucent Matte (White)"),
    ("crank", "*", "Steel - Satin"),
    ("link", "Slotted", GOLD),
    ("link", "Jam nut", "Coating - Black Oxide"),
    ("link", "*", "Steel - Satin"),
]

MATERIAL_RULES = [
    # physical materials (mass properties + STEP metadata); first match wins
    ("keep", "*", None),
    ("box", "Brake-pressure sensor", "Stainless Steel"),
    ("box", "Brake line", "Stainless Steel"),
    ("box", "Pivot bolt", "Steel"),
    ("box", "Pushrod", "Steel"),
    ("box", "*", "Aluminum 6061"),
    ("pedal", "Pivot sleeve", "Steel"),
    ("pedal", "*", "Aluminum 6061"),
    ("pin", "*", "Steel"),
    ("act", "*", "Aluminum 6061"),
    ("yoke", "Actuator yoke", "Aluminum 6061"),
    ("yoke", "Bearing carrier", "Aluminum 6061"),
    ("yoke", "EPDM", "Rubber, Black"),
    ("yoke", "*", "Steel"),
    ("crank", "Crank plate", "Aluminum 6061"),
    ("crank", "Washer crank", "PTFE"),
    ("crank", "Stub shaft", "Stainless Steel"),
    ("crank", "Crank pin", "Stainless Steel"),
    ("crank", "Screw lock ring", "Stainless Steel"),
    ("crank", "*", "Steel"),
    ("link", "Slotted", "Aluminum 6061"),
    ("link", "*", "Steel"),
]


def apply_materials(des):
    lib = APP.materialLibraries.itemByName("Fusion Material Library")
    cache = {}

    def mat(name):
        if name not in cache:
            m = des.materials.itemByName(name)
            if m is None:
                m = des.materials.addByCopy(lib.materials.itemByName(name), name)
            cache[name] = m
        return cache[name]

    root = des.rootComponent
    for key, nm in NAMES.items():
        o = occ_by_name(root, nm)
        if o is None:
            continue
        for i in range(o.component.bRepBodies.count):
            bdy = o.component.bRepBodies.item(i)
            for k, prefix, mname in MATERIAL_RULES:
                if k == key and (prefix == "*" or bdy.name.startswith(prefix)):
                    if mname:
                        bdy.material = mat(mname)
                    break


# ------------------------------------------------------------------ fillets (finish pass)
def _line_ends_mm(e):
    g = e.geometry
    if g.objectType != adsk.core.Line3D.classType():
        return None
    a, b = g.startPoint, g.endPoint
    return (a.x * 10, a.y * 10, a.z * 10), (b.x * 10, b.y * 10, b.z * 10)


def lines_along(body, axis, fixed, tol=0.05):
    out = []
    for e in body.edges:
        ends = _line_ends_mm(e)
        if ends is None:
            continue
        a, b = ends
        d = [b[i] - a[i] for i in range(3)]
        L = math.sqrt(sum(x * x for x in d))
        if L < 1e-6 or abs(abs(d[axis]) / L - 1.0) > 1e-3:
            continue
        if all(abs(a[i] - v) < tol and abs(b[i] - v) < tol for i, v in fixed.items()):
            out.append(e)
    return out


def sharp_edges(body, min_len_mm=0.6):
    """Edges where the two faces meet at a real angle (not tangent)."""
    out = []
    for e in body.edges:
        if e.faces.count != 2 or e.length * 10 < min_len_mm:
            continue
        ev = e.evaluator
        ok, p0, p1 = ev.getParameterExtents()
        ok, mp = ev.getPointAtParameter((p0 + p1) / 2)
        ns = []
        for f in e.faces:
            ok2, nv = f.evaluator.getNormalAtPoint(mp)
            if not ok2:
                break
            ns.append(nv)
        if len(ns) == 2 and abs(ns[0].dotProduct(ns[1])) < 0.985:
            out.append(e)
    return out


def _outward(body, mp, nv):
    """Face normal at an edge point, flipped if needed so it points out of the body."""
    n = V3(nv.x, nv.y, nv.z)
    n.normalize()
    q = mp.copy()
    t = n.copy()
    t.scaleBy(cm(0.02))
    q.translateBy(t)
    if body.pointContainment(q) == adsk.fusion.PointContainment.PointInsidePointContainment:
        n.scaleBy(-1.0)
    return n


def concave_edges(body, axis=None, min_len_mm=0.6, eps_mm=0.05):
    """Sharp internal (concave) edges: a point offset along (n1 - n2) from the edge lies inside the
    material for a concave corner and outside it for a convex one."""
    out = []
    for e in sharp_edges(body, min_len_mm):
        if axis is not None:
            ends = _line_ends_mm(e)
            if ends is None:
                continue
            a, b = ends
            d = [b[i] - a[i] for i in range(3)]
            L = math.sqrt(sum(x * x for x in d))
            if L < 1e-6 or abs(abs(d[axis]) / L - 1.0) > 1e-3:
                continue
        ev = e.evaluator
        ok, p0, p1 = ev.getParameterExtents()
        ok, mp = ev.getPointAtParameter((p0 + p1) / 2)
        ns = []
        for f in e.faces:
            ok2, nv = f.evaluator.getNormalAtPoint(mp)
            ns.append(_outward(body, mp, nv))
        d = V3(ns[0].x - ns[1].x, ns[0].y - ns[1].y, ns[0].z - ns[1].z)
        if d.length < 1e-6:
            continue
        d.normalize()
        d.scaleBy(cm(eps_mm))
        q = mp.copy()
        q.translateBy(d)
        if body.pointContainment(q) == adsk.fusion.PointContainment.PointInsidePointContainment:
            out.append(e)
    return out


def _fillet(comp, edges, r_mm):
    if not edges:
        return 0
    col = adsk.core.ObjectCollection.create()
    for e in edges:
        col.add(e)
    inp = comp.features.filletFeatures.createInput()
    inp.edgeSetInputs.addConstantRadiusEdgeSet(col, adsk.core.ValueInput.createByReal(cm(r_mm)), True)
    inp.isRollingBallCorner = True
    comp.features.filletFeatures.add(inp)
    return len(edges)


def _body(comp, prefix):
    for i in range(comp.bRepBodies.count):
        b = comp.bRepBodies.item(i)
        if b.name.startswith(prefix):
            return b
    return None


def fillet_plan():
    yc = (LINK_Y0 + LINK_Y1) / 2
    far = LINK + SLOT_R + 10.0
    return [
        ("yoke", "Actuator yoke", [
            ("lines", 0, {1: YOKE_IN[1], 2: FLOOR + 8}, 6.0),       # inner wall to base, inside the U
            ("lines", 0, {1: YOKE_OUT[0], 2: FLOOR + 8}, 6.0),      # outer wall to base, inside the U
            ("lines", 0, {1: YOKE_OUT[1], 2: FLOOR + 8}, 6.0),      # outer wall to base, outboard
            ("lines", 1, {0: MX - 30, 2: FLOOR + 8}, 4.0),          # inner wall side roots
            ("lines", 1, {0: MX + 30, 2: FLOOR + 8}, 4.0),
            ("lines", 2, {0: MX - 37, 1: YOKE_OUT[1]}, 4.0),        # gussets to outer wall
            ("lines", 2, {0: MX + 37, 1: YOKE_OUT[1]}, 4.0),
            ("lines", 1, {0: MX - 37, 2: FLOOR + 8}, 4.0),          # gussets to base
            ("lines", 1, {0: MX + 37, 2: FLOOR + 8}, 4.0),
            ("lines", 1, {0: MX - 30, 2: INNER_TOP}, 8.0),          # inner wall top corners
            ("lines", 1, {0: MX + 30, 2: RIGHT_HORN_TOP}, 6.0),      # trimmed right horn
            ("lines", 1, {0: MX - SLOT_W / 2, 2: INNER_TOP}, 2.0),  # slot mouth
            ("lines", 1, {0: MX + SLOT_W / 2, 2: RIGHT_HORN_TOP}, 2.0),
            ("lines", 2, {0: MX - 45, 1: YOKE_IN[0]}, 8.0),         # base plan-view corners
            ("lines", 2, {0: MX + 45, 1: YOKE_IN[0]}, 8.0),
            ("lines", 2, {0: MX - 45, 1: BASE_Y1}, 8.0),
            ("lines", 2, {0: MX + 45, 1: BASE_Y1}, 8.0),
            ("break", 0.8)]),
        ("yoke", "Bearing carrier", [("break", 0.5)]),
        ("crank", "Crank plate inner", [("concave", 1, 3.0), ("break", 0.5)]),   # R3 internal corners (lug root)
        ("crank", "Crank plate outer", [("concave", 1, 3.0), ("break", 0.5)]),
        ("link", "Slotted", [
            ("lines", 2, {0: BOSS_X1, 1: LINK_Y0}, 3.0),             # boss-to-bar step (stress relief)
            ("lines", 2, {0: BOSS_X1, 1: LINK_Y1}, 3.0),
            ("lines", 1, {0: far, 2: 13.0}, 6.0),                    # far-end corners
            ("lines", 1, {0: far, 2: -13.0}, 6.0),
            ("lines", 0, {1: yc - 8, 2: 13.0}, 3.0),                 # boss long edges
            ("lines", 0, {1: yc - 8, 2: -13.0}, 3.0),
            ("lines", 0, {1: yc + 8, 2: 13.0}, 3.0),
            ("lines", 0, {1: yc + 8, 2: -13.0}, 3.0),
            ("break", 0.3)]),                                       # slot pull faces keep their width
        ("act", "CAN servo actuator", [("break", 2.0)]),
        ("act", "Actuator output flange", [("break", 0.5)]),
    ]


def _edge_key(e):
    ev = e.evaluator
    ok, p0, p1 = ev.getParameterExtents()
    ok, mp = ev.getPointAtParameter((p0 + p1) / 2)
    return (round(mp.x * 10, 2), round(mp.y * 10, 2), round(mp.z * 10, 2))


def _break_in_chunks(comp, prefix, r):
    """Fallback when one big edge-break fillet fails: fillet the remaining sharp edges in
    chunks, re-reading the body after every success; single edges that still fail are skipped."""
    ok_n, bad = 0, set()
    for _ in range(400):
        body = _body(comp, prefix)
        cand = [e for e in sharp_edges(body) if _edge_key(e) not in bad]
        if not cand:
            break
        chunk = cand[:max(1, len(cand) // 6)]
        try:
            ok_n += _fillet(comp, chunk, r)
            continue
        except Exception:
            pass
        if len(chunk) == 1:
            bad.add(_edge_key(chunk[0]))
            continue
        for e in chunk:                       # retry the failed chunk one edge at a time
            body = _body(comp, prefix)
            live = [x for x in sharp_edges(body) if _edge_key(x) == _edge_key(e)]
            if not live:
                continue
            try:
                ok_n += _fillet(comp, live[:1], r)
            except Exception:
                bad.add(_edge_key(e))
    return ok_n, len(bad)


def finish_parts(log_path):
    """Materials, then fillets (structural first, profile corners next, edge breaks last),
    then appearances. Every stage is independent: a failure is logged and skipped."""
    d, des = get_doc(create=False)
    root = des.rootComponent
    apply_materials(des)
    log = []
    for key, prefix, steps in fillet_plan():
        comp = occ_by_name(root, NAMES[key]).component
        for st in steps:
            body = _body(comp, prefix)
            if body is None:
                log.append(f"{prefix}: body missing")
                break
            if st[0] == "lines":
                _, axis, fixed, r = st
                edges = lines_along(body, axis, fixed)
                try:
                    n = _fillet(comp, edges, r)
                    log.append(f"{prefix}: R{r} on {n} edges along {'XYZ'[axis]} {fixed}")
                except Exception as ex:
                    log.append(f"{prefix}: R{r} FAILED ({len(edges)} edges {fixed}): {str(ex)[:80]}")
            elif st[0] == "concave":
                _, axis, r = st
                edges = concave_edges(body, axis)
                try:
                    n = _fillet(comp, edges, r)
                    log.append(f"{prefix}: concave R{r} on {n} edges")
                except Exception as ex:
                    log.append(f"{prefix}: concave R{r} FAILED ({len(edges)} edges): {str(ex)[:80]}")
            else:
                r = st[1]
                edges = sharp_edges(body)
                try:
                    n = _fillet(comp, edges, r)
                    log.append(f"{prefix}: edge break R{r} on {n} edges")
                except Exception:
                    ok_n, skipped = _break_in_chunks(comp, prefix, r)
                    log.append(f"{prefix}: edge break R{r} chunked: {ok_n} ok, {skipped} skipped")
    # audit: any internal corner still sharp on a machined part is a stress riser we missed
    for key, prefix in (("yoke", "Actuator yoke"), ("yoke", "Bearing carrier"), ("crank", "Crank plate inner"),
                        ("crank", "Crank plate outer"), ("link", "Slotted")):
        body = _body(occ_by_name(root, NAMES[key]).component, prefix)
        cv = concave_edges(body)
        pts = []
        for e in cv[:12]:
            k_ = _edge_key(e)
            pts.append(f"({k_[0]:.1f},{k_[1]:.1f},{k_[2]:.1f}) L{e.length * 10:.1f}")
        log.append(f"AUDIT {prefix}: {len(cv)} sharp concave edges left " + "; ".join(pts))
        log.append(f"AUDIT {prefix}: mass {body.physicalProperties.mass * 1000:.1f} g")
    apply_looks(des)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    return log


def apply_looks(des):
    lib = APP.materialLibraries.itemByName("Fusion Appearance Library")
    cache = {}

    def look(name):
        if name not in cache:
            a = des.appearances.itemByName(name)
            if a is None:
                a = des.appearances.addByCopy(lib.appearances.itemByName(name), name)
            cache[name] = a
        return cache[name]

    root = des.rootComponent
    for key, nm in NAMES.items():
        o = occ_by_name(root, nm)
        if o is None:
            continue
        for i in range(o.component.bRepBodies.count):
            bdy = o.component.bRepBodies.item(i)
            for k, prefix, app_name in LOOK_RULES:
                if k == key and (prefix == "*" or bdy.name.startswith(prefix)):
                    bdy.appearance = look(app_name)
                    break


def build(run_id):
    lock = os.path.join(BASE, "out", "_build_lock.txt")
    if os.path.exists(lock) and open(lock, encoding="utf-8").read().strip() == run_id:
        print("build already ran for", run_id)
        return
    with open(lock, "w", encoding="utf-8") as f:
        f.write(run_id)
    d, des = get_doc(create=True)
    des.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = des.rootComponent
    while root.occurrences.count:
        root.occurrences.item(0).deleteMe()
    add_component(root, NAMES["box"], pedal_box_bodies(), "WFR-SB-REF-100")
    add_component(root, NAMES["pedal"], pedal_bodies(), "WFR-SB-REF-110")
    add_component(root, NAMES["pin"], pin_bodies(), "WFR-SB-120")
    add_component(root, NAMES["act"], actuator_bodies(), "WFR-SB-PUR-130")
    add_component(root, NAMES["yoke"], yoke_bodies(), "WFR-SB-140")
    add_component(root, NAMES["crank"], crank_bodies(), "WFR-SB-150")
    add_component(root, NAMES["link"], link_bodies(), "WFR-SB-160")
    add_component(root, NAMES["keep"], keepout_bodies(), "WFR-SB-REF-190")
    apply_looks(des)
    pose = set_pose(THETA_REST, MECH["phi_park"], capture=True)
    print("built", run_id, pose)
    return pose
