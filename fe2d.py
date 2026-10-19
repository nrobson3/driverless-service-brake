"""Small 2D plane-stress finite-element solver (linear triangles) for the flat,
in-plane-loaded parts of the service-brake actuator.

Why 2D: the pull-link, the crank plates and the yoke's inner plate are constant-
thickness plates loaded in their own plane, so plane stress is the right model.
The yoke's outer plate + gussets carry out-of-plane load and are covered by hand
calcs + SolidWorks SimulationXpress instead (see fea_results.md).

Mesh: boundary points + graded interior points, Delaunay, then triangles whose
centroid or any edge midpoint falls outside the material are removed.
Pin loads: cosine-distributed bearing pressure over the loaded half of the hole.
"""
import math

import numpy as np
from matplotlib.path import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay


# ------------------------------------------------------------------ geometry
class Circle:
    def __init__(self, cx, cy, r):
        self.c, self.r = np.array([cx, cy], float), r

    def inside(self, p):
        return np.hypot(p[:, 0] - self.c[0], p[:, 1] - self.c[1]) < self.r - 1e-9

    def boundary(self, h):
        n = max(24, int(math.ceil(2 * math.pi * self.r / h)))
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.c_[self.c[0] + self.r * np.cos(t), self.c[1] + self.r * np.sin(t)]

    def dist(self, p):
        return np.abs(np.hypot(p[:, 0] - self.c[0], p[:, 1] - self.c[1]) - self.r)


class Slot:
    """Stadium: segment a-b swept by radius r."""
    def __init__(self, ax, ay, bx, by, r):
        self.a, self.b, self.r = np.array([ax, ay], float), np.array([bx, by], float), r

    def _segdist(self, p):
        ab = self.b - self.a
        t = np.clip(((p - self.a) @ ab) / (ab @ ab), 0, 1)
        q = self.a + t[:, None] * ab
        return np.hypot(*(p - q).T)

    def inside(self, p):
        return self._segdist(p) < self.r - 1e-9

    def dist(self, p):
        return np.abs(self._segdist(p) - self.r)

    def boundary(self, h):
        ab = self.b - self.a
        L = np.linalg.norm(ab)
        u = ab / L
        n = np.array([-u[1], u[0]])
        pts = []
        m = max(2, int(math.ceil(L / h)))
        for s in np.linspace(0, L, m + 1):
            pts.append(self.a + s * u + self.r * n)
            pts.append(self.a + s * u - self.r * n)
        k = max(12, int(math.ceil(math.pi * self.r / h)))
        base = math.atan2(u[1], u[0])
        for t in np.linspace(-math.pi / 2, math.pi / 2, k + 1)[1:-1]:
            pts.append(self.b + self.r * np.array([math.cos(base + t), math.sin(base + t)]))
            pts.append(self.a - self.r * np.array([math.cos(base + t), math.sin(base + t)]))
        return np.array(pts)


def polygon_boundary(poly, h):
    pts = []
    for i in range(len(poly)):
        a, b = np.array(poly[i], float), np.array(poly[(i + 1) % len(poly)], float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / h)))
        for s in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + s * (b - a))
    return np.array(pts)


def circle_polygon(cx, cy, r, n=96):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


def mesh(outer, holes, h_far, h_near, grade=12.0, seed=1, refine=()):
    """outer: polygon vertex list; holes: Circle/Slot objects; refine: extra (x, y) points that get the
    fine mesh size around them (e.g. a fillet on the outer boundary). Returns nodes, triangles."""
    rng = np.random.default_rng(seed)
    path = Path(outer)
    xs, ys = zip(*outer)
    pts = [polygon_boundary(outer, h_near)]
    for hl in holes:
        pts.append(hl.boundary(h_near))
    # graded interior points: fine near holes, coarse elsewhere
    gx = np.arange(min(xs), max(xs) + h_near, h_near)
    gy = np.arange(min(ys), max(ys) + h_near, h_near)
    G = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
    G = G + rng.uniform(-0.15, 0.15, G.shape) * h_near
    dmin = np.full(len(G), np.inf)
    for hl in holes:
        dmin = np.minimum(dmin, hl.dist(G))
    for rx, ry in refine:
        dmin = np.minimum(dmin, np.hypot(G[:, 0] - rx, G[:, 1] - ry))
    local_h = np.clip(h_near + (dmin / grade) * h_near, h_near, h_far)
    keep = rng.uniform(0, 1, len(G)) < (h_near / local_h) ** 2
    G = G[keep]
    inside = path.contains_points(G, radius=-0.6 * h_near)
    for hl in holes:
        inside &= hl.dist(G) > 0.6 * h_near
        inside &= ~hl.inside(G)
    pts.append(G[inside])
    P = np.vstack(pts)
    P = np.unique(np.round(P, 6), axis=0)
    tri = Delaunay(P).simplices
    keep = np.ones(len(tri), bool)
    probes = [P[tri].mean(axis=1)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        probes.append((P[tri[:, i]] + P[tri[:, j]]) / 2)
    for q in probes:
        keep &= path.contains_points(q, radius=1e-6)
        for hl in holes:
            keep &= ~hl.inside(q)
    tri = tri[keep]
    # drop degenerate slivers
    a = P[tri[:, 1]] - P[tri[:, 0]]
    b = P[tri[:, 2]] - P[tri[:, 0]]
    area = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    tri = tri[np.abs(area) > 1e-8]
    flip = (0.5 * ((P[tri[:, 1]] - P[tri[:, 0]])[:, 0] * (P[tri[:, 2]] - P[tri[:, 0]])[:, 1]
                   - (P[tri[:, 1]] - P[tri[:, 0]])[:, 1] * (P[tri[:, 2]] - P[tri[:, 0]])[:, 0])) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    used = np.unique(tri)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    return P[used], remap[tri]


# ------------------------------------------------------------------ solver
def solve(P, T, E, nu, t, fixed_nodes, loads):
    """fixed_nodes: array of node ids (both dofs fixed). loads: {node: (fx, fy)} in N."""
    n = len(P)
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    rows, cols, vals = [], [], []
    Bs, areas = [], []
    for tri in T:
        x, y = P[tri, 0], P[tri, 1]
        A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        B = np.zeros((3, 6))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = b, c, c, b
        B /= 2 * A
        k = t * A * B.T @ D @ B
        dofs = np.array([[2 * i, 2 * i + 1] for i in tri]).ravel()
        rows.append(np.repeat(dofs, 6))
        cols.append(np.tile(dofs, 6))
        vals.append(k.ravel())
        Bs.append(B)
        areas.append(A)
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n)).tocsr()
    F = np.zeros(2 * n)
    for node, (fx, fy) in loads.items():
        F[2 * node] += fx
        F[2 * node + 1] += fy
    fixed = np.zeros(2 * n, bool)
    fixed[2 * np.asarray(fixed_nodes)] = True
    fixed[2 * np.asarray(fixed_nodes) + 1] = True
    free = ~fixed
    u = np.zeros(2 * n)
    u[free] = spsolve(K[free][:, free], F[free])
    vm = np.empty(len(T))
    for e, tri in enumerate(T):
        dofs = np.array([[2 * i, 2 * i + 1] for i in tri]).ravel()
        s = D @ (Bs[e] @ u[dofs])
        vm[e] = math.sqrt(s[0] ** 2 - s[0] * s[1] + s[1] ** 2 + 3 * s[2] ** 2)
    reaction = K @ u - F
    return u, vm, np.array(areas), reaction


def bearing_load(P, hole, direction, total):
    """Cosine bearing pressure on the half of a circular boundary facing `direction`."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    if isinstance(hole, Circle):
        c, r = hole.c, hole.r
    else:                                  # slot: load the end cap that faces `direction`
        ends = [hole.a, hole.b]
        c = max(ends, key=lambda e: e @ d)
        r = hole.r
    rr = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    on = np.where(np.abs(rr - r) < 1e-3)[0]
    w = []
    for i in on:
        nvec = (P[i] - c) / r
        cosang = nvec @ d
        w.append(max(cosang, 0.0))
    w = np.array(w)
    sel = w > 1e-9
    on, w = on[sel], w[sel]
    w = w / w.sum() * total
    return {int(i): (wi * d[0], wi * d[1]) for i, wi in zip(on, w)}


def half_hole_nodes(P, hole, direction):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    rr = np.hypot(P[:, 0] - hole.c[0], P[:, 1] - hole.c[1])
    on = np.where(np.abs(rr - hole.r) < 1e-3)[0]
    return np.array([i for i in on if ((P[i] - hole.c) / hole.r) @ d > 0.0])
