import sys, os, math, unittest, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fe2d
import sizing
from params import G

E, NU = 200000.0, 0.3


def grid(L, H, nx, ny):
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    P = np.array([(x, y) for y in ys for x in xs])
    idx = lambda i, j: j * (nx + 1) + i
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    return P, np.array(T), idx


def edge_w(n):
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    return w / w.sum()


def tri_areas(P, T):
    a = P[T]
    return 0.5 * ((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))


class FE(unittest.TestCase):
    def test_uniaxial_bar(self):
        F, t, L, H = 1000.0, 2.0, 100.0, 10.0
        nx, ny = 8, 4
        P, T, idx = grid(L, H, nx, ny)
        fixed = [idx(0, j) for j in range(ny + 1)]
        right = [idx(nx, j) for j in range(ny + 1)]
        loads = {n: (F * wi, 0.0) for n, wi in zip(right, edge_w(ny + 1))}
        u, vm, A, R = fe2d.solve(P, T, E, NU, t, fixed, loads)
        sig = F / (H * t)
        # clamped left edge restrains Poisson contraction: stress uniform only away from the wall
        mid = np.array([P[T[e]].mean(axis=0)[0] > L / 2 for e in range(len(T))])
        self.assertAlmostEqual(vm[mid].mean() / sig, 1.0, delta=0.02)
        tip = np.mean(u[2 * np.array(right)])
        self.assertAlmostEqual(tip / (F * L / (H * t * E)), 1.0, delta=0.02)
        self.assertAlmostEqual(A.sum(), L * H, places=9)

    def test_uniaxial_exact_no_poisson(self):
        # nu=0 removes the wall constraint -> exact uniform stress and exact FL/AE
        F, t, L, H = 1000.0, 2.0, 100.0, 10.0
        nx, ny = 5, 3
        P, T, idx = grid(L, H, nx, ny)
        fixed = [idx(0, j) for j in range(ny + 1)]
        right = [idx(nx, j) for j in range(ny + 1)]
        loads = {n: (F * wi, 0.0) for n, wi in zip(right, edge_w(ny + 1))}
        u, vm, A, R = fe2d.solve(P, T, E, 0.0, t, fixed, loads)
        np.testing.assert_allclose(vm, F / (H * t), rtol=1e-9)
        np.testing.assert_allclose(u[2 * np.array(right)], F * L / (H * t * E), rtol=1e-9)

    def test_equilibrium_reactions(self):
        F, t, L, H = 1000.0, 2.0, 100.0, 10.0
        P, T, idx = grid(L, H, 6, 3)
        left = [idx(0, j) for j in range(4)]
        right = [idx(6, j) for j in range(4)]
        loads = {n: (F * wi, 30.0 * wi) for n, wi in zip(right, edge_w(4))}
        u, vm, A, R = fe2d.solve(P, T, E, NU, t, left, loads)
        self.assertAlmostEqual(sum(R[2 * n] for n in left), -F, delta=1e-6 * F)
        self.assertAlmostEqual(sum(R[2 * n + 1] for n in left), -30.0, delta=1e-6 * F)

    def test_cantilever_convergence(self):
        L, H, t, F = 100.0, 10.0, 5.0, 100.0
        I = t * H ** 3 / 12
        beam_s = F * L ** 3 / (3 * E * I) + F * L / (5 / 6 * E / (2 * (1 + NU)) * H * t)
        ratios = []
        for ny in (1, 2, 4, 8, 16):
            nx = ny * 10
            P, T, idx = grid(L, H, nx, ny)
            fixed = [idx(0, j) for j in range(ny + 1)]
            right = [idx(nx, j) for j in range(ny + 1)]
            loads = {n: (0.0, -F * wi) for n, wi in zip(right, edge_w(ny + 1))}
            u, *_ = fe2d.solve(P, T, E, NU, t, fixed, loads)
            ratios.append(-np.mean(u[2 * np.array(right) + 1]) / beam_s)
        self.assertTrue(all(b > a for a, b in zip(ratios, ratios[1:])), ratios)
        self.assertLess(ratios[0], 0.5)
        self.assertGreater(ratios[-1], 0.93)
        self.assertLess(ratios[-1], 1.03)

    def test_patch_equilibrium_distorted(self):
        P = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [4.0, 3.5]], float)
        T = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        sx = 100.0
        loads = {1: (sx * 10 / 2, 0), 2: (sx * 10 / 2, 0)}
        u, vm, A, R = fe2d.solve(P, T, E, NU, 1.0, [0, 3], loads)
        self.assertAlmostEqual(R[0] + R[6], -sx * 10, delta=1e-6)

    def test_patch_uniform_stress_distorted_nu0(self):
        # nu=0, left edge fixed in x only matters not; with nu=0 clamped edge gives uniform sigma_x in a distorted patch
        P = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [4.0, 3.5]], float)
        T = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        sx = 100.0
        loads = {1: (sx * 10 / 2, 0), 2: (sx * 10 / 2, 0)}
        u, vm, A, R = fe2d.solve(P, T, E, 0.0, 1.0, [0, 3], loads)
        np.testing.assert_allclose(vm, sx, rtol=1e-8)

    def test_zero_load(self):
        P, T, idx = grid(10, 10, 3, 3)
        u, vm, A, R = fe2d.solve(P, T, E, NU, 1.0, [idx(0, j) for j in range(4)], {})
        self.assertEqual(np.abs(u).max(), 0.0)
        self.assertEqual(vm.max(), 0.0)
        self.assertLess(np.abs(R).max(), 1e-9)

    def test_load_on_fixed_node_gives_no_deflection(self):
        P, T, idx = grid(10, 10, 3, 3)
        fx = [idx(0, j) for j in range(4)]
        u, vm, A, R = fe2d.solve(P, T, E, NU, 1.0, fx, {fx[1]: (50.0, 0.0)})
        self.assertEqual(np.abs(u).max(), 0.0)

    def test_linearity_and_scaling(self):
        P, T, idx = grid(40, 10, 8, 2)
        fx = [idx(0, j) for j in range(3)]
        ld = {idx(8, 1): (0.0, -10.0)}
        u1, v1, *_ = fe2d.solve(P, T, E, NU, 1.0, fx, ld)
        u2, v2, *_ = fe2d.solve(P, T, E, NU, 1.0, fx, {k: (a * 3, b * 3) for k, (a, b) in ld.items()})
        np.testing.assert_allclose(u2, 3 * u1, rtol=1e-9)
        np.testing.assert_allclose(v2, 3 * v1, rtol=1e-9)
        u3, *_ = fe2d.solve(P, T, E, NU, 2.0, fx, ld)      # twice thickness -> half displacement
        np.testing.assert_allclose(u3, u1 / 2, rtol=1e-9)

    def test_ill_posed_bad_mesh_index(self):
        P, T, idx = grid(10, 10, 2, 2)
        with self.assertRaises(Exception):
            fe2d.solve(P, T, E, NU, 1.0, [99], {})

    def test_plate_with_hole_kt(self):
        W, Lx, r, t, sig = 100.0, 200.0, 5.0, 1.0, 100.0
        res = {}
        for hn in (1.0, 0.5):
            outer = [(0, 0), (Lx, 0), (Lx, W), (0, W)]
            hole = fe2d.Circle(Lx / 2, W / 2, r)
            P, T = fe2d.mesh(outer, [hole], h_far=8.0, h_near=hn, grade=6.0)
            left = np.where(P[:, 0] < 1e-6)[0]
            right = np.where(P[:, 0] > Lx - 1e-6)[0]
            order = right[np.argsort(P[right, 1])]
            ys = P[order, 1]
            trib = np.array([((ys[k + 1] if k < len(ys) - 1 else ys[k]) - (ys[k - 1] if k > 0 else ys[k])) / 2 for k in range(len(ys))])
            loads = {int(n): (sig * t * tr, 0.0) for n, tr in zip(order, trib)}
            self.assertAlmostEqual(sum(f[0] for f in loads.values()), sig * W * t, delta=1e-6 * sig * W)
            u, vm, A, R = fe2d.solve(P, T, E, NU, t, left, loads)
            res[hn] = vm.max() / sig
        for k in res.values():
            self.assertGreater(k, 2.0)
            self.assertLess(k, 3.3)
        self.assertGreaterEqual(res[0.5], res[1.0] - 0.05)

    def test_bearing_load_resultant(self):
        hole = fe2d.Circle(0, 0, 5)
        P = hole.boundary(0.5)
        l = fe2d.bearing_load(P, hole, (1, 0), 1000.0)
        self.assertAlmostEqual(sum(v[0] for v in l.values()), 1000.0, places=6)
        self.assertAlmostEqual(sum(v[1] for v in l.values()), 0.0, places=6)
        l = fe2d.bearing_load(P, hole, (1, 1), 500.0)
        self.assertAlmostEqual(sum(v[0] for v in l.values()), 500 / math.sqrt(2), places=6)
        self.assertTrue(set(l).issubset(set(map(int, fe2d.half_hole_nodes(P, hole, (1, 1))))))

    def test_mesh_area_and_orientation(self):
        P, T = fe2d.mesh([(0, 0), (60, 0), (60, 30), (0, 30)], [fe2d.Circle(30, 15, 6)], 5.0, 1.0)
        ar = tri_areas(P, T)
        self.assertTrue((ar > 0).all())
        self.assertAlmostEqual(ar.sum(), 1800 - math.pi * 36, delta=0.02 * 1800)
        self.assertEqual(len(np.unique(T)), len(P))

    def test_slot_mesh(self):
        P, T = fe2d.mesh([(0, 0), (60, 0), (60, 30), (0, 30)], [fe2d.Slot(20, 15, 40, 15, 5)], 5.0, 1.0)
        self.assertAlmostEqual(tri_areas(P, T).sum(), 1800 - (20 * 10 + math.pi * 25), delta=0.03 * 1800)


class Sizing(unittest.TestCase):
    def test_hand_chain(self):
        a = 1.0
        m, h, L, wf, R = 300., .28, 1.55, .47, .228
        W = m * G
        dW = m * a * G * h / L
        Fzf, Fzr = wf * W + dW, (1 - wf) * W - dW
        cf = a * Fzf * R / 2 / (2 * .4 * .085)
        cr = a * Fzr * R / 2 / (2 * .4 * .085)
        pf = cf / (2 * math.pi * .0254 ** 2 / 4)
        pr = cr / (math.pi * .0254 ** 2 / 4)
        Amc = math.pi * (0.625 * .0254) ** 2 / 4
        Fmc = (pf + pr) * Amc
        r = sizing.brake_chain(a)
        self.assertAlmostEqual(r["p_f_bar"], pf / 1e5, places=6)
        self.assertAlmostEqual(r["p_r_bar"], pr / 1e5, places=6)
        self.assertAlmostEqual(r["F_mc"], Fmc, places=6)
        self.assertAlmostEqual(r["T_pivot"], Fmc * .045, places=6)
        self.assertAlmostEqual(r["F_pad"], Fmc * .045 / .225, places=6)
        self.assertAlmostEqual(r["bias_front"], pf * Amc / Fmc, places=9)

    def test_monotonic_and_zero(self):
        prev = None
        for i in range(1, 20):
            r = sizing.brake_chain(i * 0.1)
            if prev:
                # rear pressure peaks near 1.5 g: ideal balance moves load forward faster than decel grows
                for k in ("p_f_bar", "F_mc", "F_pad", "T_pivot"):
                    self.assertGreater(r[k], prev[k], (i, k))
            prev = r
        z = sizing.brake_chain(0.0)
        self.assertEqual(z["F_mc"], 0.0)
        self.assertTrue(math.isnan(z["bias_front"]))

    def test_pad_mu_inverse(self):
        a, b = sizing.brake_chain(1.2), sizing.brake_chain(1.2, pad_mu=0.8)
        self.assertAlmostEqual(a["p_f_bar"] / b["p_f_bar"], 2.0, places=9)

    def test_pedal_rotation(self):
        self.assertAlmostEqual(sizing.pedal_rotation_deg(0), math.degrees(1.5e-3 / .045), places=9)
        self.assertGreater(sizing.pedal_rotation_deg(50), sizing.pedal_rotation_deg(10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
