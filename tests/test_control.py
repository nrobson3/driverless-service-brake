import sys, os, json, unittest, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "mechanism.json")):
    import mechanism
    mechanism.main()                                  # control_sim reads its output
import control_sim as cs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(REPO, "out", "control_results.json")):
    cs.main()
with open(os.path.join(REPO, "out", "control_results.json")) as f:
    RES = json.load(f)


def step_log(target=40.0, t_end=0.8, **kw):
    return cs.simulate(t_end, lambda t: target if t >= 0.1 else 0.0, **kw)


class Control(unittest.TestCase):
    def test_step_40_reproduces(self):
        m = cs.metrics_step(step_log(40.0), 0.1, 40.0)
        for k, v in RES["step_40bar"].items():
            self.assertAlmostEqual(m[k], v, delta=max(0.02 * abs(v), 0.05), msg=k)
        self.assertLess(m["rise_10_90_ms"], 150.0)
        self.assertLess(abs(m["steady_error_bar"]), 2.0)
        self.assertLess(m["peak_torque_Nm"], cs.T_MAX)

    def test_step_59_reproduces(self):
        m = cs.metrics_step(step_log(59.0), 0.1, 59.0)
        for k, v in RES["step_59bar"].items():
            self.assertAlmostEqual(m[k], v, delta=max(0.02 * abs(v), 0.05), msg=k)
        self.assertLess(m["rise_10_90_ms"], 150.0)
        self.assertLess(abs(m["steady_error_bar"]), 2.0)

    def test_override(self):
        th50 = cs.mech.theta_for_pressure(50.0) * cs.D2R
        tau_drv = cs.hyd_torque(th50)[0] + cs.hyd_torque(th50)[2]

        def driver(t):
            return tau_drv * min(1.0, (t - 0.5) / 0.08) if 0.5 <= t < 1.1 else 0.0
        log = cs.simulate(1.7, lambda t: 30.0 if 0.1 <= t < 1.4 else 0.0, driver_fn=driver)
        st = log["state"]
        t_det = next((log["t"][i] for i in range(len(st)) if st[i] == "OVERRIDE"), None)
        self.assertIsNotNone(t_det)
        self.assertGreater(t_det, 0.5)
        self.assertAlmostEqual(t_det, RES["override"]["override_detected_s"], delta=0.01)
        self.assertLess(t_det - 0.5, 0.2)
        self.assertNotIn("OVERRIDE", st[: int(0.49 / cs.CTRL_DT)])
        ten = log["tension"][log["t"] > t_det + 0.05]
        self.assertLess(ten.max(), 5.0)
        self.assertAlmostEqual(float(log["p"].max()), RES["override"]["max_pressure_bar"], delta=1.5)
        self.assertEqual(st[-1], "READY")

    def test_no_override_without_driver(self):
        log = cs.simulate(0.8, lambda t: 40.0 if t >= 0.1 else 0.0)
        self.assertNotIn("OVERRIDE", log["state"])
        self.assertNotIn("SAFE_STOP", log["state"])

    def test_timeout_safe_stop(self):
        log = cs.simulate(1.2, lambda t: 25.0 if t >= 0.1 else 0.0, valid_fn=lambda t: t < 0.5)
        st = log["state"]
        t_ss = next((log["t"][i] for i in range(len(st)) if st[i] == "SAFE_STOP"), None)
        self.assertIsNotNone(t_ss)
        self.assertAlmostEqual(t_ss - 0.5, 0.100, delta=0.003)
        self.assertAlmostEqual(t_ss, RES["command_timeout"]["safe_stop_entered_s"], delta=0.005)
        self.assertGreater(log["p"][-50:].mean(), 30.0)
        self.assertLess(log["p"][-50:].mean(), 40.0)
        i = st.index("SAFE_STOP")
        self.assertTrue(all(s == "SAFE_STOP" for s in st[i:]))

    def test_power_loss_release(self):
        log = cs.simulate(1.0, lambda t: 40.0 if t >= 0.1 else 0.0, power_fn=lambda t: t < 0.5)
        p, t = log["p"], log["t"]
        t_rel = next((t[i] for i in range(len(p)) if t[i] > 0.5 and p[i] < 2.0), None)
        self.assertIsNotNone(t_rel)
        self.assertAlmostEqual(t_rel, RES["power_loss"]["pressure_below_2bar_s"], delta=0.01)
        self.assertLess(p[-1], 1.0)
        self.assertEqual(log["state"][-1], "FAULT")

    def test_zero_command_stays_ready_no_pressure(self):
        log = cs.simulate(0.3, lambda t: 0.0)
        self.assertLess(log["p"].max(), 0.5)
        self.assertTrue(all(s == "READY" for s in log["state"]))

    def test_dt_sensitivity(self):
        orig = cs.DT
        out = {}
        try:
            for dt in (1.0e-5, 2.0e-5, 2.5e-5, 4.0e-5, 1.0e-4, 2.0e-4):
                cs.DT = dt
                log = step_log(40.0, 0.8)
                self.assertTrue(np.isfinite(log["p"]).all(), f"NaN at dt={dt}")
                self.assertLess(np.abs(log["p"]).max(), 80.0, f"blow-up at dt={dt}")
                out[dt] = cs.metrics_step(log, 0.1, 40.0)
        finally:
            cs.DT = orig
        ref = out[2.0e-5]
        for dt, m in out.items():
            self.assertAlmostEqual(m["rise_10_90_ms"], ref["rise_10_90_ms"], delta=0.1 * ref["rise_10_90_ms"] + 2, msg=f"dt={dt}")
            self.assertAlmostEqual(m["steady_error_bar"], ref["steady_error_bar"], delta=0.5, msg=f"dt={dt}")

    def test_seed_noise_robust(self):
        r = [cs.metrics_step(step_log(40.0, 0.6, seed=s), 0.1, 40.0)["rise_10_90_ms"] for s in (1, 2, 3)]
        self.assertLess(max(r) - min(r), 20.0)

    def test_p_from_phi_inverse(self):
        for p in (5, 20, 40, 70):
            th = cs.mech.theta_for_pressure(p)
            ph, _ = cs.mech.crank_for_engaged(th)
            self.assertAlmostEqual(cs.p_from_phi(ph * cs.D2R), p, delta=0.3)
        self.assertEqual(cs.p_from_phi(cs.PHI_PARK), 0.0)

    def test_gap_zero_at_engaged(self):
        th = -8.0
        ph, _ = cs.mech.crank_for_engaged(th)
        self.assertAlmostEqual(cs.gap(ph * cs.D2R, th * cs.D2R), 0.0, places=7)

    def test_stale_results_json(self):
        import subprocess, shutil, tempfile
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, "r")
        shutil.copytree(REPO, dst)
        subprocess.run([sys.executable, "control_sim.py"], cwd=dst, check=True, capture_output=True)
        b = json.load(open(os.path.join(dst, "out", "control_results.json")))
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(b, RES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
