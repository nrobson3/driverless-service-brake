import sys, os, math, unittest, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mechanism as mech
import can_codec as cc
import sizing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Mech(unittest.TestCase):
    def test_link_length_constant_over_sweep(self):
        th_design = mech.theta_for_pressure(mech.DESIGN_PT["p_f_bar"])
        for th in np.linspace(mech.THETA_REST, th_design, 41):
            phi, gam = mech.crank_for_engaged(th)
            d = np.linalg.norm(mech.crank_pin(phi) - mech.pin(th))
            self.assertAlmostEqual(d, mech.LINK, places=6, msg=f"theta={th}")
            self.assertGreater(gam, mech.GAMMA_STOP - 1e-6)

    def test_link_length_with_adjustment(self):
        for link in (mech.LINK - 3, mech.LINK + 3):
            for th in (-15, -10, -6):
                phi, _ = mech.crank_for_engaged(th, link)
                self.assertAlmostEqual(np.linalg.norm(mech.crank_pin(phi) - mech.pin(th)), link, places=6)

    def test_unreachable_returns_none(self):
        self.assertIsNone(mech.crank_for_engaged(60.0))

    def test_crank_monotonic_in_theta(self):
        phis = [mech.crank_for_engaged(t)[0] for t in np.linspace(-15, -4, 30)]
        self.assertTrue(all(b < a for a, b in zip(phis, phis[1:])))

    def test_pressure_roundtrip(self):
        for p in (1, 10, 30, 59, 80):
            self.assertAlmostEqual(mech.pressure_at(mech.theta_for_pressure(p)), p, places=9)
            self.assertAlmostEqual(mech.pressure_at(mech.theta_for_pressure(p, 1.5), 1.5), p, places=9)
        self.assertEqual(mech.pressure_at(mech.THETA_REST), 0.0)
        self.assertEqual(mech.pressure_at(mech.THETA_REST - 5), 0.0)

    def test_statics_hand_check(self):
        th = -8.0
        phi, gam = mech.crank_for_engaged(th)
        tau_p, p = mech.pedal_torque_Nm(th, margin=False)
        F, tau_m, lever = mech.forces(th, phi, tau_p)
        C = mech.pin(th) / 1000
        K = mech.crank_pin(phi) / 1000
        M = np.array(mech.MOTOR) / 1000
        u = (K - C) / np.linalg.norm(K - C)
        # independent: pedal moment about origin (z-component of C x F), link pulls pin toward K
        Fv = F * u
        m_ped = C[0] * Fv[1] - C[1] * Fv[0]
        self.assertAlmostEqual(abs(m_ped), tau_p, places=6)
        # crank: link reacts on crank pin with -Fv; moment about M
        r = K - M
        m_crank = r[0] * (-Fv[1]) - r[1] * (-Fv[0])
        self.assertAlmostEqual(abs(m_crank), tau_m, places=6)
        # hydraulic torque cross-check against sizing chain
        Fmc = p * 1e5 * sizing.area(0.625 * 0.0254) / mech.BIAS_F
        self.assertAlmostEqual(tau_p, Fmc * 0.045 + 40 * 0.225, places=6)
        self.assertTrue(math.isfinite(F) and F > 0 and tau_m > 0)
        # toggle gain: force magnification F/(tau_m) ~ 1/(R sin(gamma)) loosely

    def test_forces_finite_through_range_and_stop(self):
        th_stop = mech.gamma_to_theta(mech.GAMMA_STOP)
        phi, gam = mech.crank_for_engaged(th_stop)
        self.assertAlmostEqual(gam, mech.GAMMA_STOP, places=3)
        sc = mech.stop_case_for_link(mech.LINK, phi)
        for k, v in sc.items():
            self.assertTrue(math.isfinite(v), k)
        # at the stop the motor peak torque is applied -> link force F*tau1 consistency
        F1, tau1, lev = mech.forces(th_stop, phi, 1.0)
        self.assertAlmostEqual(sc["link_force_N"], mech.MOTOR_PEAK_NM / tau1 * F1, delta=1.0)
        # stop force bigger than max in-range design force
        MJ = json.load(open(os.path.join(REPO, "out", "mechanism.json")))
        self.assertGreater(sc["link_force_N"], MJ["link_force_max_N"])
        self.assertAlmostEqual(sc["link_force_N"], MJ["stop_link_force_N"], delta=2)

    def test_gamma_bisection(self):
        for g in (30, 35, 50, 70):
            th = mech.gamma_to_theta(g)
            self.assertAlmostEqual(mech.crank_for_engaged(th)[1], g, places=3)

    def test_lost_motion(self):
        """Parked actuator: pedal can travel past the engaged crank limit (within slot) without link load."""
        MJ = json.load(open(os.path.join(REPO, "out", "mechanism.json")))
        M = np.array(mech.MOTOR)
        C_rest = mech.pin(mech.THETA_REST)
        target = mech.LINK - mech.PARK_CLEAR
        d = M - C_rest
        D = float(np.hypot(*d))
        q = (target ** 2 - D * D - mech.R_CRANK ** 2) / (2 * mech.R_CRANK)
        phi_park = math.degrees(math.atan2(d[1], d[0]) + math.acos(q / D))
        self.assertAlmostEqual(phi_park, MJ["phi_park"], places=1)
        K = mech.crank_pin(phi_park)
        th_full = mech.THETA_REST + mech.THETA_SWEEP
        th_stop = mech.gamma_to_theta(mech.GAMMA_STOP)
        self.assertGreater(th_full, th_stop)                         # pedal goes further than engaged crank allows
        for th in np.linspace(mech.THETA_REST, th_full, 60):
            dist = np.linalg.norm(K - mech.pin(th))
            self.assertLess(dist, mech.LINK, f"link pulls at theta={th}")           # never tensioned (parked)
            self.assertGreater(dist, mech.LINK - MJ["slot_travel"], f"slot bottoms at theta={th}")
        self.assertGreaterEqual(MJ["min_manual_gap_mm"], 5.0 - 1e-9)
        self.assertGreaterEqual(MJ["min_slot_end_gap_mm"], 5.0 - 1e-9)

    def test_adjustment_study(self):
        MJ = json.load(open(os.path.join(REPO, "out", "mechanism.json")))
        for r in MJ["adjustment"]:
            self.assertLessEqual(abs(r["adjustment_used_mm"]), mech.ROD_END_ADJ + 1e-9)
            self.assertAlmostEqual(r["park_gap_mm"], mech.PARK_CLEAR, delta=0.4 + 1e-9)
            self.assertGreaterEqual(r["slot_end_gap_mm"], 5.0 - 1.0 - 1e-9)

    def test_stale_json(self):
        """out/mechanism.json committed must equal a fresh run (copy dir)."""
        import subprocess, shutil, tempfile
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, "r")
        shutil.copytree(REPO, dst)
        subprocess.run([sys.executable, "mechanism.py"], cwd=dst, check=True, capture_output=True)
        a = json.load(open(os.path.join(REPO, "out", "mechanism.json")))
        b = json.load(open(os.path.join(dst, "out", "mechanism.json")))
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(a, b)


class Can(unittest.TestCase):
    msgs = cc.parse()

    def test_dbc_parses_and_layout(self):
        self.assertEqual(set(self.msgs), {"ASB_SB_Cmd", "ASB_SB_Status", "ASB_SB_Diag"})
        self.assertEqual(self.msgs["ASB_SB_Cmd"]["id"], 272)
        n = sum(len(m["sig"]) for m in self.msgs.values())
        raw = open(cc.DBC).read().count("\n SG_ ")
        self.assertEqual(n, raw, "regex dropped a signal")
        self.assertEqual(cc.check(self.msgs), [])

    def test_check_detects_overlap(self):
        bad = {"M": dict(name="M", dlc=2, sig={"a": dict(start=0, size=8, signed=False, scale=1, offset=0, lo=0, hi=255),
                                                 "b": dict(start=4, size=8, signed=False, scale=1, offset=0, lo=0, hi=255)})}
        self.assertTrue(any("overlap" in p for p in cc.check(bad)))
        bad2 = {"M": dict(name="M", dlc=1, sig={"a": dict(start=4, size=8, signed=False, scale=1, offset=0, lo=0, hi=255)})}
        self.assertTrue(any("outside" in p for p in cc.check(bad2)))

    def test_crc_known_vector(self):
        # SAE J1850 CRC-8 check value for "123456789" is 0x4B
        self.assertEqual(cc.crc8_j1850(b"123456789"), 0x4B)

    def test_roundtrip_all_signals_all_msgs(self):
        rnd = random.Random(1)
        for m in self.msgs.values():
            for _ in range(300):
                vals = {}
                for n, s in m["sig"].items():
                    if n == "SB_CRC8":
                        continue
                    # any raw value whose physical value is inside the DBC range
                    raw_lo = math.ceil((s["lo"] - s["offset"]) / s["scale"] - 1e-9)
                    raw_hi = math.floor((s["hi"] - s["offset"]) / s["scale"] + 1e-9)
                    vals[n] = rnd.randint(raw_lo, raw_hi) * s["scale"] + s["offset"]
                fr = cc.pack(m, vals)
                self.assertEqual(len(fr), m["dlc"])
                back = cc.unpack(m, fr)
                for n, v in vals.items():
                    self.assertAlmostEqual(back[n], v, places=6, msg=f"{m['name']}.{n}")

    def test_edge_values(self):
        cmd = self.msgs["ASB_SB_Cmd"]
        for p in (0.0, 200.0, 0.01, 199.99):
            self.assertAlmostEqual(cc.unpack(cmd, cc.pack(cmd, {"SB_PressureReq": p}))["SB_PressureReq"], p, places=6)
        diag = self.msgs["ASB_SB_Diag"]
        for a in (-327.68, 327.67, -0.01, 0.0, 0.01):
            self.assertAlmostEqual(cc.unpack(diag, cc.pack(diag, {"SB_CrankAngle": a}))["SB_CrankAngle"], a, places=6)
        for tmp in (-40, 215, 25):
            self.assertAlmostEqual(cc.unpack(diag, cc.pack(diag, {"SB_MotorTemp": tmp}))["SB_MotorTemp"], tmp)
        # neighbouring signals not corrupted by negatives
        fr = cc.pack(diag, {"SB_CrankAngle": -1.0, "SB_MotorTorque": 3.0, "SB_FaultCode": 5})
        b = cc.unpack(diag, fr)
        self.assertAlmostEqual(b["SB_MotorTorque"], 3.0); self.assertEqual(b["SB_FaultCode"], 5)

    def test_out_of_range_rejected(self):
        """A value outside the DBC range must raise, never wrap into a valid-looking frame."""
        cmd = self.msgs["ASB_SB_Cmd"]
        status = self.msgs["ASB_SB_Status"]
        diag = self.msgs["ASB_SB_Diag"]
        cases = [(cmd, "SB_ModeReq", 5), (cmd, "SB_AliveCounter", 16), (cmd, "SB_PressureReq", -1.0),
                 (cmd, "SB_PressureReq", 250.0), (status, "SB_PressureFront", 300.0),
                 (diag, "SB_CrankAngle", 400.0), (diag, "SB_MotorTemp", -50)]
        for msg, sig, val in cases:
            with self.assertRaises(ValueError, msg=f"{sig}={val}"):
                cc.pack(msg, {sig: val})

    def test_unknown_signal(self):
        with self.assertRaises(KeyError):
            cc.pack(self.msgs["ASB_SB_Cmd"], {"nope": 1})

    def test_crc_in_cmd_and_single_bit_flips(self):
        cmd = self.msgs["ASB_SB_Cmd"]
        rnd = random.Random(7)
        for _ in range(50):
            fr = cc.pack(cmd, {"SB_PressureReq": rnd.uniform(0, 200), "SB_ModeReq": rnd.randint(0, 3),
                               "SB_RampLimit": rnd.randint(0, 255) * 10, "SB_AliveCounter": rnd.randint(0, 15)})
            self.assertEqual(cc.crc8_j1850(fr[:7]), fr[7])
            for bit in range(64):
                bad = bytearray(fr)
                bad[bit // 8] ^= 1 << (bit % 8)
                self.assertNotEqual(cc.crc8_j1850(bad[:7]), bad[7], f"undetected flip bit {bit}")

    def test_crc_only_cmd_and_status_flag(self):
        # Status declares SB_CRC8 too: pack must fill it
        st = self.msgs["ASB_SB_Status"]
        fr = cc.pack(st, {"SB_PressureFront": 40.0, "SB_State": 3})
        self.assertEqual(cc.crc8_j1850(fr[:7]), fr[7])
        diag = self.msgs["ASB_SB_Diag"]
        self.assertNotIn("SB_CRC8", diag["sig"])      # Diag has no CRC (by design?)

    def test_status_alive_bits_not_clobbered_by_crc(self):
        cmd = self.msgs["ASB_SB_Cmd"]
        fr = cc.pack(cmd, {"SB_AliveCounter": 15, "SB_CRC8": 0})
        self.assertEqual(cc.unpack(cmd, fr)["SB_AliveCounter"], 15)

    def test_unpack_wrong_length(self):
        with self.assertRaises(ValueError):
            cc.unpack(self.msgs["ASB_SB_Cmd"], bytes([1, 2]))

    def test_dbc_signal_ranges_vs_raw(self):
        for m in self.msgs.values():
            for n, s in m["sig"].items():
                cap_hi = ((1 << (s["size"] - (1 if s["signed"] else 0))) - 1) * s["scale"] + s["offset"]
                self.assertLessEqual(s["hi"], cap_hi + 1e-9, n)

    def test_rampLimit_default_zero_doc(self):
        # DBC comment says 0 selects default 600 bar/s: just verify 0 survives roundtrip
        cmd = self.msgs["ASB_SB_Cmd"]
        self.assertEqual(cc.unpack(cmd, cc.pack(cmd, {"SB_RampLimit": 0}))["SB_RampLimit"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
