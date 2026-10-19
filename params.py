"""Reference-car parameters for the WFR driverless service brake project.

Every number here is an ASSUMPTION for a typical FSAE EV (no WFR data was
available). Each entry carries a short source note so the team can swap in
measured values. The scripts read their car numbers from here; a few
secondary values (the 0.45 warm pad mu, the 0.8 g everyday stop, the pedal
geometry repeated in hand_calcs.py) are still written in the scripts that use
them, so check those too when swapping in measured data.
"""

G = 9.81

CAR = {
    "mass_kg":            (300.0, "car + driver/autonomy kit envelope; typical FSAE EV 230-260 kg + 70 kg"),
    "cg_height_m":        (0.28,  "typical FSAE CG height"),
    "wheelbase_m":        (1.55,  "FSAE min 1525 mm; typical 1.53-1.60 m"),
    "front_static_frac":  (0.47,  "typical 45-48 % front"),
    "tyre_radius_m":      (0.228, "10 in wheel, 18 in tyre, loaded"),
    "tyre_mu":            (1.6,   "peak, warm Hoosier slick, no aero (low-speed DV case)"),
}

BRAKES = {
    "pad_mu":             (0.40,  "conservative (cold); 0.45 typical warm"),
    "disc_reff_front_m":  (0.085, "effective radius, ~200 mm disc"),
    "disc_reff_rear_m":   (0.085, "effective radius, ~200 mm disc"),
    "caliper_front_piston_d_m": (0.0254, "opposed 4-piston, 2 pistons per side"),
    "caliper_front_pistons_per_side": (2, ""),
    "caliper_rear_piston_d_m":  (0.0254, "opposed 2-piston, 1 piston per side"),
    "caliper_rear_pistons_per_side":  (1, ""),
    "mc_bore_front_m":    (0.625 * 0.0254, "Tilton 78-series 5/8 in"),
    "mc_bore_rear_m":     (0.625 * 0.0254, "Tilton 78-series 5/8 in"),
}

PEDAL = {
    "pad_radius_m":       (0.225, "pivot to foot-pad centre"),
    "balance_bar_radius_m": (0.045, "pivot to balance-bar line -> 5:1 pedal ratio"),
    "mc_dead_stroke_mm":  (1.5,   "port closing + pad knock-back, assumed"),
    "mc_stroke_per_bar_mm": (0.10, "line/caliper/pad compliance, braided lines, assumed"),
    "max_manual_rotation_deg": (25.0, "full pedal sweep incl. over-travel to the BOTS trip, assumed"),
    "return_spring_N":    (40.0,  "pedal return spring at the pad, assumed"),
}

DESIGN = {
    "design_decel_g":     (1.6,   "actuator must reach lock-up: decel = tyre mu"),
    "actuator_margin":    (1.25,  "torque margin on top of the design point"),
    "rise_time_s":        (0.15,  "10-90 % pressure rise target for a full step"),
    "pressure_accuracy_bar": (2.0, "steady-state tracking target"),
}


def v(table, key):
    """Value without its note."""
    return table[key][0]
