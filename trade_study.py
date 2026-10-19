"""Weighted decision matrix for the service-brake actuator concepts.

Scores are 1 (poor) to 5 (excellent). Edit SCORES / WEIGHTS and re-run; the
table in trade_study_table.md and the chart are regenerated.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")                # every generated file goes here
os.makedirs(OUT, exist_ok=True)

CRITERIA = [
    # (key, label, weight %)
    ("override",  "Driver override is mechanical (zero added resistance, no software needed)", 25),
    ("failure",   "Failure behaviour (power loss, jam, comms loss)", 15),
    ("packaging", "Packaging: inside the RPE, out of the foot space, room left for EBS", 15),
    ("control",   "Control: stiffness, speed, torque margin, force sensing", 15),
    ("rules",     "Rules and brake-system integration risk", 10),
    ("build",     "Mass, cost, manufacturability", 10),
    ("team",      "Team integration: native CAN, maintenance, effort", 10),
]

CONCEPTS = {
    "A  Motor + spool + cable":             dict(override=5, failure=4, packaging=3, control=4, rules=4, build=4, team=4),
    "B  Linear actuator + lost-motion rod": dict(override=5, failure=3, packaging=3, control=2, rules=4, build=2, team=2),
    "C  Motor + crank pushing a roller":    dict(override=5, failure=4, packaging=2, control=4, rules=4, build=4, team=4),
    "D  Motor + crank + slotted pull-link": dict(override=5, failure=4, packaging=4, control=5, rules=4, build=4, team=4),
    "E  2nd master cylinder + shuttle valves": dict(override=4, failure=3, packaging=3, control=4, rules=2, build=2, team=3),
    "F  Pneumatic cylinder":                dict(override=4, failure=3, packaging=2, control=2, rules=3, build=2, team=2),
}


def total(scores):
    return sum(scores[k] * w for k, _, w in CRITERIA) / 100.0


def main():
    ranked = sorted(CONCEPTS.items(), key=lambda kv: -total(kv[1]))
    lines = ["| Criterion | Weight | " + " | ".join(name.split("  ")[0] for name, _ in ranked) + " |",
             "|---|---|" + "---|" * len(ranked)]
    for key, label, w in CRITERIA:
        lines.append(f"| {label} | {w} % | " + " | ".join(str(s[key]) for _, s in ranked) + " |")
    lines.append("| **Weighted score (out of 5)** | | " +
                 " | ".join(f"**{total(s):.2f}**" for _, s in ranked) + " |")
    lines.append("")
    lines.append("Concepts: " + "; ".join(name for name, _ in ranked))
    with open(os.path.join(OUT, "trade_study_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = [n for n, _ in ranked][::-1]
    vals = [total(s) for _, s in ranked][::-1]
    bars = ax.barh(names, vals, color=["#9aa5b1"] * (len(vals) - 1) + ["#d4a017"])
    for b, val in zip(bars, vals):
        ax.text(val + 0.03, b.get_y() + b.get_height() / 2, f"{val:.2f}", va="center")
    ax.set_xlim(0, 5)
    ax.set_xlabel("weighted score (out of 5)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trade_study_scores.png"), dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
