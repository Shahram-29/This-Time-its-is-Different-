"""
10_pipeline_diagram.py
One-picture overview of the whole project (used in the README and the project handbook).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
boxes = [
    (0.2, 4.2, "1. DATA  (01)", "Yahoo Finance daily closes\nISEQ · S&P 500 · DAX (+FTSE)\n2003 – Aug 2023", "#dbe9f6"),
    (3.6, 4.2, "2. FEATURES  (01)", "returns, squared returns\n5-day realised variance target\nno look-ahead alignment", "#dbe9f6"),
    (7.0, 4.2, "3. PRE-REGISTRATION", "hypotheses H1–H4 fixed\nbefore any model is run\n(+ Amendment A1, 30 Sep 2026)", "#fff2cc"),
    (0.2, 2.1, "4. BENCHMARKS  (04)", "GARCH(1,1) · HAR\nannual walk-forward refits", "#e2f0d9"),
    (3.6, 2.1, "5. LSTM  (05)", "tuned once, then 17 yearly refits\n× 5 seeds = 85 models", "#e2f0d9"),
    (7.0, 2.1, "6. EVALUATION  (06)", "QLIKE · Diebold–Mariano (H1)\nfrozen models (H3)", "#fce4d6"),
    (0.2, 0.0, "7. SHAP  (07)", "which market drives each forecast?\nchannel shares, lags (H2, H4)", "#fce4d6"),
    (3.6, 0.0, "8. CHECKS  (08; 11–12)", "robustness: FTSE for DAX · look-back 66\nParkinson · post-2023\nexploratory: spillovers vs SHAP\nvolatility paradox", "#fce4d6"),
    (7.0, 0.0, "9. REPORT  (09, README)", "tables and figures\n→ dissertation chapters", "#ededed"),
]
fig, ax = plt.subplots(figsize=(12, 6.4))
for x, y, title, body, col in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.6, boxstyle="round,pad=0.05", fc=col, ec="#555", lw=1))
    ax.text(x + 1.5, y + 1.25, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x + 1.5, y + 0.55, body, ha="center", va="center", fontsize=8.5)
arrow = dict(arrowstyle="-|>", color="#333", lw=1.2)
for (x1, y1), (x2, y2) in [((3.2, 5.0), (3.6, 5.0)), ((6.6, 5.0), (7.0, 5.0)), ((8.5, 4.2), (1.7, 3.7)),
                           ((3.2, 2.9), (3.6, 2.9)), ((6.6, 2.9), (7.0, 2.9)), ((8.5, 2.1), (1.7, 1.6)),
                           ((3.2, 0.8), (3.6, 0.8)), ((6.6, 0.8), (7.0, 0.8))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow)
ax.set_xlim(0, 10.2)
ax.set_ylim(-0.2, 6.0)
ax.axis("off")
ax.set_title("This Time Is Different? — project pipeline (numbers = scripts in code/)", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "project_pipeline.png", dpi=150)
print("Saved figures/project_pipeline.png")
