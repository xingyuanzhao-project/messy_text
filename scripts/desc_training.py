"""
Descriptive statistics for training data label columns.

Produces two plots:
1. Missing rate per label, sorted by missing rate (descending).
2. Class imbalance — prevalence of the dominant class — per label,
   sorted by prevalence (descending).

Output: plots/descriptive_statistics/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "df_text_by_report.csv"
OUTPUT_DIR = ROOT / "plots" / "descriptive_statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

all_cols = list(df.columns)
start = all_cols.index("vic_grupo_social")
end = all_cols.index("proced_contactado")
labels = all_cols[start : end + 1]

label_df = df[labels]

# ── Plot 1: Missing rate ────────────────────────────────────────────────────

missing_rate = label_df.isna().mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(missing_rate)), missing_rate.values, color="steelblue", edgecolor="white")
ax.set_xticks(range(len(missing_rate)))
ax.set_xticklabels(missing_rate.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Missing rate")
ax.set_ylim(0, 1)
ax.set_title("Missing rate by label")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = OUTPUT_DIR / "missing_rate_by_label.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ── Plot 2: Class imbalance (stacked, shade encodes prevalence) ─────────────

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

TOP_N = 7  # categories beyond this are collapsed into "Other"
CMAP = plt.get_cmap("Blues")

# Per-label normalized distributions (non-missing rows only)
dist: dict[str, pd.Series] = {}
for col in labels:
    vc = label_df[col].dropna().value_counts(normalize=True)
    if len(vc) > TOP_N:
        top = vc.iloc[:TOP_N]
        rest = pd.Series({"Other": float(vc.iloc[TOP_N:].sum())})
        vc = pd.concat([top, rest])
    dist[col] = vc

# Sort labels by dominant class share (descending)
label_order = sorted(dist, key=lambda c: float(dist[c].iloc[0]), reverse=True)

fig, ax = plt.subplots(figsize=(12, 6))

for x, col in enumerate(label_order):
    vc = dist[col]
    n = len(vc)
    # Map rank 0 (dominant) → darkest shade, rank n-1 → lightest shade
    shades = np.linspace(0.85, 0.25, n)
    bottom = 0.0
    for shade, (cat, share) in zip(shades, vc.items()):
        ax.bar(
            x, share, bottom=bottom,
            color=CMAP(shade), edgecolor="white", linewidth=0.5,
        )
        bottom += share

ax.set_xticks(range(len(label_order)))
ax.set_xticklabels(label_order, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Class share (non-missing rows)")
ax.set_ylim(0, 1)
ax.set_title("Class distribution by label (sorted by dominant class share)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)

legend_handles = [
    Patch(facecolor=CMAP(0.85), label="Dominant class"),
    Patch(facecolor=CMAP(0.55), label="↓ less prevalent"),
    Patch(facecolor=CMAP(0.25), label="Rarest class"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

plt.tight_layout()
out = OUTPUT_DIR / "class_imbalance_by_label.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")
