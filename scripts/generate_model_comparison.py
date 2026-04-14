"""
Cross-model performance comparison figure.

Two complementary panels:
  Left  — Precision@K% threshold decay profiles (3 LGBM models)
  Right — ROC-AUC vs Lift@1% scatter for all 6 models (3 regions × 2 algorithms)

The left panel tells the policy story: how does targeting accuracy degrade as
you widen the risk zone from 1% to 10%?  The right panel situates each model in
the discrimination–targeting space and reveals the USA's anomalous regime (near-
perfect ROC-AUC but near-zero precision due to extreme class imbalance).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Data (read from metrics_table.csv outputs)
# ---------------------------------------------------------------------------
MODELS = {
    "SA LightGBM": {
        "roc":     0.8968,
        "lift1":   66.81,
        "lift5":   14.54,
        "lift10":  7.56,
        "p1":  0.5324, "p5":  0.1159, "p10": 0.0602,
        "r1":  0.6681, "r5":  0.7271, "r10": 0.7560,
        "base": 0.007969,
        "region": "South America",
        "algo":   "LGBM",
    },
    "SA RF": {
        "roc":     0.9268,
        "lift1":   67.59,
        "p1":  0.5386, "p5":  0.1214, "p10": 0.0643,
        "r1":  0.6759, "r5":  0.7616, "r10": 0.8071,
        "base": 0.007969,
        "region": "South America",
        "algo":   "RF",
    },
    "USA LightGBM": {
        "roc":     0.9467,
        "lift1":   81.40,
        "lift5":   17.94,
        "lift10":  9.03,
        "p1":  0.0327, "p5":  0.00722, "p10": 0.00363,
        "r1":  0.8140, "r5":  0.8972,  "r10": 0.9032,
        "base": 0.000402,
        "region": "United States",
        "algo":   "LGBM",
    },
    "USA RF": {
        "roc":     0.9989,
        "lift1":   99.55,
        "p1":  0.0400,  "p5":  0.00801, "p10": 0.00401,
        "r1":  0.9955,  "r5":  0.9963,  "r10": 0.9980,
        "base": 0.000402,
        "region": "United States",
        "algo":   "RF",
    },
    "SEA LightGBM": {
        "roc":     0.9932,
        "lift1":   94.20,
        "lift5":   19.21,
        "lift10":  9.73,
        "p1":  0.5613, "p5":  0.1145, "p10": 0.0580,
        "r1":  0.9420, "r5":  0.9607, "r10": 0.9732,
        "base": 0.005959,
        "region": "SE Asia",
        "algo":   "LGBM",
    },
    "SEA RF": {
        "roc":     0.9862,
        "lift1":   90.45,
        "p1":  0.5390, "p5":  0.1144, "p10": 0.0577,
        "r1":  0.9045, "r5":  0.9601, "r10": 0.9685,
        "base": 0.005959,
        "region": "SE Asia",
        "algo":   "RF",
    },
}

REGION_COLORS = {
    "South America": "#2166ac",   # blue
    "United States": "#d6604d",   # red-orange
    "SE Asia":       "#4dac26",   # green
}

THRESHOLDS = [1, 5, 10]          # top-K% values

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(
    1, 2,
    figsize=(11, 4.5),
    gridspec_kw={"width_ratios": [1.35, 1]},
)
fig.subplots_adjust(wspace=0.36)

# ---- Panel A: Precision-at-threshold profiles ----------------------------
ax = ax_left

# Plot LGBM lines (solid, thicker) and RF dashed (thinner)
lgbm_keys = [k for k, v in MODELS.items() if v["algo"] == "LGBM"]
rf_keys   = [k for k, v in MODELS.items() if v["algo"] == "RF"]

for key in lgbm_keys:
    m = MODELS[key]
    col = REGION_COLORS[m["region"]]
    ys = [m["p1"], m["p5"], m["p10"]]
    ax.plot(THRESHOLDS, ys, "o-", color=col, linewidth=2.2, markersize=6,
            label=m["region"] + " – LightGBM", zorder=4)

for key in rf_keys:
    m = MODELS[key]
    col = REGION_COLORS[m["region"]]
    ys = [m["p1"], m["p5"], m["p10"]]
    ax.plot(THRESHOLDS, ys, "s--", color=col, linewidth=1.2, markersize=5,
            alpha=0.65, label=m["region"] + " – RF", zorder=3)

# Baseline precision lines (horizontal dotted)
for region in ["South America", "SE Asia", "United States"]:
    base = [v["base"] for v in MODELS.values() if v["region"] == region][0]
    col  = REGION_COLORS[region]
    ax.axhline(base, color=col, linestyle=":", linewidth=0.8, alpha=0.45)

# Annotation for USA baseline
ax.annotate(
    "USA baseline\n= 0.04%",
    xy=(10, 0.0035), xytext=(7.5, 0.045),
    fontsize=7, color=REGION_COLORS["United States"],
    arrowprops=dict(arrowstyle="->", color=REGION_COLORS["United States"],
                    lw=0.8),
    ha="center",
)

ax.set_xlabel("Top-$K$\\% risk zone", fontsize=9.5)
ax.set_ylabel("Precision @ top-$K$\\%", fontsize=9.5)
ax.set_xticks(THRESHOLDS)
ax.set_xticklabels(["1\\%", "5\\%", "10\\%"], fontsize=9)
ax.set_ylim(-0.02, 0.72)
ax.grid(True, alpha=0.25)
ax.set_title("(a) Targeting Precision at Different Risk Thresholds",
             fontsize=9.5, fontweight="bold", pad=8)

# Custom legend (LightGBM solid, RF dashed, colors by region)
region_handles = [
    mlines.Line2D([], [], color=col, linewidth=2, label=reg)
    for reg, col in REGION_COLORS.items()
]
algo_handles = [
    mlines.Line2D([], [], color="0.3", linestyle="-",  linewidth=2, marker="o",
                  markersize=5, label="LightGBM"),
    mlines.Line2D([], [], color="0.3", linestyle="--", linewidth=1.2, marker="s",
                  markersize=4, alpha=0.7, label="Random Forest"),
]
leg1 = ax.legend(handles=region_handles, loc="upper right",
                  fontsize=7.5, title="Region", title_fontsize=8,
                  framealpha=0.85, handlelength=1.5)
ax.add_artist(leg1)
ax.legend(handles=algo_handles, loc="center right",
          fontsize=7.5, title="Algorithm", title_fontsize=8,
          framealpha=0.85, handlelength=1.8, bbox_to_anchor=(1.0, 0.45))

# ---- Panel B: ROC-AUC vs Lift@1% scatter --------------------------------
ax = ax_right

for key, m in MODELS.items():
    col    = REGION_COLORS[m["region"]]
    marker = "o" if m["algo"] == "LGBM" else "s"
    size   = 120 if m["algo"] == "LGBM" else 80
    lw     = 1.6 if m["algo"] == "LGBM" else 1.0
    # Encode class difficulty via bubble edge thickness: thicker = rarer class
    log_difficulty = -np.log10(m["base"])   # higher = rarer = harder
    edge_lw = 0.5 + 1.2 * (log_difficulty - 2) / 1.5   # normalise 0.002 → 0.0004

    ax.scatter(
        m["roc"], m["lift1"],
        s=size, c=col, marker=marker,
        edgecolors="white", linewidths=1.0,
        alpha=0.92, zorder=4,
    )
    # Short label
    short = key.replace(" LightGBM", " LGB").replace(" RF", " RF")
    ax.annotate(
        short,
        (m["roc"], m["lift1"]),
        textcoords="offset points",
        xytext=(5, 4) if m["algo"] == "LGBM" else (5, -9),
        fontsize=7, color=col, fontweight="bold",
    )

# Formatting
ax.set_xlabel("ROC-AUC", fontsize=9.5)
ax.set_ylabel("Lift @ top-1\\%", fontsize=9.5)
ax.set_xlim(0.855, 1.01)
ax.set_ylim(55, 108)
ax.grid(True, alpha=0.25)
ax.set_title(r"(b) Discrimination vs.\ Targeting Effectiveness",
             fontsize=9.5, fontweight="bold", pad=8)

# Region colour patches
patch_handles = [
    mpatches.Patch(color=col, label=reg)
    for reg, col in REGION_COLORS.items()
]
algo_handles2 = [
    mlines.Line2D([], [], marker="o", color="0.3", linestyle="None",
                  markersize=7, label="LightGBM"),
    mlines.Line2D([], [], marker="s", color="0.3", linestyle="None",
                  markersize=6, label="RF"),
]
leg3 = ax.legend(handles=patch_handles, fontsize=7.5, loc="lower right",
                  title="Region", title_fontsize=8, framealpha=0.85)
ax.add_artist(leg3)
ax.legend(handles=algo_handles2, fontsize=7.5, loc="lower left",
          title="Algorithm", title_fontsize=8, framealpha=0.85)

# ---------------------------------------------------------------------------
out_path = ("/Users/gianluca/Desktop/thesis/code/outputs/south_america/"
            "results/model1_lgbm/model_comparison_landscape.pdf")
plt.savefig(out_path, bbox_inches="tight", dpi=200)
print(f"Saved: {out_path}")
plt.close()
