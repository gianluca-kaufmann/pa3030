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
from scipy.interpolate import PchipInterpolator

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
    # Colorblind-friendly (Okabe–Ito-ish) and more distinct than purple/red/blue
    "South America": "#E69F00",   # orange / gold
    "United States": "#0072B2",   # blue
    "SE Asia":       "#7B2CBF",   # violet
}

THRESHOLDS = [1, 5, 10]          # top-K% values with real data points
X_SMOOTH = np.linspace(1, 100, 400)   # extended x range for smooth curves

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

# X-axis shape: expand 1–10% (where the action is), compress 10–100% tail.
K_BREAK = 10.0
TAIL_COMPRESS = 0.18  # smaller = more compression of 10–100%

def _k_forward(x):
    x = np.asarray(x)
    return np.where(x <= K_BREAK, x, K_BREAK + (x - K_BREAK) * TAIL_COMPRESS)

def _k_inverse(x):
    x = np.asarray(x)
    return np.where(x <= K_BREAK, x, K_BREAK + (x - K_BREAK) / TAIL_COMPRESS)

ax.set_xscale("function", functions=(_k_forward, _k_inverse))

lgbm_keys = [k for k, v in MODELS.items() if v["algo"] == "LGBM"]
rf_keys   = [k for k, v in MODELS.items() if v["algo"] == "RF"]

def smooth_precision(m):
    """PCHIP interpolation through 3 data points + baseline anchor at 100%."""
    xs = np.array([1, 5, 10, 100])
    ys = np.array([m["p1"], m["p5"], m["p10"], m["base"]])
    interp = PchipInterpolator(xs, ys)
    y_smooth = interp(X_SMOOTH)
    return np.clip(y_smooth, 0, None)

# Plot LGBM — smooth curve + dots at actual data points
for key in lgbm_keys:
    m = MODELS[key]
    col = REGION_COLORS[m["region"]]
    ax.plot(X_SMOOTH, smooth_precision(m), "-", color=col, linewidth=0.75, zorder=4)
    ax.plot(THRESHOLDS, [m["p1"], m["p5"], m["p10"]], "o",
            color=col, markersize=4.2, zorder=5)

# Plot RF — smooth dashed curve + square dots at actual data points
for key in rf_keys:
    m = MODELS[key]
    col = REGION_COLORS[m["region"]]
    ax.plot(X_SMOOTH, smooth_precision(m), "--", color=col, linewidth=0.55,
            alpha=0.65, zorder=3)
    ax.plot(THRESHOLDS, [m["p1"], m["p5"], m["p10"]], "s",
            color=col, markersize=3.4, alpha=0.65, zorder=4)

ax.set_xlabel("Top-$K$\\% risk zone", fontsize=9.5)
ax.set_ylabel("Precision @ top-$K$\\%", fontsize=9.5)
ax.set_xlim(1, 100)
ax.set_xticks([1, 5, 10, 25, 50, 100])
ax.set_xticklabels(["1\\%", "5\\%", "10\\%", "25\\%", "50\\%", "100\\%"], fontsize=8)
ax.set_ylim(-0.02, 0.72)
ax.grid(True, alpha=0.25)
ax.set_title("(a) Targeting Precision at Different Risk Thresholds",
             fontsize=9.5, fontweight="bold", pad=8)

# Custom legend (LightGBM solid, RF dashed, colors by region)
region_handles = [
    mlines.Line2D([], [], color=col, linewidth=0.9, label=reg)
    for reg, col in REGION_COLORS.items()
]
algo_handles = [
    mlines.Line2D([], [], color="0.3", linestyle="-",  linewidth=0.9, marker="o",
                  markersize=5, label="LightGBM"),
    mlines.Line2D([], [], color="0.3", linestyle="--", linewidth=0.65, marker="s",
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
