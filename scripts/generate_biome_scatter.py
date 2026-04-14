"""
Generate biome-level precision-recall scatter plots for all three regions.
Shows "over/under-prediction" by biome: precision@1% vs recall@1%.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths to biome metrics (most recent runs)
REGIONS = {
    "South America": "/Users/gianluca/Desktop/thesis/code/outputs/south_america/results/spatial_generalisation/biome_metrics_20260406_081511.csv",
    "United States": "/Users/gianluca/Desktop/thesis/code/outputs/usa/results/spatial_generalisation/biome_metrics_20260406_082214.csv",
    "SE Asia": "/Users/gianluca/Desktop/thesis/code/outputs/se_asia/results/spatial_generalisation/biome_metrics_20260406_082851.csv",
}

# Short biome name map for clean labels
BIOME_SHORT = {
    "Deserts & Xeric Shrublands": "Deserts",
    "Flooded Grasslands & Savannas": "Flooded Grass.",
    "Mangroves": "Mangroves",
    "Mediterranean Forests, Woodlands & Scrub": "Mediter. Forests",
    "Montane Grasslands & Shrublands": "Montane Grass.",
    "Temperate Broadleaf & Mixed Forests": "Temp. Broadleaf",
    "Tropical & Subtropical Dry Broadleaf Forests": "Trop. Dry BL",
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": "Trop. Grass.",
    "Tropical & Subtropical Moist Broadleaf Forests": "Trop. Moist BL",
    "Temperate Conifer Forests": "Temp. Conifers",
    "Temperate Grasslands, Savannas & Shrublands": "Temp. Grass.",
    "Tropical & Subtropical Coniferous Forests": "Trop. Conifers",
    "Boreal Forests/Taiga": "Boreal",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Biome-level Prediction Quality: Precision@1\\% vs. Recall@1\\% (LightGBM test set)",
    fontsize=11, fontweight="bold", y=1.01
)

for ax, (region, path) in zip(axes, REGIONS.items()):
    if not os.path.exists(path):
        ax.set_title(f"{region}\n(data not found)")
        continue

    df = pd.read_csv(path)
    df = df[df["model"] == "LGBM"].copy()
    # Exclude biomes with very few positives (< 20) to avoid noise
    df = df[df["n_positives"] >= 20].copy()

    precision = df["precision_at_1pct"].values
    recall = df["recall_at_1pct"].values
    n_pos = df["n_positives"].values
    biomes = df["biome"].values

    # Bubble size scaled by log(n_positives), min size 50
    sizes = np.clip(np.log1p(n_pos) * 12, 40, 600)

    # Color by positive rate
    pos_rate = df["positive_rate"].values
    sc = ax.scatter(recall, precision, s=sizes, c=pos_rate, cmap="YlOrRd",
                    alpha=0.85, edgecolors="0.3", linewidths=0.5, zorder=3)

    # Label each biome
    for i, (r, p, b) in enumerate(zip(recall, precision, biomes)):
        short = BIOME_SHORT.get(b, b[:15])
        ax.annotate(short, (r, p), fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", zorder=4)

    # Diagonal "equal precision-recall" line for reference
    lim = [0, 1.0]
    ax.plot(lim, lim, "k--", linewidth=0.7, alpha=0.4, zorder=1, label="Precision = Recall")

    ax.set_xlabel("Recall @ top-1\\%", fontsize=9)
    ax.set_ylabel("Precision @ top-1\\%" if ax == axes[0] else "", fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(region, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, zorder=0)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Positive rate", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

# Add legend for bubble size
legend_elements = [
    plt.scatter([], [], s=np.log1p(500) * 12, c="grey", alpha=0.6, edgecolors="0.3", label="N=500"),
    plt.scatter([], [], s=np.log1p(5000) * 12, c="grey", alpha=0.6, edgecolors="0.3", label="N=5,000"),
    plt.scatter([], [], s=np.log1p(50000) * 12, c="grey", alpha=0.6, edgecolors="0.3", label="N=50,000"),
]
axes[1].legend(handles=legend_elements, title="N designations\n(bubble size)", fontsize=6.5,
               title_fontsize=7, loc="lower right", framealpha=0.8)

plt.tight_layout()

out_path = "/Users/gianluca/Desktop/thesis/code/outputs/south_america/results/model1_lgbm/biome_scatter_pr.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved: {out_path}")
plt.close()
