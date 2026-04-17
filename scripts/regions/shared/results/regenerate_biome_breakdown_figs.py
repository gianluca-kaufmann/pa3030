"""
Regenerate biome_breakdown.pdf for all three regions from existing biome_breakdown.csv files.
Right panel now shows Lift@1% (Precision@1% / positive_rate) instead of raw Precision@1%.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

SHORT_BIOME = {
    'Tropical & Subtropical Moist Broadleaf Forests':         'Trop. Moist Broadleaf',
    'Tropical & Subtropical Dry Broadleaf Forests':           'Trop. Dry Broadleaf',
    'Tropical & Subtropical Grasslands, Savannas & Shrublands': 'Trop. Grasslands/Savannas',
    'Tropical & Subtropical Coniferous Forests':               'Trop. Coniferous',
    'Temperate Broadleaf & Mixed Forests':                     'Temp. Broadleaf',
    'Temperate Grasslands, Savannas & Shrublands':             'Temp. Grasslands/Savannas',
    'Flooded Grasslands & Savannas':                           'Flooded Grasslands',
    'Montane Grasslands & Shrublands':                         'Montane Grasslands',
    'Mediterranean Forests, Woodlands & Scrub':                'Mediterranean',
    'Deserts & Xeric Shrublands':                              'Deserts & Xeric',
    'Mangroves':                                               'Mangroves',
    'Tundra':                                                  'Tundra',
    'Temperate Conifer Forests':                               'Temp. Conifer',
}

REGIONS = [
    ('south_america', 'model1_lgbm', 'South America', 'LGBM'),
    ('usa',           'model2_lgbm', 'United States', 'LGBM'),
    ('se_asia',       'model3_lgbm', 'Southeast Asia', 'LGBM'),
]

FONTSIZE_TITLE  = 10
FONTSIZE_LABEL  = 9
FONTSIZE_STATS  = 7.5
FONTSIZE_LEGEND = 7.5

for region, model_dir, region_label, model_type in REGIONS:
    csv_path = REPO_ROOT / 'outputs' / region / 'results' / model_dir / 'biome_breakdown.csv'
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path).sort_values('ROC_AUC', ascending=True)
    df['Lift_at_1pct'] = (df['Precision_at_1pct'] / (df['prevalence_pct'] / 100)).round(2)

    labels    = [SHORT_BIOME.get(b, b[:30]) for b in df['BIOME_NAME']]
    roc_vals  = df['ROC_AUC'].tolist()
    lift_vals = df['Lift_at_1pct'].tolist()
    n_vals    = df['n_pixel_years'].tolist()
    y_pos     = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.55)))

    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, roc_vals, color='#4A90D9', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax1.set_xlabel('ROC-AUC', fontsize=FONTSIZE_LABEL)
    ax1.set_title('ROC-AUC by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=FONTSIZE_LEGEND)
    ax1.grid(True, axis='x', alpha=0.3)
    for bar, val, n in zip(bars1, roc_vals, n_vals):
        ax1.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}  (n={n:,})', va='center', ha='left', fontsize=FONTSIZE_STATS)

    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, lift_vals, color='#E07B54', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax2.set_xlabel('Lift@1% (×)', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Lift@1% by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (1×)')
    lift_max = max(lift_vals) if lift_vals else 1
    ax2.set_xlim(0, lift_max * 1.3)
    ax2.legend(fontsize=FONTSIZE_LEGEND)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, lift_vals):
        ax2.text(min(val + lift_max * 0.01, lift_max * 1.25),
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}×', va='center', ha='left', fontsize=FONTSIZE_STATS)

    plt.suptitle(f'{region_label} ({model_type}) Performance by Biome (GSN Terrestrial Ecoregions)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = csv_path.parent / 'biome_breakdown.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
