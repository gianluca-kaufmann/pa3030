"""
Regenerate biome_breakdown.pdf for all three regions from existing CSV files.
Left panel: PR-AUC (primary metric, sourced from latest biome_metrics_*.csv in
spatial_generalisation/). Right panel: Lift@1%. Sorted by PR-AUC ascending.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

SHORT_BIOME = {
    'Tropical & Subtropical Moist Broadleaf Forests':           'Trop. Moist Broadleaf',
    'Tropical & Subtropical Dry Broadleaf Forests':             'Trop. Dry Broadleaf',
    'Tropical & Subtropical Grasslands, Savannas & Shrublands': 'Trop. Grasslands/Savannas',
    'Tropical & Subtropical Coniferous Forests':                'Trop. Coniferous',
    'Temperate Broadleaf & Mixed Forests':                      'Temp. Broadleaf',
    'Temperate Grasslands, Savannas & Shrublands':              'Temp. Grasslands/Savannas',
    'Flooded Grasslands & Savannas':                            'Flooded Grasslands',
    'Montane Grasslands & Shrublands':                          'Montane Grasslands',
    'Mediterranean Forests, Woodlands & Scrub':                 'Mediterranean',
    'Deserts & Xeric Shrublands':                               'Deserts & Xeric',
    'Mangroves':                                                'Mangroves',
    'Tundra':                                                   'Tundra',
    'Temperate Conifer Forests':                                'Temp. Conifer',
}

REGIONS = [
    ('south_america', 'model1_rf', 'South America', 'RF'),
    ('usa',           'model2_rf', 'United States', 'RF'),
    ('se_asia',       'model3_rf', 'Southeast Asia', 'RF'),
]

FONTSIZE_TITLE  = 10
FONTSIZE_LABEL  = 9
FONTSIZE_STATS  = 7.5
FONTSIZE_LEGEND = 7.5

# Biomes that are artifacts or have pathological prevalence rates (>10%) and
# should be excluded from the per-biome figure to avoid distorting the scale.
EXCLUDE_BIOMES = {'N/A', 'Unknown'}


def load_pr_auc(sg_dir: Path, model_type: str) -> dict[str, float]:
    """Return {biome_name: pr_auc} from the latest biome_metrics_*.csv."""
    csvs = sorted(sg_dir.glob('biome_metrics_*.csv'))
    if not csvs:
        print(f"  WARNING: no biome_metrics CSV found in {sg_dir}")
        return {}
    latest = csvs[-1]
    print(f"  PR-AUC source: {latest.name}")
    df = pd.read_csv(latest)
    df = df[(df['model'] == model_type) & (~df['biome'].isin(EXCLUDE_BIOMES | {'ALL'}))]
    return dict(zip(df['biome'], df['pr_auc']))


for region, model_dir, region_label, model_type in REGIONS:
    csv_path = REPO_ROOT / 'outputs' / region / 'results' / model_dir / 'biome_breakdown.csv'
    sg_dir   = REPO_ROOT / 'outputs' / region / 'results' / 'spatial_generalisation'

    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    df['Lift_at_1pct'] = (df['Precision_at_1pct'] / (df['prevalence_pct'] / 100)).round(2)

    # Merge PR-AUC from spatial_generalisation (computed on full test-set scores)
    pr_auc_map = load_pr_auc(sg_dir, model_type)
    df['PR_AUC'] = df['BIOME_NAME'].map(pr_auc_map)

    # Drop rows without PR-AUC (N/A, Unknown, biomes with no positives) and NaN names
    df = df.dropna(subset=['BIOME_NAME'])
    df = df[~df['BIOME_NAME'].isin(EXCLUDE_BIOMES)].dropna(subset=['PR_AUC'])
    df = df.sort_values('PR_AUC', ascending=True).reset_index(drop=True)

    labels    = [SHORT_BIOME.get(b, b[:30]) for b in df['BIOME_NAME']]
    pr_vals   = df['PR_AUC'].tolist()
    lift_vals = df['Lift_at_1pct'].tolist()
    n_vals    = df['n_pixel_years'].tolist()
    y_pos     = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.55)))

    # --- Left panel: PR-AUC ---
    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, pr_vals, color='#4A90D9', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax1.set_xlabel('PR-AUC', fontsize=FONTSIZE_LABEL)
    ax1.set_title('PR-AUC by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.text(0.01, 1.01, '(a)', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', va='bottom', ha='left')
    ax1.set_xlim(0, 1)
    ax1.grid(True, axis='x', alpha=0.3)
    for bar, val, n in zip(bars1, pr_vals, n_vals):
        ax1.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}  (n={n:,})', va='center', ha='left', fontsize=FONTSIZE_STATS)

    # --- Right panel: Lift@1% ---
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, lift_vals, color='#E07B54', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax2.set_xlabel('Lift@1% (×)', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Lift@1% by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.text(0.01, 1.01, '(b)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='bottom', ha='left')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (1×)')
    lift_max = max(lift_vals) if lift_vals else 1
    ax2.set_xlim(0, lift_max * 1.3)
    ax2.legend(fontsize=FONTSIZE_LEGEND)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, lift_vals):
        ax2.text(min(val + lift_max * 0.01, lift_max * 1.25),
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}×', va='center', ha='left', fontsize=FONTSIZE_STATS)

    plt.tight_layout()

    out_path = csv_path.parent / 'biome_breakdown.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
