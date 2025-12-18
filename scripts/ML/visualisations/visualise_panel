#!/usr/bin/env python3
"""
Panel dataset diagnostic visualizations.

Inputs:
- Parquet panel file: `data/ml/merged_panel_final.parquet`

Process:
- Uses DuckDB to efficiently query the parquet file without loading full dataset into memory
- Generates diagnostic plots: time series, spatial maps, class separation, correlation matrix

Outputs:
- PNG figures saved to `outputs/Figures/panel_visualisation/`.
"""

import os
from pathlib import Path
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Set up paths
ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "outputs" / "Figures" / "panel_visualisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resolve input file path (prefer $SCRATCH if present)
SCRATCH_ROOT = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
candidate_paths = []
if SCRATCH_ROOT is not None:
    candidate_paths.append(SCRATCH_ROOT / "data" / "ml" / "merged_panel_final.parquet")
candidate_paths.append(ROOT_DIR / "data" / "ml" / "merged_panel_final.parquet")

PANEL_FILE = None
for cand in candidate_paths:
    if cand.exists():
        PANEL_FILE = cand
        break

if PANEL_FILE is None:
    raise FileNotFoundError("merged_panel_final.parquet not found in expected locations")


def configure_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB with sensible defaults."""
    is_euler = bool(os.environ.get("SCRATCH"))
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    num_threads = slurm_cpus if slurm_cpus > 0 else (48 if is_euler else 4)

    # Match DuckDB memory to Slurm allocation when present
    slurm_mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
    if slurm_mem_per_cpu_mb and slurm_cpus:
        total_mem_mb = int(slurm_mem_per_cpu_mb) * slurm_cpus
        memory_limit_gb = max(1, total_mem_mb // 1024)
    else:
        memory_limit_gb = 128 if is_euler else 16

    con.execute(f"SET threads={num_threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET memory_limit='{memory_limit_gb}GB'")

    temp_dir = os.environ.get("SCRATCH") or (ROOT_DIR / "temp")
    temp_dir_sql = str(temp_dir).replace("'", "''")
    con.execute(f"SET temp_directory='{temp_dir_sql}/duckdb_temp'")

    if is_euler:
        con.execute("PRAGMA max_temp_directory_size='200GB'")


def plot_time_series(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Create dual-axis time series plot: row count and transition_01 sum per year."""
    print("Creating time series plot...")
    
    escaped_path = str(parquet_path).replace("'", "''")
    query = f"""
    SELECT
        year,
        COUNT(*) AS row_count,
        SUM(transition_01) AS transitions
    FROM read_parquet('{escaped_path}')
    GROUP BY year
    ORDER BY year
    """
    
    df = con.execute(query).df()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: row count (line)
    color1 = 'tab:blue'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Risk Set Size (row count)', color=color1, fontsize=12)
    line1 = ax1.plot(df['year'], df['row_count'], color=color1, marker='o', linewidth=2, label='Risk Set Size')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: transitions (bars)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Number of Designations', color=color2, fontsize=12)
    bars = ax2.bar(df['year'], df['transitions'], color=color2, alpha=0.6, label='Designations')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    plt.title('Panel Dataset: Risk Set Size and Designations Over Time', fontsize=14, fontweight='bold', pad=20)
    
    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "time_series.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file.name}")
    plt.close()


def plot_spatial_map(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Create hexbin plot of spatial distribution of designations (transition_01 = 1)."""
    print("Creating spatial map...")
    
    escaped_path = str(parquet_path).replace("'", "''")
    query = f"""
    SELECT col, row
    FROM read_parquet('{escaped_path}')
    WHERE transition_01 = 1
    """
    
    df = con.execute(query).df()
    
    if len(df) == 0:
        print("   Warning: No rows with transition_01 = 1 found. Skipping spatial map.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Hexbin plot
    hb = ax.hexbin(df['col'], df['row'], gridsize=100, cmap='YlOrRd', mincnt=1)
    
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title('Spatial Distribution of Designations (transition_01 = 1)', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count of Designations', fontsize=12)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "spatial_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file.name}")
    plt.close()


def plot_class_separation(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Create boxplots for top 4 features, split by transition_01, using balanced sampling."""
    print("Creating class separation boxplots...")
    
    # Top 4 features
    top_features = ['dist_wdpa', 'HNTL_b1', 'NDVI_b1', 'deforestation_b1']
    
    escaped_path = str(parquet_path).replace("'", "''")
    
    # First, get counts of positives
    count_query = f"""
    SELECT 
        SUM(CASE WHEN transition_01 = 1 THEN 1 ELSE 0 END) AS pos_count,
        SUM(CASE WHEN transition_01 = 0 THEN 1 ELSE 0 END) AS neg_count
    FROM read_parquet('{escaped_path}')
    """
    
    counts = con.execute(count_query).fetchone()
    pos_count = counts[0]
    neg_count = counts[1]
    
    # Sample size: all positives + 5x positives of negatives
    sample_neg = min(pos_count * 5, neg_count)
    
    print(f"   Positive samples: {pos_count:,}, Negative samples: {sample_neg:,}")
    
    # Get feature columns - check which ones exist
    schema_query = f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}') LIMIT 0"
    schema_df = con.execute(schema_query).df()
    available_cols = set(schema_df['column_name'].tolist())
    
    # Filter to only features that exist
    existing_features = [f for f in top_features if f in available_cols]
    
    if not existing_features:
        print(f"   Warning: None of the requested features found. Available columns: {list(available_cols)[:10]}...")
        return
    
    # Build query for balanced sample
    # Use random sampling with ORDER BY RANDOM() LIMIT for negatives
    feature_list = ', '.join(existing_features)
    
    sample_query = f"""
    (
        SELECT {feature_list}, transition_01
        FROM read_parquet('{escaped_path}')
        WHERE transition_01 = 1
    )
    UNION ALL
    (
        SELECT {feature_list}, transition_01
        FROM read_parquet('{escaped_path}')
        WHERE transition_01 = 0
        ORDER BY random()
        LIMIT {sample_neg}
    )
    """
    
    df = con.execute(sample_query).df()
    
    # Create subplots
    n_features = len(existing_features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(existing_features):
        ax = axes[idx]
        
        # Prepare data for boxplot
        pos_data = df[df['transition_01'] == 1][feature].dropna()
        neg_data = df[df['transition_01'] == 0][feature].dropna()
        
        # Create boxplot
        box_data = [neg_data, pos_data]
        bp = ax.boxplot(box_data, labels=['No Designation (0)', 'Designation (1)'], patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(feature, fontsize=11)
        ax.set_title(f'Class Separation: {feature}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations (boxplot positions are at 1 and 2)
        y_max = ax.get_ylim()[1]
        ax.text(1, y_max * 0.98, f'n={len(neg_data):,}', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(2, y_max * 0.98, f'n={len(pos_data):,}', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplots if needed
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Class Separation: Top Features by Designation Status', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "class_separation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file.name}")
    plt.close()


def plot_correlation_matrix(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Create correlation matrix heatmap of numeric features on 1% random sample."""
    print("Creating correlation matrix...")
    
    escaped_path = str(parquet_path).replace("'", "''")
    
    # Get all numeric columns (exclude metadata columns)
    schema_query = f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}') LIMIT 0"
    schema_df = con.execute(schema_query).df()
    
    # Metadata columns to exclude
    exclude_cols = {'year', 'x', 'y', 'row', 'col', 'transition_01'}
    
    # Filter to numeric columns (excluding metadata)
    all_cols = schema_df['column_name'].tolist()
    numeric_cols = [col for col in all_cols if col not in exclude_cols]
    
    # Get 1% sample using random() < 0.01
    sample_query = f"""
    SELECT {', '.join(numeric_cols)}
    FROM read_parquet('{escaped_path}')
    WHERE random() < 0.01
    """
    
    df = con.execute(sample_query).df()
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Use a diverging colormap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', ax=ax)
    
    ax.set_title('Feature Correlation Matrix (1% Random Sample)', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file.name}")
    plt.close()


def main():
    """Main visualization function"""
    print("=" * 70)
    print("PANEL DATASET DIAGNOSTIC VISUALIZATIONS")
    print("=" * 70)
    print(f"Input file: {PANEL_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Initialize DuckDB
    con = duckdb.connect()
    configure_duckdb(con)
    
    try:
        # 1. Time series plot
        plot_time_series(con, PANEL_FILE)
        
        # 2. Spatial map
        plot_spatial_map(con, PANEL_FILE)
        
        # 3. Class separation boxplots
        plot_class_separation(con, PANEL_FILE)
        
        # 4. Correlation matrix
        plot_correlation_matrix(con, PANEL_FILE)
        
        print()
        print("=" * 70)
        print("VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"All plots saved to: {OUTPUT_DIR}")
        print("\nGenerated plots:")
        print("  1. time_series.png - Risk set size and designations over time")
        print("  2. spatial_map.png - Spatial distribution of designations")
        print("  3. class_separation.png - Feature distributions by class")
        print("  4. correlation_matrix.png - Feature correlation matrix")
        
    finally:
        con.close()


if __name__ == "__main__":
    main()

