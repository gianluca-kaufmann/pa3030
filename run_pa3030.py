"""PA3030 entry point: forward projections and 30x30 scenarios.

    Configuration: level 4 of 4 -- scenarios CSV (region x model x allocation) + parameters CSV
    Code layout:   library -- tasks call the stage code in scripts/regions/shared/forward/
    Ownership:     self-contained

Each row of pa3030_scenarios.csv is one forward run: a region, a scored model and a 30x30
allocation mode (regional_pool or national_quota, see scenario_allocation.py).

The stage code reads its locations from PA3030_FORWARD_* environment variables. The tasks set
those from the parameters CSV. The tracked input_template/pa3030_parameters.csv has blank values.
Each machine fills in its own untracked copy at <project_dir>/input/pa3030_parameters.csv.
With outputs_root filled in (on Euler: the $SCRATCH outputs directory), the tasks read and
reuse the existing forward outputs there. With it blank, they write under this project's
intermediate/ directory.
"""
import os

import pandas as pd
import hazelbean as hb

from scripts.regions.shared.forward import run_region_forward


def build_task_tree(p):
    p.forward_task = p.add_task(forward)
    p.coverage_baseline_task = p.add_task(coverage_baseline, parent=p.forward_task)
    p.forward_features_task = p.add_task(forward_features, parent=p.forward_task)
    p.forward_scores_task = p.add_task(forward_scores, parent=p.forward_task)
    p.scenario_results_task = p.add_task(scenario_results, parent=p.forward_task)


def forward_dir_for(p, region):
    """Forward output root for one region, in the layout the stage code expects."""
    if p.outputs_root:
        return os.path.join(p.outputs_root, region, 'results', 'forward')
    return os.path.join(p.forward_dir, region)


def run_stage(p, region, stage, model_type, allocation=None):
    """Point the stage code at this project's locations, then run one stage."""
    os.environ['PA3030_FORWARD_OUTPUT_DIR'] = forward_dir_for(p, region)
    if p.ml_models_root:
        os.environ['PA3030_ML_MODELS_DIR'] = os.path.join(p.ml_models_root, region, 'ml', 'models')
    if allocation:
        os.environ['PA3030_FORWARD_ALLOCATION'] = allocation
    run_region_forward(region, stage=stage, model_type=model_type)


def forward(p):
    """Parent of the forward stages. Holds one output root per region."""
    p.regions = sorted(p.scenarios_df['region'].unique())


def coverage_baseline(p):
    """Protected and total area per region and country in 2024, and the km2 each target needs."""
    if p.run_this:
        for region in p.regions:
            if not hb.path_exists(os.path.join(forward_dir_for(p, region), 'coverage_baseline.json')):
                run_stage(p, region, 'coverage_baseline', 'lgbm')


def forward_features(p):
    """The 2024 predictor snapshot for every unprotected pixel."""
    if p.run_this:
        for region in p.regions:
            if not hb.path_exists(os.path.join(forward_dir_for(p, region), 'forward_features_2024.parquet')):
                run_stage(p, region, 'features', 'lgbm')


def forward_scores(p):
    """Calibrated designation probability per pixel, one file per region and model."""
    if p.run_this:
        for (region, model_type), _ in p.scenarios_df.groupby(['region', 'model_type']):
            if not hb.path_exists(os.path.join(forward_dir_for(p, region), model_type, 'forward_scored_2024.parquet')):
                run_stage(p, region, 'predict', model_type)


def scenario_results(p):
    """BAU, moderate and 30x30 selections with maps and breakdowns, one directory per scenario row."""
    if p.run_this:
        for _, row in p.scenarios_df.iterrows():
            out_dir = os.path.join(forward_dir_for(p, row['region']), row['model_type'], row['allocation'])
            if not hb.path_exists(os.path.join(out_dir, 'scenario_summary.json')):
                run_stage(p, row['region'], 'results', row['model_type'], row['allocation'])


def run_project(p):
    """Execute the forward projections against the ProjectFlow the caller configured.

    Reads p.scenario_definitions_filename, and optionally p.tasks_to_skip. Returns p.
    """
    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)

    # get_path finds a machine's own copy in input/ first, then the tracked input_template/.
    parameters_df = pd.read_csv(p.get_path('pa3030_parameters.csv'))
    for _, row in parameters_df.iterrows():
        setattr(p, row['key'], None if pd.isna(row['value']) else row['value'])

    p.scenarios_df = pd.read_csv(p.get_path(p.scenario_definitions_filename))

    p.L = hb.get_logger(p.project_name)
    hb.log('Created ProjectFlow object at ' + p.project_dir)

    p.execute()

    return p


if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='pa3030', run_mode='check')
    p.scenario_definitions_filename = 'pa3030_scenarios.csv'

    run_project(p)
