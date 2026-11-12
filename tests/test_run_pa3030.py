import os

import hazelbean as hb

import run_pa3030

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_TASKS = ['coverage_baseline', 'forward_features', 'forward_scores', 'scenario_results']


def test_task_tree_reads_template_and_places_outputs(tmp_path):
    p = hb.ProjectFlow(project_dir=str(tmp_path / 'pa3030_test'), run_mode='check')
    p.input_template_dir = os.path.join(REPO_ROOT, 'input_template')
    p.scenario_definitions_filename = 'pa3030_scenarios.csv'
    p.tasks_to_skip = STAGE_TASKS
    run_pa3030.run_project(p)

    assert p.regions == ['south_america']
    assert set(p.scenarios_df['allocation']) == {'national_quota', 'regional_pool'}
    assert p.outputs_root is None
    assert run_pa3030.forward_dir_for(p, 'south_america') == os.path.join(p.forward_dir, 'south_america')

    p.outputs_root = '/cluster/scratch/user/outputs'
    assert run_pa3030.forward_dir_for(p, 'south_america') == '/cluster/scratch/user/outputs/south_america/results/forward'
