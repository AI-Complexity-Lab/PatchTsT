import numpy as np
from pathlib import Path
import yaml

from utils import pickle_load, pickle_save
from eval import get_test_results_meta


def shift_results(results, region2shift):
    output_results = results.copy()
    output_test_results = output_results['test_results']
    regions, aheads, CLs = get_test_results_meta(output_test_results)
    for region, shift_val in region2shift.items():
        for ahead in aheads:
            y_pred, CIs = output_test_results[region][ahead-1]
            y_pred = max(0, y_pred + shift_val)
            for cl in CLs:
                CIs[cl] = max(0, CIs[cl] + shift_val)
            output_test_results[region][ahead-1] = (y_pred, CIs)
    output_results['test_results'] = output_test_results
    return output_results


def aggregate_test_results(default_exp_result, alter_exp_results, region2exp):
    output_results = default_exp_result.copy()
    output_test_results = output_results['test_results']
    for region, exp in region2exp.items():
        output_test_results[region] = alter_exp_results[exp]['test_results'][region]
    output_results['test_results'] = output_test_results
    return output_results


def load_exp_res(exp_id, seed):
    dir_path = Path(f'../../results/{exp_id}')
    results = pickle_load(dir_path / f'results_{seed}.pickle')
    return results


if __name__ == '__main__':
    params = {}
    with open('../../setup/postprocess.yaml', 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    exp_ids = params['exp_ids']
    seeds = params['seeds']
    output_exp = int(params['output_exp'])
    region2exp = params['region2exp']
    region2shift = params['region2shift']
    
    # load experiment results
    default_exp_result = load_exp_res(exp_ids[0], seeds[0])
    alter_exp_results = {}
    for i in range(len(exp_ids)-1):
        alter_exp_results[exp_ids[i+1]] = load_exp_res(exp_ids[i+1], seeds[i+1])
    
    # combine and shift
    output_exp_result = aggregate_test_results(default_exp_result, alter_exp_results, region2exp)
    output_exp_result = shift_results(output_exp_result, region2shift)
    output_exp_result['params'] = {**output_exp_result['params'], **params}
    
    exp_dir_path = Path('../../results/') / f'{output_exp}'
    if not exp_dir_path.exists():
        exp_dir_path.mkdir()
    pickle_save(exp_dir_path / 'results.pickle', output_exp_result)
    
    