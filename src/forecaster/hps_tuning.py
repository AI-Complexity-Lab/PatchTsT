# Hyperparameter tuning. Use the predefined hyperparameters from setup/hpst.yaml. Select the one with best performance and save to results/hpst/expid/grid_search.pickle

"""
The saved results is in the following format: 
    id (corresponds to one set of hyperparameters) -> 
           set of hyperparameters
           value of performance
"""

import yaml
import random
import numpy as np
import copy

from online_training import run_online_training, get_params
from aci import prepare_scores
from utils import pickle_save
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_mse(y_pred, y_true):
    tmp0 = np.array(y_pred)
    tmp1 = np.array(y_true)
    return np.mean((tmp0 - tmp1)**2)


def test_one_paramset(params):
    # evaluate mse in normalized scale
    params['true_scale'] = False
    results = run_online_training(params)
    # evaluate the results. One concern is that the first performance in beginning weeks might be different from those in the end due to possible temporal distribution shift. So I take a window which includes only the most recent ten weeks.
    base_pred = results['base_pred']
    regions = list(base_pred[0][0].keys())
    aheads = params['aheads']
    mean_mse = []
    for region in regions:
        for ahead in aheads:
            _, y_preds, y_trues = prepare_scores(base_pred, region, ahead)
            y_preds = y_preds[-10:]
            y_trues = y_trues[-10:]
            current_mse = calculate_mse(y_preds, y_trues)
            mean_mse.append(current_mse)
    mean_mse = np.mean(mean_mse)
    test_result = {
        'mean_mse': mean_mse,
        'base_pred': base_pred,
    }
    return test_result
            


def check_hps(hpst_params, params):
    for key in hpst_params['hps']:
        if key not in init_params:
            print(key)
        else:
            cur_dict = hpst_params['hps'][key]
            for key1 in cur_dict:
                if key1 not in init_params[key]:
                    print(key1)


def viz_hpst(cur_params):
    hpst_params = None
    with open('../../setup/hpst.yaml', 'r') as stream:
        try:
            hpst_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    hps = hpst_params['hps']
    for key0 in hps:
        print(f'###{key0}###')
        for key1 in hps[key0]:
            print(f'{key1}: {cur_params[key0][key1]}')


def main():
    init_params = get_params()
    # print(init_params.keys())
    
    # load hyperparameter set
    hpst_params = None
    with open('../../setup/hpst.yaml', 'r') as stream:
        try:
            hpst_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    hps_id = hpst_params['hps_id']
    rounds = hpst_params['rounds']
    hps = hpst_params['hps']
    
    hpst_results = {}
    
    for i in range(rounds):
        current_params = init_params.copy()
        for key0, vals0 in hps.items():
            for key1, vals1 in vals0.items():
                random.seed()
                idx = random.randint(1, 10000) % len(vals1)
                current_params[key0][key1] = vals1[idx]
                # print(vals1[idx])
        viz_hpst(current_params)
        one_test_results = test_one_paramset(current_params)
        
        hpst_results[i] = {
            'params': copy.deepcopy(current_params),
            'results': one_test_results,
        }

    pickle_save(f'../../results/hpst/{hps_id}.pkl', hpst_results)
                
    


if __name__ == '__main__':
    main()
