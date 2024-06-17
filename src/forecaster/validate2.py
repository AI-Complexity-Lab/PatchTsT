import numpy as np
import torch
from epiweeks import Week
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from online_training import get_params
from utils import pickle_load, pickle_save
from eval_quantile import get_all_eval_results

# q_idx = 1
# aheadidx = 1

# all_test_results = pickle_load('./all_eval_results0.pkl')
# q0_res = all_test_results[q_idx]

# x_ticks = []
# y_values = []
# for region in q0_res:
#     x_ticks.append(region)
#     y_values.append(q0_res[region][aheadidx])
#     # print(q0_res[region][0])
# plt.figure(figsize=(20,6))
# plt.bar(x_ticks, y_values)
# plt.show()


scale_factors = [0.5, 1, 1.5, 3]
err_windows = [3]


for scale_factor in scale_factors:
    for err_window in err_windows:
        cp_params = None
        with open('../../setup/exp_params/10.yaml', 'r') as stream:
            try:
                cp_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print('Error in reading parameters file')
                print(exc)
        cp_params['scale_factor'] = scale_factor
        cp_params['err_window'] = err_window
        all_results = get_all_eval_results(cp_params)
        pickle_save(f'../../results/erri/{int(scale_factor*10)}_{err_window}.pkl', all_results)
