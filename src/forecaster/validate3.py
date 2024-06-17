import numpy as np
import torch
from epiweeks import Week
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from online_training import get_params
from utils import pickle_load, pickle_save
from eval_quantile import save_plots


base_pred_file = f'../../results/base_pred/saved_pred_{20234401}.pickle'
base_pred = pickle_load(base_pred_file, version5=True)['base_pred']
regions = list(base_pred[0][0].keys())

for region in regions:
    save_plots(region)