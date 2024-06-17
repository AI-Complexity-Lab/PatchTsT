from epiweeks import Week
import argparse
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import tqdm
import yaml

from utils import pickle_load
from metrics import rmse, crps, mape, norm_rmse
from data_utils import load_ground_truth_before_test


def get_all_regions(dir_path, combined_file):
    if combined_file:
        preds = pickle_load(dir_path / 'results.pickle')['preds']
        regions_in_exp = list(preds.keys())
        return regions_in_exp
    regions = None
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            results = pickle_load(file_path)
            for _, pred in results['predictions'].items():
                _, _, _, _, regions = pred
                break
        break
    return regions


def get_desc(dir_path):
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            results = pickle_load(file_path)
            return results['desc']


def get_params(dir_path):
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            results = pickle_load(file_path)
            return results['params']


def combine_predictions_all(dir_path, combined_file=False, weeks_ahead=4, seed=None):
    preds, regions_in_exp, pred_weeks = None, None, None
    if combined_file:
        if seed is None:
            preds = pickle_load(dir_path / 'results.pickle')['preds']
        else:
            preds = pickle_load(dir_path / f'results_{seed}.pickle')['preds']
        regions_in_exp = list(preds.keys())
    else:
        # create dicts
        preds = {}
        regions_in_exp = get_all_regions(dir_path)
        for region in regions_in_exp:
            preds[region] = {} # weeks_ahead ->     pred_week -> (pred, true, lower, upper)
            for i in range(weeks_ahead):
                preds[region][i] = {} # pred_week -> (pred, true, lower, upper)
        
        # fill in data
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                results = pickle_load(file_path)
                pred_week = results['pred_week']
                for alpha, pred in results['predictions'].items():
                    y_preds, y_trues, lowers, uppers, regions = pred
                    for i in range(len(regions)):
                        for w in range(weeks_ahead):
                            if pred_week not in preds[regions[i]][w]:
                                preds[regions[i]][w][pred_week] = {
                                    'pred': y_preds[i][w],
                                    'true': y_trues[i][w],
                                    'lower': {},
                                    'upper': {},
                                }
                            preds[regions[i]][w][pred_week]['lower'][alpha] = lowers[i][w]
                            preds[regions[i]][w][pred_week]['upper'][alpha] = uppers[i][w]
    
    # sort according to pred_week and reshape to list
    for region in regions_in_exp:
        for w in range(weeks_ahead):
            sorted_dict_items = sorted(preds[region][w].items())
            cur_preds = dict(sorted_dict_items)
            
            if pred_weeks is None:
                pred_weeks = list(cur_preds.keys())
            
            cur_y_trues = []
            cur_y_preds = []
            cur_lowers = {}
            cur_uppers = {}
            start = True
            alpha_list = []
            for _, pred_dict in cur_preds.items():
                cur_y_preds.append(pred_dict['pred'])
                cur_y_trues.append(pred_dict['true'])
                if start:
                    start = False
                    for alpha in pred_dict['lower']:
                        ralpha = round(alpha, 2)
                        cur_lowers[ralpha] = []
                        cur_uppers[ralpha] = []
                        alpha_list.append(alpha)
                for alpha in alpha_list:
                    ralpha = round(alpha, 2)
                    cur_lowers[ralpha].append(pred_dict['lower'][alpha])
                    cur_uppers[ralpha].append(pred_dict['upper'][alpha])
            preds[region][w] = {
                'y_trues': cur_y_trues,
                'y_preds': cur_y_preds,
                'lowers': cur_lowers,
                'uppers': cur_uppers,
            }
    return preds, regions_in_exp, pred_weeks


def combine_predictions(dir_path, week_id, alpha=None, region=None, return_dict=True, all_preds=None, pred_weeks=None):
    preds = {}
    if all_preds:
        target_pred = all_preds[region][week_id]
        for i, pred_week in enumerate(pred_weeks):
            y_pred = target_pred['y_preds'][i]
            y_true = target_pred['y_trues'][i]
            lower = target_pred['lowers'][alpha][i]
            upper = target_pred['uppers'][alpha][i]
            preds[pred_week] = (y_pred, y_true, lower, upper)
    else:
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                results = pickle_load(file_path)
                pred_week = results['pred_week']
                if alpha is None:
                    alpha = list(results['predictions'].keys())[0]
                y_preds, y_trues, lowers, uppers, regions = results['predictions'][alpha]
                for i in range(len(regions)):
                    if regions[i] == region:
                        y_pred = y_preds[i][week_id]
                        y_true = y_trues[i][week_id]
                        lower = lowers[i][week_id]
                        upper = uppers[i][week_id]
                        preds[pred_week] = (y_pred, y_true, lower, upper)
    sorted_dict_items = sorted(preds.items())
    preds = dict(sorted_dict_items)
    if return_dict:
        return preds
    else:
        x_ticks = list(preds.keys())
        y_preds, y_trues, lowers, uppers = [], [], [], []
        for i in range(len(x_ticks)):
            y_pred, y_true, lower, upper = preds[x_ticks[i]]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            lowers.append(lower)
            uppers.append(upper)
        return y_preds, y_trues, lowers, uppers


def calculate_stds_helper(y_pred, lower):
    val_prob = {}
    alphas = sorted(list(lower.keys()))
    for i in range(len(alphas)):
        if i == len(alphas) - 1:
            alpha1 = alphas[i]
            itv1 = y_pred - lower[alpha1]
            val_prob[y_pred] = 1 - alpha1
        else:
            alpha1 = alphas[i]
            alpha2 = alphas[i + 1]
            itv1 = y_pred - lower[alpha1]
            itv2 = y_pred - lower[alpha2]
            val_prob[y_pred + 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
            val_prob[y_pred - 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
    variance = 0
    total_prob = 0
    for val, prob in val_prob.items():
        total_prob += prob
        variance += (val - y_pred) ** 2 * prob
    return variance ** 0.5


def calculate_stds(y_preds, lowers):
    stds = []
    # take one week from data
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        current_std = calculate_stds_helper(y_pred, lower)
        stds.append(current_std)
    return stds


def norm_crps(y_preds, y_trues, lowers):
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    stds = calculate_stds(y_preds, lowers)
    crps_val = crps(y_preds, stds, y_trues)
    return crps_val


def interval_score(y_true, upper, lower, alpha):
    return (upper - lower) + 2 / alpha * (lower - y_true) * (y_true < lower) + 2 / alpha * (y_true - upper) * (upper < y_true)


def weighted_IS_helper(y_pred, y_true, lower):
    # alphas = sorted(list(lower.keys()), reverse=True)[:-1]
    alphas = sorted(list(lower.keys()), reverse=True)
    K = len(alphas)
    sum_term = 0
    for i in range(len(alphas)):
        alpha = alphas[i]
        upper = y_pred - lower[alpha] + y_pred
        sum_term += alphas[i] / 2 * interval_score(y_true, upper, lower[alpha], alpha)
    return 1 / (K + 1/2) * (alphas[0] / 2 * abs(y_true - y_pred) + sum_term)


def weighted_IS(y_preds, y_trues, lowers):
    weighted_IS_vals = []
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        y_true = y_trues[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        weighted_IS_vals.append(weighted_IS_helper(y_pred, y_true, lower))
    return weighted_IS_vals    


def eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False):
    """_summary_

    Args:
        y_preds (list): []
        y_trues (list): []
        lowers (dict): alpha -> []
        uppers (dict): alpha -> []
    """
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)

    # weighted IS
    weighted_IS_vals = weighted_IS(y_preds, y_trues, lowers.copy())
    mean_wis = np.mean(weighted_IS_vals)
    
    # crps
    stds = calculate_stds(y_preds, lowers.copy())
    crps_val = crps(y_preds, stds, y_trues)
    
    norm_crps_val = norm_crps(y_preds, y_trues, lowers.copy())
    
    rmse_val = rmse(y_preds, y_trues)
    nrmse_val = norm_rmse(y_preds, y_trues)
    mape_val = mape(y_preds, y_trues)
    # cs score
    cs_score = 0
    for alpha in lowers:
        in_range_count = 0
        for i in range(len(y_preds)):
            if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
                in_range_count += 1
        cov_rate = in_range_count / len(y_preds)
        cs_score += 1 / len(lowers) * abs(cov_rate - 1 + alpha)
    if verbose:
        print(rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis)
    return rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis


def write_exp_results(preds, regions, output_file, weeks_ahead=4):
    # get all regions
    rows = []
    rows.append(['regions', 'weeks_ahead', 'rmse', 'nrmse', 'mape', 'cs', 'ncrps', 'mean_WIS'])
    
    # regions
    region_rows = []
    for region in regions:
        for w in range(weeks_ahead):
            cur_preds = preds[region][w]
            rmse_val, nrmse_val, mape_val, cs_score, crps_val, mean_wis = eval_metrics(cur_preds['y_preds'], cur_preds['y_trues'], cur_preds['lowers'], cur_preds['uppers'], verbose=False)
            region_rows.append([region, w, rmse_val, nrmse_val, mape_val, cs_score, crps_val, mean_wis])

    # median
    for w in range(weeks_ahead):
        tmp_dict = {i: [] for i in range(len(region_rows[0])-2)}
        for row in region_rows:
            if row[1] == w:
                for idx in tmp_dict:
                    tmp_dict[idx].append(row[idx+2])
        cur_row = ['median', w]
        for _, tmp_list in tmp_dict.items():
            cur_row.append(np.median(tmp_list))
        rows.append(cur_row)
    
    # mean
    for w in range(weeks_ahead):
        tmp_dict = {i: [] for i in range(len(region_rows[0])-2)}
        for row in region_rows:
            if row[1] == w:
                for idx in tmp_dict:
                    tmp_dict[idx].append(row[idx+2])
        cur_row = ['mean', w]
        for _, tmp_list in tmp_dict.items():
            cur_row.append(np.mean(tmp_list))
        rows.append(cur_row)
    
    # overall
    for w in range(weeks_ahead):
        y_preds, y_trues, lowers, uppers = [], [], {}, {}
        start = True
        for region in regions:
            y_preds += preds[region][w]['y_preds']
            y_trues += preds[region][w]['y_trues']
            if start:
                start = False
                for alpha in preds[region][w]['lowers']:
                    lowers[alpha] = []
                    uppers[alpha] = []
            for alpha in preds[region][w]['lowers']:
                lowers[alpha] += preds[region][w]['lowers'][alpha]
                uppers[alpha] += preds[region][w]['uppers'][alpha]
        rmse_val, nrmse_val, mape_val, cs_score, crps_val, mean_wis = eval_metrics(y_preds, y_trues, lowers, uppers)
        rows.append(['all', w, rmse_val, nrmse_val, mape_val, cs_score, crps_val, mean_wis])
    
    rows += region_rows
    
    with open(output_file, mode='w+') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    return rows


def plot_predictions(preds, week_id, figpath=None, savefig=False):
    x_ticks = list(preds.keys())
    y_preds, y_trues, lowers, uppers = [], [], [], []
    for i in range(len(x_ticks)):
        y_pred, y_true, lower, upper = preds[x_ticks[i]]
        y_preds.append(y_pred)
        y_trues.append(y_true)
        lowers.append(lower)
        uppers.append(upper)
        x_ticks[i] = (Week.fromstring(x_ticks[i]) + week_id + 1).cdcformat()
    plt.figure(figsize=(20, 4))
    plt.plot(x_ticks, y_preds, '-o', label='prediction')
    plt.plot(x_ticks, y_trues, '-o', label='ground truth')
    plt.fill_between(x_ticks, lowers, uppers, alpha=0.3)
    plt.xticks(x_ticks, rotation=45, ha='right')
    plt.legend()
    if savefig:
        plt.savefig(figpath)
    else:
        plt.show()
    plt.clf()


def get_header(weeks_ahead, quantile_list):
    header = ['regions']
    for i in range(weeks_ahead):
        header.append(f'week_{i+1}')
        for q in quantile_list:
            header.append(f'week_{i+1}_q_{q}')
    return header


def WriteRows2CSV(rows:list, output_path:Path):
    with open(output_path, 'w+') as f: 
        csv_writer = csv.writer(f)  
        csv_writer.writerows(rows)


def get_test_results_meta(results):
    regions = list(results.keys())
    second_keys = list(results[regions[0]].keys())
    aheads = [i+1 for i in range(len(second_keys))]
    print(aheads)
    CLs = list(results[regions[0]][1][1])
    print(CLs)
    return regions, aheads, CLs


def write_test_results(results, output_path):    
    regions, aheads, CLs = get_test_results_meta(results)
    rows = []
    header = get_header(len(aheads), CLs)
    rows.append(header)
    
    for region in regions:
        region_row = [region]
        for ahead in aheads:
            y_pred, CIs = results[region][ahead-1]
            region_row.append(y_pred)
            for cl in CLs:
                region_row.append(CIs[cl])
        rows.append(region_row)

    WriteRows2CSV(rows, output_path)


def viz_helper(results, viz_region, CLs, gt_length=10, savefig=False, results_path='', CL2display=None):
    # use past 10 ground truth values and <weeks_ahead> predictions along with confidence intervals to plot
    test_week = str(results['params']['test_week'])

    # data file week is test_week + 2
    data_file_week = (Week.fromstring(test_week) + 1).cdcformat()
    data_file = f'../../data/weekly/weeklydata/{data_file_week}.csv'
    
    weeks_ahead = len(results['params']['aheads'])
    test_results = results['test_results'][viz_region]
    
    # prepare x ticks
    test_week_ = Week.fromstring(test_week)
    gt_weeks = [test_week]
    test_weeks = []
    for i in range(gt_length-1):
        currect_week_ = test_week_ - i - 1
        gt_weeks.append(currect_week_.cdcformat())
    for i in range(weeks_ahead):
        currect_week_ = test_week_ + i + 1
        test_weeks.append(currect_week_.cdcformat())
    gt_weeks.sort()
    test_weeks.sort()

    # prepare values
    gt_until_test = load_ground_truth_before_test(data_file, test_week, viz_region, length_before_test=gt_length).reshape(-1)
    
    pred = []
    CIs = {}
    for cl in CLs:
        CIs[cl] = []
    for i in range(weeks_ahead):
        cur_pred, cur_CIs = test_results[i]
        pred.append(cur_pred)
        for cl in CLs:
            CIs[cl].append(cur_CIs[cl])

    # plot
    plt.clf()
    plt.plot(gt_weeks, gt_until_test)
    plt.plot(test_weeks, pred, label='prediction')
    for cl in CLs:
        if (not CL2display) or (CL2display and cl in CL2display):  
            plt.plot(test_weeks, CIs[cl], label=f'{cl}') 
    plt.xticks(rotation=45)
    plt.legend()
    if savefig:
        plt.savefig(results_path/f'pred_{viz_region}.png')
    else:
        plt.show()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='experiment id')
    args = parser.parse_args()

    # load exp params
    input_file = args.input
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            ot_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    exp_id = ot_params['exp_id']
    seeds = ot_params['seeds']
    
    all_rows = []
    for seed in seeds:
        dir_path = Path(f'../../results/{exp_id}')
        results_path = Path(f'../../results/results_{exp_id}')
        res_dir = Path(results_path)
        if not res_dir.exists():
            res_dir.mkdir()
        
        results = pickle_load(dir_path / f'results_{seed}.pickle')
        test_results = results['test_results']
        pred_results = results['preds']
        regions, aheads, CLs = get_test_results_meta(test_results)
        weeks_ahead = len(aheads)
        
        combined_file = True
        preds, regions, pred_weeks = combine_predictions_all(dir_path, combined_file=combined_file, weeks_ahead=weeks_ahead, seed=seed)
        
        # write a desc file
        try:
            desc_file_dict = get_params(dir_path)
            desc_file_dict['description'] = get_desc(dir_path)
            desc_file_rows = []
            for key, value in desc_file_dict.items():
                desc_file_rows.append([key, value])
            with open(results_path / 'desc.csv', mode='w+') as file:
                    writer = csv.writer(file)
                    writer.writerows(desc_file_rows)
        except:
            pass

        # write each seed in this experiment
        cur_rows = write_exp_results(preds, regions, res_dir/f'{exp_id}_{seed}.csv', weeks_ahead=weeks_ahead)
        all_rows.append(cur_rows)
    
    # write combined results
    avg_rows = None
    num_seeds = len(seeds)
    for rows in all_rows:
        # skip first row and first two columns
        if avg_rows is None:
            avg_rows = rows
            continue
        for i, row in enumerate(rows):
            if i == 0:
                continue
            for j in range(len(row)-2):
                avg_rows[i][j+2] += row[j+2]
    for i in range(len(avg_rows)):
        if i == 0:
            continue
        for j in range(len(avg_rows[0])-2):
            avg_rows[i][j+2] = avg_rows[i][j+2] / num_seeds
    with open(res_dir/f'{exp_id}_averaged.csv', mode='w+') as file:
        writer = csv.writer(file)
        writer.writerows(rows)