# from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from epiweeks import Week
import yaml
from utils import pickle_load, pickle_save
from aci import aci, aci_with_prev_scores, aci_with_error_intgr, round3, CL2alpha, alpha2CL, convert2CI, calculate_starting_week_idx


def prepare_scores(base_pred, target_region, ahead, oss=False, test_week_idx=0):
    scores = []
    y_preds = []
    y_trues = []
    ahead_idx = ahead - 1
    for i in range(len(base_pred)):
        predictions, addition_infos = base_pred[i]
        _, y, _ = addition_infos[target_region]
        y_pred = predictions[target_region]
        y_trues.append(y[ahead_idx])
        y_preds.append(y_pred[ahead_idx])
        if oss:
            scores.append(y[ahead_idx] - y_pred[ahead_idx])
        else:
            scores.append(np.abs(y_pred[ahead_idx] - y[ahead_idx]))
        if y[ahead_idx] == -9:
            scores[-1] = -1e5
    scores = np.array(scores)
    
    # set test week to be true test week - test week index
    n = len(scores) - test_week_idx
    
    yt_pred = y_preds[n]
    yt_true = y_trues[n]
    
    scores = scores[:n]
    y_preds = y_preds[:n]
    y_trues = y_trues[:n]
    
    return scores, y_preds, y_trues, yt_pred, yt_true


def prediction_plot_helper(y_trues, y_preds, q_preds):
    x_ticks = [i for i in range(len(y_trues))]
    uppers, lowers = [], []
    for i in range(len(y_trues)):
        uppers.append(y_preds[i] + q_preds[i])
        lowers.append(y_preds[i] - q_preds[i])
    plt.figure(figsize=(20, 4))
    plt.plot(x_ticks, y_preds, '-o', label='prediction')
    plt.plot(x_ticks, y_trues, '-o', label='ground truth')
    plt.fill_between(x_ticks, lowers, uppers, alpha=0.3)
    plt.xticks(x_ticks, rotation=45, ha='right')
    plt.legend()
    plt.savefig('tmp.png')


def coverage_plot_helper(y_trues, y_preds, q_preds):
    pass


def get_CI_on_test(cp_params, CLs, test_week_idx):
    """Test week is true test week - test week idx."""
    
    # prepare data and scores (use saved data from online_quantile.ipynb)
    target_regions = cp_params['target_regions']
    aheads = cp_params['aheads']
    one_side_score = cp_params['one_side_score']
    
    cp_lr = cp_params['cp_lr']
    use_prev_scores = cp_params['use_prev_scores']
    prev_scores = None
    if use_prev_scores:
        prev_scores = pickle_load('../../results/prev_scores.pkl')
    
    use_error_integrator = cp_params['use_error_intergrator']
    scale_factor = cp_params['scale_factor']
    err_window = cp_params['err_window']
    
    # CLs = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
    for i in range(len(CLs)):
        CLs[i] = round3(CLs[i])
    # print(CLs)
    sort_CIs = cp_params['sort_CIs']
    alphas = []
    for cl in CLs:
        alpha = round3(CL2alpha(cl, oss=one_side_score))
        if alpha not in alphas:
            alphas.append(alpha)
    # print(alphas)

    exp_id = cp_params['exp_id']
    base_pred_file = f'../../results/base_pred/saved_pred_{cp_params["data_id"]}.pickle'
    
    starting_week = str(cp_params['pred_starting_week'])
    skip_beginning = cp_params['skip_beginning']
    
    base_pred = pickle_load(base_pred_file, version5=True)['base_pred']
    
    regions = list(base_pred[0][0].keys())
    if target_regions != 'all':
        regions2eval = []
        for region in regions:
            if region in target_regions:
                regions2eval.append(region)
        regions = regions2eval
    
    preds = {}
    test_results = {}
    for region in regions:
        preds[region] = {} # weeks_ahead ->     pred_week -> (pred, true, lower, upper)
        test_results[region] = {} # weeks_ahead -> (pred, CIs)
        for ahead in aheads:
            preds[region][ahead-1] = {} # pred_week -> (pred, true, lower, upper)
    
    for region in regions:
        for ahead in aheads:
            scores, y_preds, y_trues, yt_pred, yt_true = prepare_scores(base_pred, region, ahead, oss=one_side_score, test_week_idx=test_week_idx)
            y_test_pred = yt_pred
            if y_test_pred < 0:
                y_test_pred = 0
            CIs = {}
            
            for i in range(len(scores) - skip_beginning):
                idx = i + skip_beginning
                current_week = (Week.fromstring(starting_week) + idx).cdcformat()
                # if i == len(scores) - skip_beginning - 1:
                #     print(current_week)
                y_pred = y_preds[idx]
                y_true = y_trues[idx]
                preds[region][ahead-1][current_week] = {
                    'pred': y_pred,
                    'true': y_true,
                    'lower': {},
                    'upper': {},
                }
            for alpha in alphas:
                aci_results = None
                if use_error_integrator:
                    aci_results = aci_with_error_intgr(scores=scores, alpha=alpha, lr=cp_lr, T_burnin=5, window_length=20, ahead=ahead, err_window=err_window, scale_factor=scale_factor)
                else:
                    if use_prev_scores:
                        tmp_prev_scores = prev_scores[region][ahead][one_side_score]
                        tmp_starting_week_idx = calculate_starting_week_idx('202220', starting_week)
                        aci_results = aci_with_prev_scores(
                            scores=scores,
                            alpha=alpha,
                            lr=cp_lr,
                            T_burnin=5,
                            window_length=20,
                            ahead=ahead,
                            prev_scores=tmp_prev_scores,
                            starting_week_idx=tmp_starting_week_idx
                        )
                    else:
                        aci_results = aci(
                            scores=scores,
                            alpha=alpha,
                            lr=cp_lr,
                            T_burnin=5,
                            window_length=20,
                            ahead=ahead,
                        )
                qpreds = aci_results['q']
                cp_out = aci_results['pred_CI']
                cur_CIs = convert2CI(alpha, CLs, y_test_pred, cp_out, oss=one_side_score)
                for key, val in cur_CIs.items():
                    CIs[key] = val
                for j in range(len(scores) - skip_beginning):
                    idx = j + skip_beginning
                    current_week = (Week.fromstring(starting_week) + idx).cdcformat()
                    y_pred = y_preds[idx]
                    y_true = y_trues[idx]
                    lower = y_pred - qpreds[idx]
                    upper = y_pred + qpreds[idx]
                    preds[region][ahead-1][current_week]['lower'][alpha] = lower
                    preds[region][ahead-1][current_week]['upper'][alpha] = upper
            
            # sort CIs based on keys (CLs)
            myKeys = list(CIs.keys())
            myKeys.sort()
            sorted_CIs = {i: CIs[i] for i in myKeys}
            if sort_CIs:
                vals = list(sorted_CIs.values())
                vals.sort()
                for i in range(len(vals)):
                    sorted_CIs[myKeys[i]] = vals[i] 
            test_results[region][ahead-1] = (y_test_pred, yt_true, sorted_CIs)
    return test_results


def eval(all_results, region, ahead, window=10, viz=False, save_cov_plot=False, save_path=''):
    preds = []
    trues = []
    lowers = []
    uppers = []
    lower_CL = list(all_results[0][region][ahead][2].keys())[0]
    upper_CL = list(all_results[0][region][ahead][2].keys())[1]
    # print(lower_CL, upper_CL)
    new_length = len(all_results)-ahead
    for i in range(new_length):
        yt_pred, yt_true, sorted_CIs = all_results[i][region][ahead]
        preds.append(yt_pred)
        trues.append(yt_true)
        lowers.append(sorted_CIs[lower_CL])
        uppers.append(sorted_CIs[upper_CL])
    # coverage history
    cov_history = []
    for i in range(new_length):
        current_cov = 1 if lowers[i] <= trues[i] and uppers[i] >= trues[i] else 0
        cov_history.append(current_cov)
    mean_cov_rate = np.mean(cov_history)
    deviation = upper_CL - lower_CL - np.mean(cov_history[-window:])
    
    if viz:
        # prediction plot
        x_ticks = [i for i in range(new_length)]
        plt.figure(figsize=(20, 4))
        plt.plot(x_ticks, preds, '-o', label='prediction')
        plt.plot(x_ticks, trues, '-o', label='ground truth')
        plt.fill_between(x_ticks, lowers, uppers, alpha=0.3)
        plt.xticks(x_ticks, rotation=45, ha='right')
        plt.legend()
        plt.show()

        # coverage history
        plt.plot(x_ticks, cov_history, '-o')
        plt.show()

        # coverage plot (10 weeks window)
        avg_cov_rates = []
        for i in range(len(x_ticks) - 10):
            running_mean = 0
            for j in range(10):
                running_mean += cov_history[i+j]
            running_mean = running_mean / 10
            avg_cov_rates.append(running_mean)

        plt.figure(figsize=(20, 4))
        x_ticks = x_ticks[:-10]
        plt.ylim([0, 1])
        plt.plot(x_ticks, avg_cov_rates, label='ten-week average coverage rate', )
        plt.plot(x_ticks, [(upper_CL-lower_CL)] * len(avg_cov_rates), label='ideal coverage rate')
        plt.plot(x_ticks, [mean_cov_rate] * len(avg_cov_rates), label='mean coverage rate')
        plt.xticks(x_ticks, rotation=45, ha='right')
        plt.legend()
        plt.show()

    if save_cov_plot:
        plt.clf()
        x_ticks = [i for i in range(new_length)]
        avg_cov_rates = []
        for i in range(len(x_ticks) - 10):
            running_mean = 0
            for j in range(10):
                running_mean += cov_history[i+j]
            running_mean = running_mean / 10
            avg_cov_rates.append(running_mean)

        plt.figure(figsize=(20, 4))
        x_ticks = x_ticks[:-10]
        plt.ylim([0, 1])
        plt.plot(x_ticks, avg_cov_rates, label='ten-week average coverage rate', )
        plt.plot(x_ticks, [(upper_CL-lower_CL)] * len(avg_cov_rates), label='ideal coverage rate')
        plt.plot(x_ticks, [mean_cov_rate] * len(avg_cov_rates), label='mean coverage rate')
        plt.xticks(x_ticks, rotation=45, ha='right')
        plt.legend()
        plt.savefig(save_path)

    return deviation


def get_all_eval_results(cp_params):
    # cp_params = None
    # with open('../../setup/exp_params/1.yaml', 'r') as stream:
    #     try:
    #         cp_params = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print('Error in reading parameters file')
    #         print(exc)
    weeks2eval = 15
    CLs_list = [[0.25, 0.75]]
    all_eval_results = []
    for CLs in CLs_list:
        all_test_results = []
        for w in tqdm(range(weeks2eval)):
            week_idx = weeks2eval - w
            test_results = get_CI_on_test(cp_params, CLs, week_idx)
            all_test_results.append(test_results)
        
        regions = list(all_test_results[-1].keys())
        eval_results = {}
        for region in regions:
            eval_results[region] = {}
            for ahead in range(4):
                devia = eval(all_test_results, region, ahead)
                # print(devia)
                eval_results[region][ahead] = devia
        all_eval_results.append(eval_results)
    return all_eval_results


def save_plots(region):
    viz_one_region = True
    target_region = region
    target_ahead_idx = 0
    
    weeks2eval = 30
    cp_params = None
    with open('../../setup/exp_params/10.yaml', 'r') as stream:
        try:
            cp_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    if viz_one_region:
        cp_params['target_regions'] = [target_region]
    
    CLs_list = [[0.25, 0.75]]
    all_eval_results = []
    for CLs in CLs_list:
        all_test_results = []
        for w in tqdm(range(weeks2eval)):
            week_idx = weeks2eval - w
            test_results = get_CI_on_test(cp_params, CLs, week_idx)
            all_test_results.append(test_results)
        
        if viz_one_region:
            devia = eval(all_test_results, target_region, target_ahead_idx, viz=False, save_cov_plot=True, save_path=f'../../results/cov_plots/{region}.jpg')


if __name__ == '__main__':
    # when coverage level = 0.95, take [0.025, 0.975]
    viz_one_region = True
    target_region = 'US'
    target_ahead_idx = 0
    
    weeks2eval = 30
    cp_params = None
    with open('../../setup/exp_params/11.yaml', 'r') as stream:
        try:
            cp_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    if viz_one_region:
        cp_params['target_regions'] = [target_region]
    
    CLs_list = [[0.25, 0.75], [0.025, 0.975]]
    all_eval_results = []
    for CLs in CLs_list:
        all_test_results = []
        for w in tqdm(range(weeks2eval)):
            week_idx = weeks2eval - w
            test_results = get_CI_on_test(cp_params, CLs, week_idx)
            all_test_results.append(test_results)
        
        if viz_one_region:
            devia = eval(all_test_results, target_region, target_ahead_idx, viz=True)
        else:
            regions = list(all_test_results[-1].keys())
            eval_results = {}
            for region in regions:
                eval_results[region] = {}
                for ahead in range(5):
                    devia = eval(all_test_results, region, ahead)
                    # print(devia)
                    eval_results[region][ahead] = devia
            all_eval_results.append(eval_results)
    
    if not viz_one_region:
        pickle_save('all_eval_results0.pkl', all_eval_results)