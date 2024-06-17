from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from epiweeks import Week
import yaml
import argparse
from sklearn import linear_model

from utils import pickle_load, pickle_save


def smooth_scores(scores_in, linear=False, oss=False):
    y = np.array(sorted(scores_in)).reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    ransac = linear_model.RANSACRegressor()
    if linear:
        ransac = linear_model.LinearRegression()
    ransac.fit(x, y)
    pred = ransac.predict(x)
    if not oss:
        pred = np.abs(pred)
    return pred.reshape(-1)
    # plt.plot(y)
    # plt.plot(pred)
    # plt.show()
    # exit(0)


def concat(array1, array2):
    return np.concatenate((array1, array2))


def err2quantile(err, scale_factor=2.5):
    # out = 0
    # if abs(err) < 0.1:
    #     return 0
    # if abs(err) < 0.15:
    #     return err * 2
    # return err * 4
    sign = 1.5 if err > 0 else -1
    scaled_err = min(abs(err) * 3, 0.9)
    out = np.tan(scaled_err) * sign * scale_factor
    return out


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
):
    """This implementation follows https://github.com/aangelopoulos/conformal-time-series/

    Args:
        scores: nonconformity scores (absolute value of residues in this case)
        alpha: error rate
        lr: learning rate
        window_length: the number of scores used for calibrating
        T_burnin: skip the first T_burnin time steps before starting updating alpha
        ahead: weeks ahead
    """
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    grads = np.zeros((T_test,))
    predicted_CI = 0
    for t in range(T_test+1):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            if t == T_test:
                predicted_CI = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1))

            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1))
                covereds[t] = qs[t] >= scores[t]
                grad = -alpha if covereds[t_pred] else 1-alpha
                alphat = alphat - lr*grad
                if t < T_test - 1:
                    alphas[t+1] = alphat
                    grads[t_pred] = grad

        else:
            if t_pred > 0:
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = 0
    results = { "method": "ACI", "q" : qs, "alpha" : alphas, 'pred_CI':predicted_CI}
    return results


def aci_with_error_intgr(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    err_window=5,
    scale_factor=2,
):
    """This implementation follows https://github.com/aangelopoulos/conformal-time-series/

    Args:
        scores: nonconformity scores (absolute value of residues in this case)
        alpha: error rate
        lr: learning rate
        window_length: the number of scores used for calibrating
        T_burnin: skip the first T_burnin time steps before starting updating alpha
        ahead: weeks ahead
    """
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    grads = np.zeros((T_test,))
    grads1 = np.zeros((T_test,))
    predicted_CI = 0
    for t in range(T_test+1):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            if t == T_test:
                predicted_CI = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1))
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1))
                covereds[t] = qs[t] >= scores[t]
                
                # error integrate
                window_covered = np.mean(covereds[max(t_pred-err_window, 0):t_pred+1])
                window_err = 1 - window_covered - alpha
                grad1 = err2quantile(window_err, scale_factor=scale_factor)
                if t_pred < 15:
                    grad1 = 0
                
                grad = -alpha if covereds[t_pred] else 1-alpha
                alphat = alphat - lr*(grad + grad1)
                if t < T_test - 1:
                    alphas[t+1] = alphat
                    grads[t_pred] = grad
                    grads1[t_pred] = grad1
        else:
            if t_pred > 0:
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = 0
    results = { "method": "ACI", "q" : qs, "alpha" : alphas, 'pred_CI':predicted_CI}
    return results


def aci_with_prev_scores(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    prev_scores,
    starting_week_idx,
):
    """This implementation follows https://github.com/aangelopoulos/conformal-time-series/

    Args:
        scores: nonconformity scores (absolute value of residues in this case)
        alpha: error rate
        lr: learning rate
        window_length: the number of scores used for calibrating
        T_burnin: skip the first T_burnin time steps before starting updating alpha
        ahead: weeks ahead
        prev_scores: scores from last year
        starting_week_idx: the index of (starting week - 1 year) in prev_scores. 
    """
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    predicted_CI = 0
    for t in range(T_test+1):
        t_pred = t - ahead + 1
        current_week_idx = starting_week_idx + t
        current_addition_scores = np.array(prev_scores[:5])
        if current_week_idx+3 > 1:
            current_addition_scores = prev_scores[max(current_week_idx-3, 0): min(current_week_idx+3, len(prev_scores))]
        if t == T_test:
            predicted_CI = np.quantile(concat(scores[max(t_pred-window_length,0):t_pred], current_addition_scores), 1-np.clip(alphat, 0, 1))
        else:
            if t_pred > 0:
                smoothed_scores = smooth_scores(concat(scores[max(t_pred-window_length,0):t_pred], current_addition_scores))
                qs[t] = np.quantile(smoothed_scores, 1-np.clip(alphat, 0, 1))
                covereds[t] = qs[t] >= scores[t]
                grad = -alpha if covereds[t_pred] else 1-alpha
                alphat = alphat - lr*grad
                if t < T_test - 1:
                    alphas[t+1] = alphat
            else:
                qs[t] = np.quantile(current_addition_scores, 1-np.clip(alphat, 0, 1))
                covereds[t] = qs[t] >= scores[t]
                grad = -alpha if covereds[t_pred] else 1-alpha
                alphat = alphat - lr*grad
                if t < T_test - 1:
                    alphas[t+1] = alphat
                
    results = { "method": "ACI", "q" : qs, "alpha" : alphas, 'pred_CI':predicted_CI}
    return results


def prepare_scores(base_pred, target_region, ahead, oss=False):
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
        if round3(y[ahead_idx]) == -9:
            scores[-1] = -1e5
    scores = np.array(scores)
    # if ahead == 1:
    #     plt.plot(y_preds, '-')
    #     plt.plot(y_trues, '-')
    #     plt.show()
    return scores, y_preds, y_trues


def plot_helper(y_trues, y_preds, q_preds):
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


def round3(x):
    return round(x, 3)


def CL2alpha(cl, oss):
    if oss:
        return round3(1-cl)
    if cl <= 0.5:
        cl = 1 - cl
    alpha = 1 - 2 * (cl - 0.5)
    return alpha


def alpha2CL(alpha, oss):
    if oss:
        return round3(1-alpha)
    ci1 = min(alpha/2, 1-alpha/2)
    ci2 = max(alpha/2, 1-alpha/2)
    return ci1, ci2


def convert2CI(alpha, CLs, y_pred, cp_out, oss=False):
    CIs = {}
    if oss:
        cl = alpha2CL(alpha, oss=True)
        ci = max(0, y_pred + cp_out) if cl != 0.5 else y_pred
        if cl in CLs:
            CIs[cl] = ci
        return CIs
    cl1, cl2 = alpha2CL(alpha, oss=oss)
    ci1 = max(y_pred - cp_out, 0)
    ci2 = y_pred + cp_out
    if cl1 in CLs:
        CIs[cl1] = ci1
    if cl2 in CLs:
        CIs[cl2] = ci2
    return CIs


def calculate_starting_week_idx(prev_scores_first_week, starting_week):
    week1 = Week.fromstring(prev_scores_first_week) + 52
    week2 = Week.fromstring(starting_week)
    for i in range(200):
        if week1 + i - 100 == week2:
            return i - 100
    assert False
    return 0


def predict_and_save(cp_params, base_pred_results, seed=0):
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
    
    CLs = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
    for i in range(len(CLs)):
        CLs[i] = round3(CLs[i])
    print(CLs)
    sort_CIs = cp_params['sort_CIs']
    alphas = []
    for cl in CLs:
        alpha = round3(CL2alpha(cl, oss=one_side_score))
        if alpha not in alphas:
            alphas.append(alpha)
    print(alphas)

    exp_id = cp_params['exp_id']
    starting_week = str(cp_params['pred_starting_week'])
    skip_beginning = cp_params['skip_beginning']
    base_pred = base_pred_results['base_pred']
    test_pred = base_pred_results['test_pred']
    
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
    
    for region in tqdm(regions):
        for ahead in aheads:
            scores, y_preds, y_trues = prepare_scores(base_pred, region, ahead, oss=one_side_score)
            y_test_pred = test_pred[region][ahead-1]
            if y_test_pred < 0:
                y_test_pred = 0
            CIs = {}
            
            for i in range(len(scores) - skip_beginning):
                idx = i + skip_beginning
                current_week = (Week.fromstring(starting_week) + idx).cdcformat()
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
                            window_length=100,
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
                            window_length=30,
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

            test_results[region][ahead-1] = (y_test_pred, sorted_CIs)
    results = {
        'params': cp_params,
        'preds': preds,
        'test_results': test_results,
        'method': 'aci',
    }
    exp_dir_path = Path('../../results/') / f'{exp_id}'
    if not exp_dir_path.exists():
        exp_dir_path.mkdir()
    pickle_save(exp_dir_path / f'results_{seed}.pickle', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    args = parser.parse_args()
    input_file = args.input    # for example: 1
    
    cp_params = None
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            cp_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    
    base_pred_file = f'../../results/base_pred/saved_pred_{cp_params["data_id"]}.pickle'

    base_pred_results_all = pickle_load(base_pred_file, version5=True)
    if cp_params['multi_seed']:
        for seed in base_pred_results_all:
            base_pred_results = base_pred_results_all[seed]
            predict_and_save(cp_params, base_pred_results, seed)
    else:
        predict_and_save(cp_params, base_pred_results_all, cp_params['seed'])
    
    
    
    
                    
                    
                    
