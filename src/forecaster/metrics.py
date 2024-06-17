from typing import Callable, Dict, List
import numpy as np
import properscoring as ps
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

EPS = 1e-6


def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def norm_rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    try:
        scale = MinMaxScaler()
        targets = scale.fit_transform(targets[:,None])
        predictions = scale.transform(predictions[:,None])
    finally:
       return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):
    """
    Mean Absolute Percentage Error
    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model
    Returns:
        float: MAPE
    """
    targets[targets==0] = np.nan
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100


# target ground truth
# mean -
def crps(mean, std, targets):
    """
    Quantile-based CRPS
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: CRPS
    """
    std_clip = np.clip(std, EPS, None)
    ans1 = ps.crps_gaussian(targets, mean, std_clip)
    ans2 = np.abs(targets - mean)
    ans = (~np.isclose(std, 0, atol=1e-6)) * ans1 + (np.isclose(std, 0, atol=1e-6)) * ans2
    return ans.mean()


# -1
def crps_samples(samples, targets):
    """
    Quantile-based CRPS
    Args:
        samples (np.ndarray): Samples of the distribution (N, samples)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: CRPS
    """
    return ps.crps_ensemble(targets, samples).mean()


def log_score(mean, std, targets, window=0.1):
    """
    Log Score
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: Log Score
    """

    # rescale, changed code
    std = np.clip(std, EPS, None)
    scale = MinMaxScaler()
    targets = scale.fit_transform(targets)
    mean = scale.transform(mean)
    std = scale.scale_ * std

    t1 = norm.cdf(targets - window / 2.0, mean, std)
    t2 = norm.cdf(targets + window / 2.0, mean, std)
    a = np.log(np.clip(t2 - t1, EPS, 1.0)).mean()
    return np.clip(a, -10, 10)


# put in slack
def interval_score(mean, std, targets, window=1.0):
    """
    Interval Score
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: Interval Score
    """

    # rescale, changed code
    std = np.clip(std, EPS, None)
    scale = MinMaxScaler()
    targets = scale.fit_transform(targets)
    mean = scale.transform(mean)
    std = scale.scale_ * std

    rd_val = np.round(targets, decimals=1)
    low_val = np.clip(rd_val - window / 2, a_min=0.0, a_max=None)
    high_val = np.clip(rd_val + window / 2, a_min=None, a_max=13)
    t1 = norm.cdf(low_val, loc=mean, scale=std)
    t2 = norm.cdf(high_val, loc=mean, scale=std)
    return np.log(np.clip(t2 - t1, a_min=EPS, a_max=1.0)).mean()


def conf_interval(mean, var, conf):
    """
    Confintance Interval for given confidence level
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        conf (float): Confidence level
    Returns:
        tuple: (low, high) interval
    """
    var = np.clip(var, EPS, None)
    out_prob = 1.0 - conf
    high = norm.ppf(1.0 - (out_prob / 2), loc=mean, scale=var**0.5)
    low = norm.ppf((1.0 - conf) / 2, loc=mean, scale=var**0.5)
    return low, high


def pres_recall(mean, var, target, conf):
    """
    Fraction of GT points within the confidence interval
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        conf (float): Confidence level
    Returns:
        np.ndarray: Fraction of GT points within the confidence interval
    """
    low, high = conf_interval(mean, var, conf)
    truth = ((target > low) & (target < high)).astype("float32")
    return truth.mean(-1)


# Plot
def get_pr(pred, var, target, color="blue", label="FluFNP"):
    """
    Plot confidence and return Confidence score and AUC
    Args:
        pred (np.ndarray): Predictions of the model (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        color (str): Color of the line
        label (str): Label of the model
    Returns:
        tuple: (Confidence score, AUC, fraction values)
    """ 
    pred_, var_, target_ = pred.squeeze(), var.squeeze(), target.squeeze()
    x = np.arange(0.05, 1.0, 0.01)
    y = np.array([pres_recall(pred_, var_, target_, c) for c in x])
    #     plt.plot(list(x) + [1.0], list(y) + [1.0], label=label, color=color)
    conf_score = np.abs(y - x).sum() * 0.01
    auc = y.sum() * 0.01
    return auc, conf_score, list(y) + [1.0]


def dist_from_quantiles(quantiles: Dict[float, float]) -> Callable[[float], float]:
    """
    Returns an approximation cdf for a given quantile
    Args:
        quantiles (Dict[float, float]): Quantiles and corresponding values
    Returns:
        Callable: Approximation cdf
    """

    def cdf(x: float) -> float:
        for q, v in quantiles.items():
            if x < v:
                return q
        return 1.0

    return cdf


def crps_integrated(quantiles: List[Dict[float, float]], target: np.ndarray) -> float:
    """
    Returns CRPS for a given quantile
    Args:
        quantiles (Dict[float, float]): Quantiles and corresponding values
        target (float): Target of the model
    Returns:
        float: CRPS
    """
    cdfs = [dist_from_quantiles(q) for q in quantiles]
    # TODO: This is veeery slow. Can we do better?
    return np.array(
        [
            ps.crps_quadrature([t], cdf, tol=1e-3)
            for cdf, t in zip(cdfs, target.flatten())
        ]
    ).mean()
