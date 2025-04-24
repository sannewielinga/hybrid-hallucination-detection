import numpy as np
import scipy
from sklearn import metrics
import pandas as pd
import logging


def bootstrap(function, rng, n_resamples=1000):
    """
    A decorator that computes the standard error and 90% confidence interval for
    the given function.

    Parameters
    ----------
    function : callable
        The function to bootstrap.
    rng : numpy.random.Generator
        A random number generator object.
    n_resamples : int, optional
        The number of bootstrap resamples to generate. Defaults to 1000.

    Returns
    -------
    inner : callable
        A function that takes in data and returns a dictionary with the standard
        error, lower confidence limit, and upper confidence limit.
    """
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data,),
            function,
            n_resamples=n_resamples,
            confidence_level=0.9,
            random_state=rng,
        )
        return {
            "std_err": bs.standard_error,
            "low": bs.confidence_interval.low,
            "high": bs.confidence_interval.high,
        }

    return inner

def rejection_accuracy_curve(y_true_correct, y_score_uncertainty, rejection_fractions=None):
    """
    Calculates the accuracy of accepted samples at different rejection fractions.

    Args:
        y_true_correct (np.ndarray): Boolean array where True indicates a correct sample.
        y_score_uncertainty (np.ndarray): Array of uncertainty scores where HIGHER means
                                           MORE uncertain (more likely incorrect).
        rejection_fractions (np.ndarray, optional): Array of rejection fractions (0 to <1)
                                                    to evaluate. Defaults to np.linspace(0, 0.95, 20).

    Returns:
        tuple: (rejection_fractions, accuracies)
               - rejection_fractions: The fractions used for calculation.
               - accuracies: The accuracy of accepted samples at each fraction.
                 Returns np.nan for accuracy if all samples are rejected.
    """
    if rejection_fractions is None:
        rejection_fractions = np.linspace(0, 0.95, 20)

    y_true_correct = np.asarray(y_true_correct)
    y_score_uncertainty = np.asarray(y_score_uncertainty)

    valid_mask = ~np.isnan(y_true_correct) & ~np.isnan(y_score_uncertainty)
    if not np.any(valid_mask):
        logging.warning("AURAC calculation: No valid samples after NaN filtering.")
        return rejection_fractions, np.full_like(rejection_fractions, np.nan)

    y_true_valid = y_true_correct[valid_mask]
    y_score_valid = y_score_uncertainty[valid_mask]
    n_total_valid = len(y_true_valid)

    if n_total_valid == 0:
        logging.warning("AURAC calculation: Zero valid samples.")
        return rejection_fractions, np.full_like(rejection_fractions, np.nan)

    desc_score_indices = np.argsort(y_score_valid)[::-1]
    y_true_sorted = y_true_valid[desc_score_indices]

    accuracies = []
    for fraction in rejection_fractions:
        num_reject = int(np.floor(n_total_valid * fraction))
        num_keep = n_total_valid - num_reject

        if num_keep == 0:
            accuracies.append(np.nan)
        else:
            accepted_samples_correctness = y_true_sorted[num_reject:]
            accuracy_at_fraction = np.mean(accepted_samples_correctness)
            accuracies.append(accuracy_at_fraction)

    return rejection_fractions, np.array(accuracies)

def aurac(y_true_correct, y_score_uncertainty, rejection_fractions=None):
    """
    Calculates the Area Under the Rejection Accuracy Curve (AURAC).

    Args:
        y_true_correct (np.ndarray): Boolean array where True indicates a correct sample.
        y_score_uncertainty (np.ndarray): Array of uncertainty scores where HIGHER means
                                           MORE uncertain (more likely incorrect).
        rejection_fractions (np.ndarray, optional): Array of rejection fractions (0 to <1)
                                                    to evaluate. Defaults to np.linspace(0, 0.95, 20).

    Returns:
        float: The AURAC score, or np.nan if calculation is not possible.
    """
    fractions, accuracies = rejection_accuracy_curve(
        y_true_correct, y_score_uncertainty, rejection_fractions
    )

    valid_acc_mask = ~np.isnan(accuracies)
    if np.sum(valid_acc_mask) < 2:
        logging.warning("AURAC calculation: Less than 2 valid accuracy points, cannot calculate area.")
        return np.nan

    try:
        area = metrics.auc(fractions[valid_acc_mask], accuracies[valid_acc_mask])
        return area
    except Exception as e:
        logging.error(f"Error calculating AUC for rejection curve: {e}")
        return np.nan

def auroc(y_true, y_score):
    """
    Computes the area under the ROC curve given true binary labels and predicted
    scores.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted scores.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    """
    Compute the accuracy of samples with uncertainty below a certain quantile.

    Parameters
    ----------
    accuracies : array_like
        Array of accuracy values.
    uncertainties : array_like
        Array of uncertainty values.
    quantile : float
        Quantile of the uncertainty distribution to use as a cutoff.

    Returns
    -------
    accuracy : float
        Mean accuracy of samples with uncertainty below the cutoff.
    """
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def area_under_thresholded_accuracy(accuracies, uncertainties):
    """
    Compute area under the accuracy curve, with accuracy computed at quantiles
    of the uncertainty distribution.

    Parameters
    ----------
    accuracies : array-like
        Accuracy of each sample.
    uncertainties : array-like
        Uncertainty of each sample.

    Returns
    -------
    area : float
        The area under the accuracy curve.
    """
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array(
        [accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles]
    )
    dx = quantiles[1] - quantiles[0]
    area = (select_accuracies * dx).sum()
    return area


def compatible_bootstrap(func, rng):
    """
    Converts a function to be compatible with bootstrapping for performance evaluation.

    This function wraps the input arrays `y_true` and `y_score` into a format suitable for
    bootstrapping and applies the specified `func` to compute metrics such as AUROC.

    Parameters
    ----------
    func : callable
        The function to compute a performance metric, e.g., AUROC, using `y_true` and `y_score`.
    rng : np.random.Generator
        A NumPy random number generator for reproducibility of the bootstrap sampling.

    Returns
    -------
    callable
        A function that takes `y_true` and `y_score` arrays, wraps them for bootstrapping,
        and returns the bootstrapped evaluation metric.
    """

    def helper(y_true_y_score):
        y_true = np.array([i["y_true"] for i in y_true_y_score])
        y_score = np.array([i["y_score"] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{"y_true": i, "y_score": j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)

    return converted_func


def compatible_bootstrap_aurac(func, rng):
    def helper(y_true_correct_y_score):
        y_true_correct = np.array([i["y_true_correct"] for i in y_true_correct_y_score])
        y_score = np.array([i["y_score"] for i in y_true_correct_y_score])
        out = func(y_true_correct, y_score)
        return out

    def wrap_inputs(y_true_correct, y_score):
        return [{"y_true_correct": i, "y_score": j} for i, j in zip(y_true_correct, y_score)]

    def converted_func(y_true_correct, y_score):
        y_true_correct_y_score = wrap_inputs(y_true_correct, y_score)
        return bootstrap(helper, rng=rng)(y_true_correct_y_score)

    return converted_func