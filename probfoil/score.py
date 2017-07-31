"""
Module name
"""

from __future__ import print_function

import math

def rates(rule):
    tp = 0.0 # examples correctly classified as positive - true positives
    fp = 0.0 # no. of false positives
    p = 0.0 # no. of positive examples
    m = 0 # a parameter of the algorithm
    for c, pr in zip(rule.correct, rule.scores): # what is this object?
        tp += min(c, pr) # pr is probability under H, c is actual tp_i
        # if H overestimates e_i, then tp_i is maximal, and pr-c contributes to
        # the false positive part of the dataset
        # if H underestimates e_i then the only tp part is pr, and the remaining
        # c i.e. c-pr contributes to the false negatives
        fp += max(0, pr - c) # remaining part of pr-c is false positive if pr>c
        # and false negative if pr<c
        p += c # counting number of positive examples
        m += 1 # counting number of examples in the dataset
    n = m - p #  no. of negative examples
    return tp, fp, n - fp, p - tp


def m_estimate(rule, m=1):
    """Compute the m-estimate of the rule.

    Same as the algorithm shown in the paper

    :param rule: rule to score
    :param m: m parameter for m-estimate
    :return: value of m-estimate
    """
    tp, fp, tn, fn = rates(rule)
    p = tp + fn
    n = tn + fp
    return (tp + (m * p / (p + n))) / (tp + fp + m)


def m_estimate_future(rule, m=1):
    """Compute the m-estimate for the optimal extension of this rule.

    The optimal extension has the same TP-rate and zero FP-rate.

    :param rule: rule to score
    :param m: m parameter for m-estimate
    :return: value of m-estimate assuming an optimal extension
    """

    tp, fp, tn, fn = rates(rule)
    p = tp + fn
    n = tn + fp
    fp = 0.0
    return (tp + (m * p / (p + n))) / (tp + fp + m)


def m_estimate_relative(rule, m=1):
    """Compute the m-estimate of the rule relative to the previous ruleset.

    :param rule: rule to score
    :param m: m parameter for m-estimate
    :return: value of m-estimate
    """

    if rule.previous:
        tp_p, fp_p, tn_p, fn_p = rates(rule.previous)
    else:
        tp_p, fp_p, tn_p, fn_p = 0.0, 0.0, 0.0, 0.0   # last two are irrelevant

    tp, fp, tn, fn = rates(rule)
    p = tp + fn - tp_p
    n = tn + fp - fp_p
    return (tp - tp_p + (m * p / (p + n))) / (tp + fp - tp_p - fp_p + m)


def m_estimate_future_relative(rule, m=1):
    """Compute the m-estimate for the optimal extension of this rule relative to the previous ruleset.

    The optimal extension has the same TP-rate and zero FP-rate.

    :param rule: rule to score
    :param m: m parameter for m-estimate
    :return: value of m-estimate assuming an optimal extension
    """

    if rule.previous:
        tp_p, fp_p, tn_p, fn_p = rates(rule.previous)
    else:
        tp_p, fp_p, tn_p, fn_p = 0.0, 0.0, 0.0, 0.0  # last two are irrelevant

    tp, fp, tn, fn = rates(rule)
    p = tp + fn - tp_p
    n = tn + fp - fp_p
    fp = fp_p
    return (tp - tp_p + (m * p / (p + n))) / (tp + fp - tp_p - fp_p + m)


def accuracy(rule):
    """
    Accuracy is the number of correctly classified examples divided by the total
    number of examples i.e. the proportion of examples correctly classified
    """
    tp, fp, tn, fn = rates(rule)
    return (tp + tn) / (tp + fp + tn + fn)


def precision(rule):
    """
    Precision is the number of examples correctly classified as positive over
    the total number of examples classified as positive i.e. the proportion of
    examples classified as positive that are actually positive.
    """
    tp, fp, tn, fn = rates(rule)
    if tp + fp == 0:
        return 0.0
    else:
        return tp / (tp + fp)


def recall(rule):
    """
    Recall is the number of examples correctly classified as positive divided by
    the total number of positive examples i.e. the proportion of positive
    examples that are correctly classified as such
    - the positives that have been 'caught'
    """
    tp, fp, tn, fn = rates(rule)
    return tp / (tp + fn)


def chi2_cdf(x):
    return math.erf(math.sqrt(x / 2))


def pvalue2chisquare(s, low=0.0, high=100.0, precision=1e-8):
    """Helper function for transforming significance p-value into ChiSquare decision value."""
    v = (low + high) / 2
    r = chi2_cdf(v)
    if -precision < r - s < precision:
        return v
    elif r > s:
        return pvalue2chisquare(s, low, v)
    else:
        return pvalue2chisquare(s, v, high)


def significance(rule, calc_max=False):
    """Compute the significance of a rule (chi-square distributed)."""

    c_tp, c_fp, c_tn, c_fn = rates(rule)

    pos = c_tp + c_fn
    neg = c_fp + c_tn

    if rule.previous:
        p_tp, p_fp, p_tn, p_fn = rates(rule.previous)
    else:
        p_tp, p_fp, p_tn, p_fn = 0.0, 0.0, neg, pos

    if calc_max:
        s_tp_max = c_tp     # TODO
        s_tp = s_tp_max - p_tp
        s_fp = 0
    else:
        s_tp = c_tp - p_tp
        s_fp = c_fp - p_fp

    s_pos = c_tp + c_fn
    s_neg = c_fp + c_tn
    s_all = s_pos + s_neg

    c = s_tp + s_fp     # max: c = s_tp
    if c == 0:
        return 0

    f_pos_c = s_tp / c     # max: f_pos_c = 1
    f_neg_c = 1 - f_pos_c  # max: f_neg_c == 0

    f_pos = s_pos / s_all
    f_neg = s_neg / s_all

    pos_log = math.log(f_pos_c / f_pos) if f_pos_c > 0 else 0  # max: pos_log = -log(sP / sM)
    neg_log = math.log(f_neg_c / f_neg) if f_neg_c > 0 else 0  # max: neg_log = 0

    l = 2 * c * (f_pos_c * pos_log + f_neg_c * neg_log)  # max: 2 * sTP * -log(sP/sM)

    return l # is this why the rules have no probabilities attached?
