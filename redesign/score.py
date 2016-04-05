"""
Module name
"""

from __future__ import print_function


def rates(rule):
    tp = 0.0
    fp = 0.0
    p = 0.0
    m = 0
    for c, pr in zip(rule.correct, rule.scores):
        tp += min(c, pr)
        fp += max(0, pr - c)
        p += c
        m += 1
    n = m - p
    return tp, fp, n - fp, p - tp


def m_estimate(rule, m=1):
    tp, fp, tn, fn = rates(rule)
    p = tp + fn
    n = tn + fp
    return (tp + (m * p / (p + n))) / (tp + fp + m)


def accuracy(rule):
    tp, fp, tn, fn = rates(rule)
    return (tp + tn) / (tp + fp + tn + fn)


def precision(rule):
    tp, fp, tn, fn = rates(rule)
    if tp + fp == 0:
        return 0.0
    else:
        return tp / (tp + fp)


def recall(rule):
    tp, fp, tn, fn = rates(rule)
    return tp / (tp + fn)

