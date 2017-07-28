"""
Implementation of an algorithm for learning categorical defaults from a set of
probabilistic rules, with the addition of an abnormality predicate

Implemented as an extension of the Prob2FOIL algorithm
"""

from __future__ import print_function

from problog.program import PrologFile
from .data import DataFile
from .language import TypeModeLanguage
from problog.util import init_logger
from problog.logic import Term, Clause, And
from .rule import FOILRule
from .learn import CandidateBeam, LearnEntail

from logging import getLogger

import time
import argparse
import sys
import random
import traceback
import re

from .score import rates, accuracy, m_estimate_relative, precision, recall, m_estimate_future_relative, significance, pvalue2chisquare

def construct_ab_pred(rules, examples, datafiles):
    """
    Take the head of a rule and construct a new abnormality predicate from it
    which covers exceptions to the category
    e.g. flies(x):- bird(x)
    abnormality predicate is ab_bird(x)

    Extend the original rule using the new abnormality predicate
    e.g. flies(x):- bird(x) and ab_bird(x) will produce the new rule
    flies(x):- bird(x), \+ab_bird(x)

    :param: FOILRule
    :type: term
    :return: abnormal predicate
    :rtype: term??
    """

    try:
        clauses = rules.to_clauses()
        preds = {}
        ab_preds = {}
        for clause in clauses:
            neg_preds = []
            neg_pred_examples = []
            pred = clause.body
            
            if str(pred) == ('fail' or 'true'): # skip fail/true antecedents
                pass
            else:
                if isinstance (pred, And): # check if body is conjunction
                    pred = pred.to_list()  # turn into list of disjuncts
                    neg_preds = [str(p)[2:] for p in pred if p.is_negated()==True]
                    pos_preds = [p for p in pred if p.is_negated()==False]
                    pred = pos_preds[0] # change to deal with two preds
                    if neg_preds: # would be good to get arity automatically
                        neg_pred_examples = [examples._data.query(Term(r[:-3]), 1) for r in neg_preds]
                    preds[clause] = pred
                    ab_pred = 'ab_' + str(pred)[:-3] # create name for abnormal predicate
                    ab_preds[clause] = ab_pred
                    objects = [i[0][0] for i in neg_pred_examples]
                    ab_pred_examples = [ab_pred+'(%s)' % i for i in objects] # create ab_pred data
                    ab_pred_mode = 'mode(%s(+)).' % ab_pred
                    ab_pred_type = 'base(%s(x)).' % ab_pred
                    print(ab_pred_examples)

                    settings = datafiles[0]
                    data = datafiles[1]
                    with open(data, 'a') as w:
                        for i in ab_pred_examples:
                            w.write(i)
                            w.write('.\n')
                    with open(settings, 'a') as w:
                        w.write(ab_pred_mode)
                        w.write('\n')
                        w.write(ab_pred_type)

        print(ab_preds)
    except:
        traceback.print_exc()
        #print ("Something went wrong")

    # return ab_preds

# In main module execution
# threshold = args[threshold]
#for rule in best_rules:
#    if rule.probability >= threshold:
#        learn.learn_defaults
