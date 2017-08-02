"""
Implementation of an algorithm for learning categorical defaults from a set of
probabilistic rules, with the addition of an abnormality predicate

Implemented as an extension of the Prob2FOIL algorithm
"""

from __future__ import print_function

from .data import DataFile
from problog.logic import Term, Clause, And
from .rule import FOILRule
from .learn import CandidateBeam, LearnEntail

def construct_ab_pred(rules, examples, datafiles):
    """
    Take the head of a rule and construct a new abnormality predicate from it
    which covers exceptions to the category
    e.g. flies(x):- bird(x)
    abnormality predicate is ab_bird(x)

    Find instances which the abnormality predicate applies to and write these
    instances to the data file

    :param: rules
    :type: ProbFOIL rules
    :param: examples
    :type: learn class
    :param: datafiles
    :type: files
    """

    try:
        clauses = rules.to_clauses() # turn into clauses
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
                    neg_preds = [str(p)[2:] for p in pred if p.is_negated()==True] # get negated predicates
                    pos_preds = [p for p in pred if p.is_negated()==False] # get non-negated predicates
                    pred = pos_preds[0] # change to deal with two preds?
                    if neg_preds: # would be good to get arity automatically
                        neg_pred_examples = [examples._data.query(Term(r[:-3]), 1) for r in neg_preds] # get instances of neg_preds
                        #print(neg_pred_examples)
                    ab_pred = 'ab_' + str(pred)[:-3] # create name for abnormal predicate
                    ab_preds[clause] = ab_pred # store ab_pred for each clause
                    objects = [i[0] for li in neg_pred_examples for i in li] # get instance from nested lists/tuples
                    #print(objects)
                    ab_pred_examples = [ab_pred+'(%s)' % i for i in objects] # create ab_pred data points
                    ab_pred_mode = 'mode(%s(+)).' % ab_pred # create mode
                    ab_pred_type = 'base(%s(x)).' % ab_pred # create type
                    #print(ab_pred_examples)
                    settings = datafiles[0]
                    data = datafiles[1]
                    with open(data, 'a') as w: # write ab_pred instances to data file
                        for i in ab_pred_examples:
                            w.write(i)
                            w.write('.\n')
                    with open(settings, 'a') as w: # write settings to settings file
                        w.write(ab_pred_mode)
                        w.write('\n')
                        w.write(ab_pred_type)
        print(ab_preds)
    except:
        traceback.print_exc()
