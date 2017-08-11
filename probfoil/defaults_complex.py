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

import sys, traceback

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
            neg_preds = [] # for storing exceptions to the rule (negated predicates)
            neg_pred_examples = [] # store objects of neg_preds in the data
            predicate = [] # for storing the predicate to turn into ab_pred

            pred = clause.body

            if str(pred) == ('fail' or 'true'): # skip fail/true antecedents
                pass
            else:
                if isinstance (pred, And): # check if body is conjunction
                    pred = pred.to_list()  # turn into list of disjuncts
                    neg_preds = [str(p)[2:] for p in pred if p.is_negated()==True] # get negated predicates
                    pos_preds = [str(p) for p in pred if p.is_negated()==False] # get non-negated predicates

                    if neg_preds: # would be good to get arity automatically
                        neg_pred_examples = [examples._data.query(Term(r[:-3]), 1) for r in neg_preds] # get instances of neg_preds
                        #print(neg_pred_examples)
                    neg_objects = [i[0] for li in neg_pred_examples for i in li] # get instance from nested lists/tuples

                    #if pos_preds:
                    #    pos_pred_examples = [examples._data.query(Term(r[:-3]), 1) for r in pos_preds] # get instances of pos_preds
                    #pos_objects = [i[0] for li in pos_pred_examples for i in li] # get instance from nested lists/tuples
                    #print(pos_objects)

                    if len(pos_preds) > 1:
                        print('Candidates for abnormality:', pos_preds)
                        for p in pos_preds:
                            print(p)
                            pos_examples = examples._data.query(Term(p[:-3]), 1)
                            p_objects = [i[0] for i in pos_examples]
                            #print(p_objects, type(p_objects))
                            if set(p_objects).issuperset(set(neg_objects)):
                                print(p, 'will be used to form the abnormal predicate')
                                predicate.append(p)
                            else:
                                print('Removing', p, 'from candidate predicates')
                        pred = predicate[0] # change to create two abnormal predicates?
                    else:
                        pred = pos_preds[0]

                    #pred = predicate[0] # change to create two abnormal predicates?

                    ab_pred = 'ab_' + str(pred)[:-3] # create name for abnormal predicate
                    ab_preds[clause] = ab_pred # store ab_pred for each clause
                    ab_pred_examples = [ab_pred+'(%s)' % i for i in neg_objects] # create ab_pred data points
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
                        w.write('\n')
                        w.write(ab_pred_mode)
                        w.write('\n')
                        w.write(ab_pred_type)
        print(ab_preds)
    except:
        traceback.print_exc()
