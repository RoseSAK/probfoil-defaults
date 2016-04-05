from __future__ import print_function

from problog.program import PrologFile
from problog.logic import Term, Var
from data import DataFile
from language import TypeModeLanguage
from problog.util import init_logger, Timer
from rule import FOILRule
from learn import CandidateBeam

from logging import getLogger

from itertools import product

import sys
import argparse

from score import rates, accuracy, m_estimate, precision, recall


class LearnEntail(object):

    def __init__(self, data, language, logger=None):
        self._language = language
        self._target = None
        self._examples = None
        self._logger = logger

        self._data = data
        self._scores_correct = None

    @property
    def target(self):
        """The target predicate of the learning problem."""
        return self._target

    @property
    def examples(self):
        """The examples (tuple of target arguments) of the learning problem."""
        return self._examples

    @property
    def language(self):
        """Language specification of the learning problem."""
        return self._language

    def load(self, data):
        """Load the settings from a data file.

        Initializes language, target and examples.

        :param data: data file
        :type data: DataFile
        """
        self.language.load(data)  # for types and modes

        target = data.query('learn', 1)[0]
        target_functor, target_arity = target[0].args
        target_arguments = [Var(chr(65 + i)) for i in range(0, int(target_arity))]
        self._target = Term(str(target_functor), *target_arguments)

        # Find examples:
        #  if example_mode is closed, we will only use examples that are defined in the data
        #      this includes facts labeled with probability 0.0 (i.e. negative example)
        #  otherwise, the examples will consist of all combinations of values appearing in the data
        #      (taking into account type information)
        example_mode = data.query(Term('example_mode'), 1)
        if example_mode and str(example_mode[0][0]) == 'auto':
            types = self.language.get_argument_types(self._target.functor, self._target.arity)
            values = [self.language.get_type_values(t) for t in types]
            self._examples = list(product(*values))
        else:
            self._examples = [r for r in data.query(self._target.functor, self._target.arity)]

        with Timer('Computing scores', logger=self._logger):
            self._scores_correct = self._compute_scores_correct()

    def _compute_scores_correct(self):
        """Computes the score for each example."""
        result = self._data.evaluate(rule=None, functor=self.target.functor, arguments=self.examples)

        scores_correct = []
        for example in self.examples:
            scores_correct.append(result.get(Term(self.target.functor, *example), 0.0))
        return scores_correct

    def _compute_scores_predict(self, rule):
        functor = 'eval_rule'
        result = self._data.evaluate(rule, functor=functor, arguments=self.examples)
        scores_predict = []
        for example in self.examples:
            scores_predict.append(result.get(Term(functor, *example), 0.0))
        return scores_predict


class ProbFOIL(LearnEntail):

    def __init__(self, data, beam_size=5, logger='probfoil', m=1, **kwargs):
        LearnEntail.__init__(self, data, TypeModeLanguage(**kwargs), logger=logger)

        self._beamsize = beam_size
        self._m_estimate = m

        self.load(data)   # for types and modes

        getLogger(self._logger).info('Number of examples (M): %d' % len(self._examples))
        getLogger(self._logger).info('Positive weight (P): %d' % sum(self._scores_correct))
        getLogger(self._logger).info('Negative weight (N): %d' %
                                     (len(self._scores_correct) - sum(self._scores_correct)))

    def best_rule(self, current):
        """Find the best rule to extend the current rule set.

        :param current:
        :return:
        """

        current_rule = FOILRule(target=self.target, previous=current, correct=self._scores_correct)
        current_rule.scores = [0.0] * len(self._scores_correct)
        current_rule.score = self._compute_rule_score(current_rule)
        current_rule.processed = False

        best_rule = current_rule
        candidates = CandidateBeam(self._beamsize)
        candidates.push(current_rule)
        while candidates:
            next_candidates = CandidateBeam(self._beamsize)
            while candidates:
                current_rule = candidates.pop()
                for ref in self.language.refine(current_rule):
                    rule = current_rule & ref
                    rule.scores = self._compute_scores_predict(rule)
                    rule.score = self._compute_rule_score(rule)
                    rule.processed = False
                    getLogger(self._logger).debug('%s %s %s' % (rule, rule.score, rates(rule)))
                    if rule.score > current_rule.score:
                        next_candidates.push(rule)
                    if rule.score > best_rule.score:
                        best_rule = rule
            candidates = next_candidates
        self._select_rule(best_rule)
        return best_rule

    def learn(self):
        hypothesis = None
        current_score = 0.0

        while True:
            next_hypothesis = self.best_rule(hypothesis)
            new_score = accuracy(next_hypothesis)
            getLogger(self._logger).info('RULE LEARNED: %s %s' % (next_hypothesis, new_score))
            if new_score > current_score:
                hypothesis = next_hypothesis
                current_score = new_score
            else:
                break

        return hypothesis

    def _compute_rule_score(self, rule):
        return m_estimate(rule, self._m_estimate)

    def _select_rule(self, rule):
        pass


class ProbFOIL2(ProbFOIL):

    def __init__(self, *args, **kwargs):
        ProbFOIL.__init__(self, *args, **kwargs)

    def _select_rule(self, rule):
        # set rule probability and update scores
        if rule.max_x > 1 - 1e-8:
            rule.set_rule_probability(None)
        else:
            rule.set_rule_probability(rule.max_x)
        if rule.previous is None:
            scores_previous = [0.0] * len(rule.scores)
        else:
            scores_previous = rule.previous.scores

        x = rule.max_x
        for i, lu in enumerate(zip(scores_previous, rule.scores)):
            l, u = lu
            s = u - l
            rule.scores[i] = l + x * s

    def _compute_rule_score(self, rule):
        if rule.previous is None:
            scores_previous = [0.0] * len(rule.scores)
        else:
            scores_previous = rule.previous.scores

        pos = 0.0
        all = 0.0

        tp_prev = 0.0
        fp_prev = 0.0
        fp_base = 0.0
        tp_base = 0.0
        ds_total = 0.0
        pl_total = 0.0
        values = []
        for p, l, u in zip(rule.correct, scores_previous, rule.scores):
            pos += p
            all += 1.0

            tp_prev += min(l, p)
            fp_prev += max(0, l - p)

            ds = u - l  # improvement on previous prediction (note: ds >= 0)
            if ds == 0:  # no improvement
                pass
            elif p < l:  # lower is overestimate
                fp_base += ds
            elif p > u:  # upper is underestimate
                tp_base += ds
            else:   # correct value still achievable
                ds_total += ds
                pl_total += p - l
                y = (p - l) / (u - l)   # for x equal to this value, prediction == correct
                values.append((y, p, l, u))

        neg = all - pos
        mpnp = self._m_estimate * (pos / all)

        def comp_m_estimate(tp, fp):
            score = (tp + mpnp) / (tp + fp + self._m_estimate)
            # print (self._m_estimate, mpnp, tp, fp, score)
            return score

        max_x = 1.0
        tp_x = pl_total + tp_base + tp_prev
        fp_x = ds_total - pl_total + fp_base + fp_prev
        score_x = comp_m_estimate(tp_x, fp_x)
        max_score = score_x

        if values:
            values = sorted(values)
            tp_x, fp_x, tn_x, fn_x = 0.0, 0.0, 0.0, 0.0
            ds_running = 0.0
            pl_running = 0.0
            prev_y = None
            for y, p, l, u in values + [(None, 0.0, 0.0, 0.0)]:     # extra element forces compute at end
                if y is None or prev_y is not None and y > prev_y:
                    # There is a change in y-value.
                    x = prev_y  # set current value of x
                    tp_x = pl_running + x * (ds_total - ds_running) + x * tp_base + tp_prev
                    fp_x = x * ds_running - pl_running + x * fp_base + fp_prev
                    score_x = comp_m_estimate(tp_x, fp_x)

                    if max_score is None or score_x > max_score:
                        max_score = score_x
                        max_x = x

                prev_y = y
                pl_running += p - l
                ds_running += u - l

        rule.max_x = max_x
        return max_score


def main(args):

    logger = 'probfoil'

    log = init_logger(verbose=args.verbose, name=logger)

    # Load input files
    data = DataFile(*(PrologFile(source) for source in args.files))

    if args.probfoil1:
        learn_class = ProbFOIL
    else:
        learn_class = ProbFOIL2
    learn = learn_class(data, symmetry_breaking=False, beam_size=1, logger=logger, **vars(args))

    hypothesis = learn.learn()

    print ('================= FINAL THEORY =================')
    rule = hypothesis
    rules = [rule]
    while rule.previous:
        rule = rule.previous
        rules.append(rule)
    for rule in reversed(rules):
        print (rule)
    print ('==================== SCORES ====================')
    print ('Accuracy:\t', accuracy(hypothesis))
    print ('Precision:\t', precision(hypothesis))
    print ('Recall:\t', recall(hypothesis))


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('-1', '--det-rules', action='store_true', dest='probfoil1',
                        help='learn deterministic rules')
    parser.add_argument('-m', help='parameter m for m-estimate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('-v', action='count', dest='verbose', default=None)
    return parser

if __name__ == '__main__':
    main(argparser().parse_args())


