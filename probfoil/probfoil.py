from __future__ import print_function

from problog.program import PrologFile
from data import DataFile
from language import TypeModeLanguage
from problog.util import init_logger
from rule import FOILRule
from learn import CandidateBeam, LearnEntail

from logging import getLogger

import time
import argparse
import sys

from score import rates, accuracy, m_estimate, precision, recall, m_estimate_future


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

        self.interrupted = False
        self._stats_evaluations = 0

    def best_rule(self, current):
        """Find the best rule to extend the current rule set.

        :param current:
        :return:
        """

        current_rule = FOILRule(target=self.target, previous=current, correct=self._scores_correct)
        current_rule.scores = [0.0] * len(self._scores_correct)
        current_rule.score = self._compute_rule_score(current_rule)
        current_rule.processed = False
        current_rule.avoid_literals = set()

        best_rule = current_rule
        self.interrupted = False
        try:
            candidates = CandidateBeam(self._beamsize)
            candidates.push(current_rule)
            iteration = 1
            while candidates:
                next_candidates = CandidateBeam(self._beamsize)
                getLogger(self._logger).debug('Candidates for iteration %s:' % iteration)
                iteration += 1
                getLogger(self._logger).debug(candidates)
                while candidates:
                    current_rule = candidates.pop()
                    current_rule_literal_avoid = current_rule.avoid_literals

                    for ref in self.language.refine(current_rule):
                        if ref in current_rule_literal_avoid:
                            continue
                        rule = current_rule & ref
                        rule.scores = self._compute_scores_predict(rule)
                        self._stats_evaluations += 1
                        rule.score = self._compute_rule_score(rule)
                        rule.score_future = self._compute_rule_future_score(rule)
                        rule.processed = False
                        rule.avoid_literals = current_rule_literal_avoid

                        if rule.score_future <= best_rule.score:
                            getLogger(self._logger).log(8, '%s %s %s %s [REJECT potential]' % (rule, rule.score, rates(rule), rule.score_future))
                            current_rule_literal_avoid.add(rule.get_literal())
                        else:
                            if next_candidates.push(rule):
                                getLogger(self._logger).log(9, '%s %s %s %s [ACCEPT]' % (rule, rule.score, rates(rule), rule.score_future))
                            else:
                                getLogger(self._logger).log(8, '%s %s %s %s [REJECT beam]' % (rule, rule.score, rates(rule), rule.score_future))

                        if rule.score > best_rule.score:
                            best_rule = rule
                candidates = next_candidates
        except KeyboardInterrupt:
            self.interrupted = True
            getLogger(self._logger).info('LEARNING INTERRUPTED BY USER')
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
            if self.interrupted:
                break

        return hypothesis

    def _compute_rule_score(self, rule):
        return m_estimate(rule, self._m_estimate)

    def _compute_rule_future_score(self, rule):
        return m_estimate_future(rule, self._m_estimate)

    def _select_rule(self, rule):
        pass

    def statistics(self):
        return [('Rule evaluations', self._stats_evaluations)]


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

    def _compute_rule_future_score(self, rule):
        return self._compute_rule_score(rule, future=True)

    def _compute_rule_score(self, rule, future=False):
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
        if future:
            fp_x = fp_prev
        else:
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
                    if future:
                        fp_x = fp_prev
                    else:
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


def main(argv=sys.argv[1:]):
    args = argparser().parse_args(argv)

    logger = 'probfoil'

    log = init_logger(verbose=args.verbose, name=logger)

    # Load input files
    data = DataFile(*(PrologFile(source) for source in args.files))

    if args.probfoil1:
        learn_class = ProbFOIL
    else:
        learn_class = ProbFOIL2

    time_start = time.time()
    learn = learn_class(data, logger=logger, **vars(args))
    time_total = time.time() - time_start

    hypothesis = learn.learn()

    if learn.interrupted:
        print('================ PARTIAL THEORY ================')
    else:
        print('================= FINAL THEORY =================')
    rule = hypothesis
    rules = [rule]
    while rule.previous:
        rule = rule.previous
        rules.append(rule)
    for rule in reversed(rules):
        print (rule)
    print ('==================== SCORES ====================')
    print ('            Accuracy:\t', accuracy(hypothesis))
    print ('           Precision:\t', precision(hypothesis))
    print ('              Recall:\t', recall(hypothesis))
    print ('================== STATISTICS ==================')
    for name, value in learn.statistics():
        print ('%20s:\t%s' % (name, value))
    print ('          Total time:\t%.4fs' % time_total)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('-1', '--det-rules', action='store_true', dest='probfoil1',
                        help='learn deterministic rules')
    parser.add_argument('-m', help='parameter m for m-estimate', type=float,
                        default=argparse.SUPPRESS)
    parser.add_argument('-b', '--beam-size', type=int, default=5,
                        help='size of beam for beam search')
    parser.add_argument('-v', action='count', dest='verbose', default=None,
                        help='increase verbosity (repeat for more)')
    parser.add_argument('--symmetry-breaking', action='store_true',
                        help='avoid symmetries in refinement operator')
    return parser

if __name__ == '__main__':
    main()


