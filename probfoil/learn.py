from problog.logic import Var, Term
from itertools import product
from problog.util import Timer


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

class CandidateSet(object):

    def __init__(self):
        pass

    def push(self, candidate):
        raise NotImplementedError('abstract method')

    def pop(self):
        raise NotImplementedError('abstract method')

    def __bool__(self):
        raise NotImplementedError('abstract method')


class BestCandidate(CandidateSet):

    def __init__(self, candidate=None):
        CandidateSet.__init__(self)
        self.candidate = candidate

    def push(self, candidate):
        if self.candidate is None or self.candidate.score < candidate.score:
            self.candidate = candidate

    def pop(self):
        if self.candidate is not None:
            return self.candidate
        else:
            raise IndexError('Candidate set is empty!')

    def __bool__(self):
        return not self.candidate is None


class CandidateBeam(CandidateSet):

    def __init__(self, size):
        CandidateSet.__init__(self)
        self._size = size
        self._candidates = []

    def _bottom_score(self):
        if self._candidates:
            return self._candidates[-1].score
        else:
            return -1e1000

    def _insert(self, candidate):
        for i, x in enumerate(self._candidates):
            if x.score < candidate.score:
                self._candidates.insert(i, candidate)
                return False
            elif x.is_equivalent(candidate):
                raise ValueError('duplicate')
        self._candidates.append(candidate)
        return True

    def push(self, candidate):
        """Adds a candidate to the beam.

        :param candidate: candidate to add
        :return: True if candidate was accepted, False otherwise
        """
        if candidate.score > self._bottom_score():
            #  We should add it to the beam.
            try:
                is_last = self._insert(candidate)
                if len(self._candidates) > self._size:
                    self._candidates.pop(-1)
                    return not is_last
            except ValueError:
                return False
            return True
        return False

    def pop(self):
        return self._candidates.pop(0)

    def __bool__(self):
        return bool(self._candidates)

    def __nonzero__(self):
        return bool(self._candidates)

    def __str__(self):
        s = '==================================\n'
        for candidate in self._candidates:
            s += str(candidate) + ' ' + str(candidate.score) + '\n'
        s += '=================================='
        return s
