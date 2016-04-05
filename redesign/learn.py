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
                return
        self._candidates.append(candidate)

    def push(self, candidate):
        if candidate.score > self._bottom_score():
            #  We should add it to the beam.
            self._insert(candidate)
            if len(self._candidates) > self._size:
                self._candidates.pop(-1)

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
        s += '==================================\n'
        return s
