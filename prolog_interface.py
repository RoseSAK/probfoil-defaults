# Copyright (C) 2013 Anton Dries (anton.dries@cs.kuleuven.be)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import os
import sys

from util import Timer

from problog.logic import Term, Var, Constant
from problog import get_evaluatable
from problog.program import PrologString
from problog.engine import DefaultEngine
from problog.program import PrologFile


def bin_path(relative):
    return os.path.join(os.path.split(os.path.abspath(__file__))[0], relative)


sys.path.append(bin_path('problog/'))


def lit2term(lit):
    args = []
    for arg in lit.arguments:
        if 'A' <= arg[0] <= 'Z':
            args.append(Var(arg))
        else:
            args.append(Term(arg))
    return Term(lit.functor, *args)


class PrologInterface(object):
    def __init__(self, env, propositional=False):
        self.isPropositional = propositional
        self.datafile = None
        self.gp = None

    def reset(self):
        print ('reset()')

    def enqueue(self, rule):
        parsed_rule = PrologString('q_' + str(rule))

        with Timer(category="evaluate"):
            db = self.datafile.compiled.extend()
            for clause in parsed_rule:
                clause.head = Term('pf_rule', Var('_'), *clause.head.args)
                db += clause

            queries = [Term('pf_rule', Constant(exid), *example) for exid, example in rule.enum_examples()]

            gp = DefaultEngine().ground_all(db, queries=queries)
            results = get_evaluatable().create_from(gp).evaluate()

            rule.initScorePredict()
            for k, v in results.items():
                rule.setScorePredict(int(k.args[0]), v)

    def process_queue(self):
        pass

    def base(self, predicate, arity):
        return self.datafile.base(predicate, arity)

    def values(self, predicate, arity):
        return self.datafile.values(predicate, arity)

    def loadData(self, datafile):
        self.datafile = datafile
        self.datafile.initialize_grounding(self)

    def evaluate_facts(self, target, examples):
        result = [0.0] * len(examples)
        engine = DefaultEngine()

        target_term = lit2term(target)
        queries = [target_term.with_args(*ex) for ex in examples]

        ground_program = engine.ground_all(self.datafile.compiled, queries=queries)
        results = get_evaluatable().create_from(ground_program).evaluate()

        for i, q in enumerate(queries):
            result[i] = results.get(q)
        return result


class DataFile(object):
    def __init__(self, filename):
        self._filename = filename
        self._pl_data = []
        self.target = None
        self.modes = None
        self.dimension = None

        self._read()

    def toProlog(self, linefilter=None):
        pass

    def _read(self):
        raise NotImplementedError('This is an abstract method!')

    def getTarget(self):
        return self.target

    def getModes(self):
        return self.modes

    @classmethod
    def load(cls, filename, **args):
        if filename.endswith('.arff'):
            return ARFFDataFile(filename, **args)
        else:
            return PrologDataFile(filename, **args)


class PrologDataFile(DataFile):
    def __init__(self, filename, **extra):
        super(PrologDataFile, self).__init__(filename)
        self._engine = DefaultEngine()
        self.compiled = self._engine.prepare(PrologFile(self._filename))

    def base(self, predicate, arity):
        base_term = Term(predicate, *([None] * arity))
        result = self._engine.query(self.compiled, Term('base', base_term))
        if len(result) == 0:
            raise Exception("No type information found for predicate '%s'." % base_term.signature)
        elif len(result) > 1:
            raise Exception("Ambiguous information found for predicate '%s'." % base_term.signature)
        else:
            return [result[0][0].args]

    def values(self, predicate, arity):
        base_term = Term(predicate, *([None] * arity))
        result = self._engine.query(self.compiled, base_term)
        return result

    def toProlog(self, linefilter=None):
        if not linefilter: linefilter = lambda x: True
        return '\n'.join(filter(linefilter, self._pl_data))

    def _read(self):
        with open(self._filename) as datafile:
            for line in datafile:
                line = line.strip()
                if line.startswith('%LEARN') or line.startswith('#LEARN'):
                    line = line.split()
                    self.target = line[1]
                    self.modes = line[2:]
                elif line and not line[0] in '%':
                    self._pl_data.append(line)

    def initialize_grounding(self, grounding):
        pass
#
#
# class ARFFDataFile(DataFile):
#     def __init__(self, filename, target_index=None, **extra):
#         self.target_index = target_index
#         super(ARFFDataFile, self).__init__(filename)
#         self.dimension = self.value_matrix.shape[1]
#
#     def toProlog(self, linefilter=None):
#         return '\n'.join(self._pl_data)
#
#     def base(self, predicate, arity):
#         return [('id',)]
#
#     def values(self, predicate, arity):
#         return ([(str(x),) for x in range(0, self.value_matrix.shape[1])])
#
#     def _read(self):
#         import numpy as np
#         value_matrix = []
#
#         dashSplitted = self._filename.strip().split('-')
#         target = dashSplitted[len(dashSplitted) - 1].split('_')[0]
#
#         with open(self._filename) as file_in:
#             line_num = 0
#             counter = 0
#             for line_in in file_in:
#                 line_in = line_in.strip()
#                 if line_in.startswith('@attribute') and self.target_index == None:
#                     if target + ' ' in line_in:
#                         self.target_index = counter
#                     counter += 1
#                 elif line_in and not line_in.startswith('@') and not line_in.startswith('#'):
#                     values = list(map(float, line_in.split(',')))
#                     num_atts = len(values)
#                     value_matrix.append(np.array(values))
#                     self._pl_data += ['%.6f::att%s(%s).' % (float(val), att, line_num) for att, val
#                                       in enumerate(values)]
#                     line_num += 1
#
#         if self.target_index == None:
#             self.target_index = num_atts - 1
#
#         self.target = 'att%s/1' % self.target_index
#         self.modes = ['att%s/+' % att for att in range(0, num_atts) if att != self.target_index]
#
#         self.value_matrix = np.transpose(np.row_stack(value_matrix))
#
#     def initialize_grounding(self, pl):
#         for i, row in enumerate(self.value_matrix):
#             name = 'att%s' % i
#             pl.grounding.addFact(name, row)
#         pl.isPropositional = True
#
#