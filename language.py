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

from functools import total_ordering
from collections import namedtuple, defaultdict
from itertools import product
import math
import time
import sys
from util import Log, Timer

import numpy as np

@total_ordering
class Rule(object) :
    
    ids = 0
    
    def __init__(self, parent) :
        self.parent = parent
        self.literal = None
        self.__score = None
        self.__score_predict = None
        self.__identifier = Rule.ids
        if parent :
            self.__length = len(parent) + 1
        else :
            self.__length = 1
        Rule.ids += 1
        
        self.__eval_nodes = None
        self.__self_nodes = None
        self.examples_to_evaluate = None
        
    def initSelfNodes(self, value=None) :
        if self.knowledge.isPropositional :
            self.__self_nodes = [value]
        else :
            self.__self_nodes = [value] * len( self.examples )
    
    def initEvalNodes(self, value=None) :
        if self.knowledge.isPropositional :
            self.__eval_nodes = [value]
        else :
            self.__eval_nodes = [value] * len( self.examples )
        
    def initScorePredict(self, value=0) :
        if value == 0 :
            self.__score_predict = np.zeros( len(self.examples))
        elif value == 1 :
            self.__score_predict = np.ones( len(self.examples))
        else :
            raise ValueError('Invalid initial value: %s' % value)
            
    def samePredictions(self, other) :
        for x, y in zip(self.getScorePredict(), other.getScorePredict()) :
            if x != y : return False
        return True

    def getSelfNode(self, ex_id) :
        if ex_id == None : ex_id = 0
        return self.__self_nodes[ex_id]
    
    def getEvalNode(self, ex_id) :
        if ex_id == None : ex_id = 0
        return self.__eval_nodes[ex_id]
        
    def setSelfNode(self, ex_id, node_id) :
        if ex_id == None : ex_id = 0        
        self.__self_nodes[ex_id] = node_id
    
    def setEvalNode(self, ex_id, node_id) :
        if ex_id == None : ex_id = 0
        self.__eval_nodes[ex_id] = node_id
        
    def getScorePredict(self, ex_id=None) :
        if ex_id == None :
            return self.__score_predict
        else :
            return self.__score_predict[ex_id]
        
    def setScorePredict(self, ex_id, score) :
        self.__score_predict[ex_id] = score
        
    def __add__(self, literal) :
        """Adds the given literal to the body of the rule and returns the new rule."""
        result = RuleBody(literal, self)
        self.knowledge.enqueue(result)
        
        if not self.learning_problem.PACK_QUERIES :
            
            if not (self.getScorePredict() >= result.getScorePredict() - 1e-10  ).all() :
                print ('PARENT', self, self.score)
                print ('NEW', result, x)

                i = 0
                for x, y in zip(result.getScorePredict(),self.getScorePredict()) :
                    if y >= x + 1e-10  :
                        print (i, y, x, self.examples[i])
                    i += 1
                raise Exception('PREDICTION CANNOT INCREASE!!!!')
                
                
        return result
        
    def _get_language(self) :
        return self.previous.language

    def _get_knowledge(self) :
        return self.previous.knowledge
                
    def _get_target(self) :
        return self.previous.target
        
    def _get_previous(self) :
        return self.parent.previous
        
    def _get_root(self) :
        return self.previous.root
        
    def _get_identifier(self) :
        return self.__identifier # id(self)
    
    def _get_typed_variables(self) :
        return self.target.getTypedVariables(self.language)
        
    def _get_variables(self) :
        return set(map(lambda x : x[0], self.typed_variables))

    def _get_body_variables(self) :
        return set(map(lambda x : x[0], self._body_vars))
        
    def enum_examples(self) :
        if self.knowledge.isPropositional :
            yield (None, () )
        elif self.parent :
            for ex_id, example in self.parent.enum_examples() :
                if self.parent.getScorePredict(ex_id) != 0 :
                    yield ex_id, example
        else :
            for x in enumerate(self.examples) :
                yield x
        
    def _get_examples(self) :
        return self.previous.examples    
                
    def _get_learning_problem(self) :
        return self.previous.learning_problem

    def _get_literals(self) :
        if self.parent and self.literal :
            return self.parent.literals + [self.literal]
        elif self.literal : # Should not happen
            return [self.literal]
        else :
            return []
                
    literals = property( lambda s : s._get_literals() )                
    target = property( lambda s : s._get_target() )
    previous = property( lambda s : s._get_previous() )
    identifier = property( lambda s : s._get_identifier() )
    variables = property( lambda s : s._get_variables() )
    typed_variables = property( lambda s : s._get_typed_variables() )
    body_variables = property( lambda s : s._get_body_variables() )
    language = property( lambda s : s._get_language() )
    knowledge = property( lambda s : s._get_knowledge() )
    examples = property ( lambda s : s._get_examples() )
    learning_problem = property( lambda s : s._get_learning_problem() )
    root = property( lambda s : s._get_root() )
    
    def __repr__(self) :
        return str(self)
    
    def __str__(self) :
        parts = self._str_parts()
        lines = []
        
        for p, prob in parts :
            if prob != 1 :
                prob = ' %% (p=%.5f)' % prob
            else :
                prob = ''
            body = ', '.join(p)
            if body :
                lines.append( str(self.target) + ' :- ' + body + '.' + prob )
            else :
                lines.append( str(self.target) + '.' + prob )
        return '\t'.join(lines)
        
    def __len__(self) :
        return self.__length
    
    def refine(self, update=False) :
        """Generate refinement literals for this rule."""
        # generate refined clauses
        if update :
            # Only generate extensions that use a variable introduced by the last literal
            use_vars = [ vn for vn,vt in self._new_vars ]
            if not use_vars : return [] # no variables available
        else :
            use_vars = None
        return [ literal for literal in self.language.refinements( self.typed_variables, use_vars ) if literal != self.literal ]
        
    def hasScore(self) :
         return self.__score != None
        
    def _get_score(self) :
        if self.__score == None :
#            if self.score_predict == None :
            self.knowledge.process_queue()
            with Timer(category='scoring') :
                self.__score = self.learning_problem.calculateScore(self)
        return self.__score
    
    def _get_score_predict(self) :
        return self.__score_predict
        
    def _set_score_predict(self, predictions) :
        self.__score_predict = predictions
    
    def _get_score_correct(self) :
        return self.previous.score_correct
    
    def count_negated(self) :
        return 0
    
    def __lt__(self, other) :
        return (self.localScore, -len(self), -self.count_negated(), str(self)) < (other.localScore, -len(other), -other.count_negated(), str(other))             
        
    def __eq__(self, other) :
        return str(self) == str(other)
            
    # Calculate maximal achievable significance
    def _calc_max_significance(self) :
        return self._calc_significance(calc_max=True)
        
    # Calculate actual significance
    def _calc_significance(self, calc_max=False) :
        
        self.score.pTP = self.previous.score.TP
        self.score.pFP = self.previous.score.FP
                
        return self.score.calculate_significance(calc_max)
        
    score_correct = property( lambda s: s._get_score_correct() )
    #score_predict = property( lambda s : s._get_score_predict(), lambda s,v : s._set_score_predict(v) )
        
    score = property ( lambda s : s._get_score() )
    
    def _get_global_score(self) :
        if self.learning_problem.USE_RECALL : 
            return self._get_score().recall() 
        elif self.learning_problem.USE_LIMITED_ACCURACY :
            TP = 0.0
            FP = 0.0
            N = 0.0
            M = 0.0
            
            for ex, c, p in zip(self.examples, self.score_correct, self.getScorePredict()) :
                if c > 0 :
                    TP += min(p,c)
                    FP += max(0.0,p-c)
                    N += (1-c) 
                    M += 1.0
                    #print (p,c, min(p,c), max(0.0,p-c), ex)
            return (TP + N - FP) / M
        else : 
            return self._get_score().accuracy()
        
    
    globalScore = property( lambda s : s._get_global_score() )
    localScore = property( lambda s : s._get_score().localScore )
    localScoreMax = property( lambda s : s._get_score().localScoreMax )  
    
    max_significance = property( _calc_max_significance ) # TODO get real value
    significance = property(_calc_significance)
    
    def _getProbability(self) :
        if self.hasScore() :
            return self.score.max_x
        else :
            return 1
    
    probability = property(lambda self: self._getProbability() )
    
    def getTheory(self) :
        if self.previous :
            return self.previous.getTheory() + [ str(self) ]
        else :
            return []
            
    def consolidate(self) : 
        
        if self.probability != 1 :
            p = self.probability
            for i, ex in self.enum_examples() :
                n = self.getEvalNode(i)
                
                if n != None :
                    if self.previous.previous :
                        prev_node = self.previous.getEvalNode(i)
                    else :
                        prev_node = None

                    if prev_node != n :                        
                        if n == 0 :
                            fact_name = 'rule_prob_%s_%s' % (self.identifier,i)
                            new_fact = self.knowledge.grounding.addFact(fact_name, p)
                            self.setEvalNode(i, self.knowledge.grounding.addOrNode( (prev_node, new_fact) ) )
                        else :
                            nodetype, content = self.knowledge.grounding.getNode(abs(n))  
                        
                            if nodetype == 'or' :
                                if content[0] == prev_node :
                                    rule_node = content[1]
                                else :
                                    rule_node = content[0]
                                
                                fact_name = 'rule_prob_%s_%s' % (self.identifier,i)
                                new_fact = self.knowledge.grounding.addFact(fact_name, p)
                                new_node = self.knowledge.grounding.addAndNode( (new_fact, rule_node) )
                            
                                self.setEvalNode(i, self.knowledge.grounding.addOrNode( (prev_node, new_node) ) )
                            else :
                                fact_name = 'rule_prob_%s_%s' % (self.identifier,i)
                                f = self.knowledge.grounding.addFact(fact_name, p)
                                self.setEvalNode( i, self.knowledge.grounding.addAndNode( (f, n) ) )
                            
                        l = self.previous.getScorePredict(i) if self.previous else 0
                        u = self.getScorePredict(i)
                        self.setScorePredict(i, (u-l) * p + l )            
        return self
    
class RuleBody(Rule) :
    """Rule with at least one literal in the body."""
    
    def __init__(self, literal, parent) :
        """Create a new FOIL rule by adding the given literal to the give parent rule."""
        super(RuleBody, self).__init__(parent)
        self.literal = literal
        
        old_vars = parent.typed_variables
            
        current_vars = set(literal.getTypedVariables(self.language))
        self._all_vars = old_vars | current_vars
        self._new_vars = current_vars - old_vars
        self._body_vars = parent._body_vars | current_vars
        
    def _get_typed_variables(self) :
        return self._all_vars
        
    def _str_parts(self) :
        par = self.parent._str_parts()
        if self.hasScore() :
            par[-1][1] = self.probability
        else :
            par[-1][1] = 1
        par[-1][0].append(str(self.literal))
        return par
                
class RuleHead(Rule) :
    """Rule with empty body."""
    
    def __init__(self, previous) :
        """Adds a new empty rule with the given head to the given FOIL rule set."""
        super(RuleHead,self).__init__(None)
        self.__previous = previous.consolidate()
        
        current_vars = set(self.target.variables)
        self._all_vars = current_vars
        self._new_vars = set([])
        self._body_vars = set([])
    
        self.initScorePredict(1)
        
        self.initEvalNodes(0)
        self.initSelfNodes(0)
                        
    def _get_target(self) :
        return self.previous.target
    
    def _get_previous(self) :
        return self.__previous
        
    def _str_parts(self) :
        par = []
#        par = self.previous._str_parts()
        if self.hasScore() :
            par.append( [[], self.probability ] )
        else :
            par.append( [[], 1 ] )
        return par
        
class RootRule(Rule) :
    """Rule with fail in the body."""
    
    def __init__(self, target, learning_problem) :
        self.__target = target
        self.__learning_problem = learning_problem
        self.__examples = []
        super(RootRule,self).__init__(None)
    
    def _get_language(self) :
        return self.__learning_problem.language
    
    def _get_target(self) :
        return self.__target
    
    def _get_previous(self) :
        return None
        
    def _get_identifier(self) :
        return None
        
    def _get_knowledge(self) :
        return self.__learning_problem.knowledge
        
    def _get_examples(self) :
        return self.__examples

    def _get_learning_problem(self) :
        return self.__learning_problem
        
    def _get_root(self) :
        return self
        
    def _get_score_correct(self) :
        return self.__score_correct
        
    def initialize(self, examples = None) :
        # 1) Determine types of arguments of self.target
        #   => requires access to 'language'
        # argument_types = self.language.getArgumentTypes( self.target ) 
        
        # 2) Retrieve values for types of arguments 
        #   => requires access to 'language'

        # 3) Populate list of examples
        #   => carthesian product of values
        
        if examples == None :
            self.__examples = self.language.getArgumentValues( self.target )
        else :
            self.__examples = examples
        
        # 4) Populate scores => fact probabilities
        #   => requires access to 'facts'
        #   => scores are 1 - p where p is probability of the fact
        
        self.knowledge.enqueue( self )  
        self.knowledge.process_queue()
        self.__score_correct = self.getScorePredict()
        
        if self.learning_problem.NO_CLOSED_WORLD :
            new_examples = []
            new_score_correct = []
            for i, s in enumerate(self.__score_correct) :
                if s > 0 :
                    new_examples.append(self.__examples[i])
                    new_score_correct.append(s)
            self.__examples = new_examples
            self.__score_correct = new_score_correct
        
        if self.learning_problem.VERBOSE > 3 :
            print ('Number of examples:', len(self.examples))    
        
        self.initScorePredict()
        self.eval_nodes = None
        self.self_nodes = None
            
    def _str_parts(self) :
        return []
        
    def __str__(self) :
        return str(self.target) + ' :- fail.'

class Literal(object) :
    
    def __init__(self, functor, arguments, is_negated=False) :
        self.functor = functor
        self.arguments = arguments
        self.is_negated = is_negated
        
    def withArgs(self, args) :
        return Literal(self.functor, args, self.is_negated)
        
    def _is_var(self, arg) :
        return arg[0] in '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
    def _get_variables(self) :
        return set(filter(self._is_var,self.arguments))
        
    def getTypedVariables(self, language) :
        types = language.getArgumentTypes(self)
        result = []
        for vn, vt in zip(self.arguments, types) :
            if self._is_var(vn) :
                result.append( (vn, vt) )
        return set(result)
        
    variables = property( _get_variables )  
    arity = property (lambda s : len(s.arguments) )
    key = property( lambda s : ( s.functor, s.arity ) )
    
    def __repr__(self) :
        r = self.functor
        if self.arguments :
            r += '(' + ','.join(map(str,self.arguments)) + ')'
        else :
            r += ''
        if self.is_negated :
            return 'not(%s)'  % r
        else :
            return r
            
    def __neg__(self) :
        return Literal(self.functor, self.arguments, not self.is_negated)
            
    def __abs__(self) :
        if self.is_negated :
            return -self
        else :
            return self
            
    def __hash__(self) :
        return hash(str(self))
            
    def __eq__(self, other) :
        if other == None :
            return False
        else :
            return self.functor == other.functor and self.arguments == other.arguments and self.is_negated == other.is_negated

class Language(object) :
    
    def __init__(self) :
        self.__types = {}
        self.__values = defaultdict(set)
        self.__modes = {}
        self.__targets = []
        self.__varcount = 0
        
    def initialize(self, knowledge) :
        predicates = list(self.__modes) + self.__targets
        for predicate, arity in predicates :
            # Load types
            types = knowledge.base( predicate, arity )
            if len(types) == 1 :
                types = types[0]
                self.setArgumentTypes( Literal( predicate, types ) )
            elif len(types) > 1 :
                raise Exception("Multiple 'base' declarations for predicate '%s/%s'!" % (predicate,arity))
            else :
                self.setArgumentTypes( Literal( predicate, ['id'] * arity ) )
                print ("Missing 'base' declaration for predicate '%s/%s'!" % (predicate,arity), file=sys.stderr)
            
            values = list(zip(*knowledge.values( predicate, arity )))
            for tp, vals in zip(types,values) :
                self.__values[tp] |= set(vals)
        
    def addTarget(self, predicate, arity) :
        self.__targets.append( (predicate, arity) )
        
    def addValue(self, typename, value) :
        self.__values[typename].add(value)
        
    def setArgumentTypes(self, literal) :
        self.__types[ literal.key ] = literal.arguments
        
    def setArgumentModes(self, literal) :
        self.__modes[ literal.key ] = literal.arguments
        
    def getArgumentTypes(self, literal=None, key=None) :
        if literal : key = literal.key
        return self.__types.get(key,[])
            
    def getArgumentModes(self, literal=None, key=None) :
        if literal : key = literal.key
        return self.__modes.get(key,[])
                
    def getTypeValues(self, typename) :
        return self.__values.get(typename,[])
        
    def getArgumentValues(self, literal) :
        types = self.getArgumentTypes(literal)
        return list( product( *[ self.getTypeValues(t) for t in types ] ) )
        
    def newVar(self) :
        self.__varcount += 1
        return 'Var_' + str(self.__varcount)
        
    def refinements(self, variables, use_vars) :
        existing_variables = defaultdict(list)
        for varname, vartype in variables :
            existing_variables[vartype].append( varname ) 
        
        if use_vars != None :
            use_vars = set(use_vars) # set( [ varname for varname, vartype in use_vars ] )
        
        for pred_id in self.__modes :
            pred_name = pred_id[0]
            arg_info = list(zip(self.getArgumentTypes(key = pred_id), self.getArgumentModes(key = pred_id)))
            for args in self._build_refine(existing_variables, True, arg_info, use_vars) :
                new_lit = Literal(pred_name, args)
                yield new_lit
            
            for args in self._build_refine(existing_variables, False, arg_info, use_vars) :
                new_lit = Literal(pred_name, args, True)                
                yield new_lit
    
    def _build_refine_one(self, existing_variables, positive, arg_type, arg_mode) :
        if arg_mode in ['+','-'] :
            for var in existing_variables[arg_type] :
                yield var
        if arg_mode == '-' and positive :
            yield '#'
        if arg_mode == 'c' :
            if positive :
                for val in self.kb.types[arg_type] :
                    yield val
            else :
                yield '_'
    
    def _build_refine(self, existing_variables, positive, arg_info, use_vars) :
        if arg_info :
            for arg0 in self._build_refine_one(existing_variables, positive, arg_info[0][0], arg_info[0][1]) :
                if use_vars != None and arg0 in use_vars :
                    use_vars1 = None
                else :
                    use_vars1 = use_vars 
                for argN in self._build_refine(existing_variables, positive, arg_info[1:], use_vars1) :
                    if arg0 == '#' :
                        yield [self.newVar()] + argN
                    else :
                        yield [arg0] + argN
        else :
            if use_vars == None :
                yield []
