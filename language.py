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
from collections import defaultdict
from itertools import product
import sys
from util import Log, Timer

import numpy as np

from problog.logic import Term, is_variable, Var


@total_ordering
class Rule(object):
    
    ids = 0
    
    def __init__(self, parent):
        self.parent = parent
        self.literal = None
        self.__score = None
        self.__score_predict = None
        self.__identifier = Rule.ids
        if parent:
            self.__length = len(parent) + 1
        else:
            self.__length = 1
        Rule.ids += 1
        
        self.__eval_nodes = None
        self.__self_nodes = None
        self.example_filter = None
        self.examples_to_evaluate = None
        self.invalid_scores = False
        
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
            if self.learning_problem.VERBOSE > 6 :
                print ('Evaluating rule', result, '...')
            self.knowledge.process_queue()    
#            ss = self.score
            # print (ss)
            # if not (self.getScorePredict() >= result.getScorePredict() - 1e-10  ).all() :
            #     print ('PARENT', self, self.score)
            #     print ('NEW', result, x)
            # 
            #     i = 0
            #     for x, y in zip(result.getScorePredict(),self.getScorePredict()) :
            #         if y >= x + 1e-10  :
            #             print (i, y, x, self.examples[i])
            #         i += 1
            #     raise Exception('PREDICTION CANNOT INCREASE!!!!')
                
                
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
        return self.language.getTypedVariables(self.target)
        
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
        elif self.example_filter != None :
            for i in self.example_filter :
                yield i, self.examples[i]
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
        
    def refine_rpf(self) :
        return [MultiLiteral( *l ) for l in self.language.RPF_paths]

    
    def refine(self, update=False) :
        """Generate refinement literals for this rule."""
        RPF = self.learning_problem.RPF
        if RPF and not self.parent  :
            if not update :
                return self.refine_rpf()
            else :
                return []
#                 use_vars = None
#                 return [ literal for literal in self.language.refinements( self.typed_variables, use_vars ) if literal != self.literal ]
        elif RPF and self.parent :
            # Determine whether this is the first extension after RPF
            if isinstance(self.literal, MultiLiteral) : # YES
                return [ literal for literal in self.language.refinements( self.typed_variables, None ) if not literal in self.literal ]
            else :
                return []
        else :        
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
                                fact_name = 'rule_prob_%s_%s' % (self.identifier,i)
                                new_fact = self.knowledge.grounding.addFact(fact_name, p)
                                new_node = self.knowledge.grounding.addAndNode( (new_fact, n) )
                            
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
            
        current_vars = set(self.language.getTypedVariables(literal))
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
        
        current_vars = set(self.target.variables())
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
        
    def initialize_from(self, rootrule) :
      self.__examples = rootrule.examples
      self.__score_correct = rootrule.score_correct
      self.initScorePredict()
      self.eval_nodes = None
      self.self_nodes = None
        
    def initialize(self, examples = None) :
        # 1) Determine types of arguments of self.target
        #   => requires access to 'language'
        # argument_types = self.language.getArgumentTypes( self.target ) 
        
        # 2) Retrieve values for types of arguments 
        #   => requires access to 'language'

        # 3) Populate list of examples
        #   => carthesian product of values
        if examples is None:
            self.__examples = self.language.getArgumentValues(self.target)
        else :
            self.__examples = examples
        
        # 4) Populate scores => fact probabilities
        #   => requires access to 'facts'
        #   => scores are 1 - p where p is probability of the fact
        
        if self.learning_problem.VERBOSE > 3 :
            print ("Found %s potential examples." % len(self.examples))
        
        self.__score_correct = self.knowledge.evaluate_facts(self.target, self.examples)
        
        # self.knowledge.enqueue( self )  
        # self.knowledge.process_queue()
        # self.__score_correct = self.getScorePredict()
        
        if self.learning_problem.BALANCE_NEGATIVE or self.learning_problem.BALANCE_NEGATIVE_BIASED :
            biased = self.learning_problem.BALANCE_NEGATIVE_BIASED 
            #   first select combinations of values as they appear in the positive examples
            
            import random
            if self.learning_problem.RANDOM_SEED != None :
              random.seed(self.learning_problem.RANDOM_SEED)
            new_examples = []
            new_score_correct = []
            neg_examples = []
            
            if biased :
                values_in_positive = [ set([]) for x in self.examples[0] ]
            
            neg = 0.0
            for i, s in enumerate(self.__score_correct) :
                if s > 0 :
                    new_examples.append(self.__examples[i])
                    new_score_correct.append(s)
                    neg += (1-s)
                    
                    if biased :
                        for j,x in enumerate(self.__examples[i]) :
                            values_in_positive[j].add(x)
                        
                else :
                    neg_examples.append(self.__examples[i])

            random.shuffle(neg_examples)
            if biased :
                positives = set(new_examples)
                negatives = list(set( product( *values_in_positive ) ) - positives)
                random.shuffle(negatives)
                if self.learning_problem.VERBOSE > 3 :
                    print ("Using biased sampling: %s biased examples available." % len(negatives))
            
                
                neg_examples = negatives + neg_examples 
                
#                negatives = argumentValue
                    

            num_negs = int((len(new_examples) - neg) * self.learning_problem.CLASS_BALANCE)
            assert(num_negs >= 0)
            neg_examples = neg_examples[:num_negs]
            
            if self.learning_problem.VERBOSE > 3 :
                print ("Found %s positive examples with a total weight of %s." % (len(new_examples), (len(new_examples)-neg)))
            
            self.__examples = new_examples + neg_examples
            self.__score_correct = new_score_correct + [0] * num_negs 
            
            
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
            print ('Number of examples: %s' % (len(self.examples),))    
        
        if self.learning_problem.GENERATE_DATA != None :
            import re
            re_target = re.compile( '(.*::)?' + self.target.functor + '[(].*[)].')
            with open(self.learning_problem.GENERATE_DATA,'w') as f :
                line_filter = lambda x : re_target.match( x ) == None 
            
                print (self.knowledge.datafile.toProlog( line_filter ), file=f)
                for ex, sc in zip(self.__examples, self.__score_correct) :
                    print ('%s::%s.' % ( sc, self.target.with_args(*ex) ), file=f)
            
            # Stop here
            sys.exit(0)
        
        self.initScorePredict()
        self.eval_nodes = None
        self.self_nodes = None
        
        if (self.learning_problem.RPF) :
            if self.learning_problem.VERBOSE > 0 : 
                print ('Performing relational path finding...')
            self.language.rpf_init(self)
            
    def _str_parts(self) :
        return []
        
    def __str__(self) :
        return str(self.target) + ' :- fail.'

from itertools import chain
class MultiLiteral(object) :
    
    def __init__(self, *literals) :
        self.literals = literals
        self.arguments = None
                
    def withAssign(self, assign) :
        return MultiLiteral( *[ lit.withAssign(assign) for lit in self.literals ] )
                
    def _get_variables(self) :
        return set(chain( *[ l.variables for l in self.literals ] ) )
        
    def getTypedVariables(self, language) :
        return set(chain( *[ l.getTypedVariables(language) for l in self.literals ] ) )
        
    variables = property( _get_variables )  
    #arity = property (lambda s : len(s.arguments) )
    #key = property( lambda s : ( s.functor, s.arity ) )
    
    def __repr__(self) :
        return ', '.join(map(repr, self.literals))
                                    
    def __hash__(self) :
        return hash(str(self))
            
    def __eq__(self, other) :
        if other == None :
            return False
        else :
            return self.literals == other.literals

    def __contains__(self, literal) :
        return literal in self.literals


class Language(object) :
    
    def __init__(self) :
        self.__types = {}
        self.__values = defaultdict(set)
        self.__modes = {}
        self.__targets = []
        self.__varcount = 0
        self.learning_problem = None
        
    modes = property( lambda s : s.__modes )
        
    def initialize(self, knowledge):
        predicates = list(self.__modes) + self.__targets
        for predicate, arity in predicates:
            # Load types
            types = knowledge.base(predicate, arity)
            if len(types) == 1:
                types = types[0]
                self.setArgumentTypes(Term(predicate, *types))
            elif len(types) > 1:
                raise Exception("Multiple 'base' declarations for predicate '%s/%s'!" % (predicate, arity))
            else :
                self.setArgumentTypes(Term(predicate, *([Term('id')] * arity)))
                print ("Missing 'base' declaration for predicate '%s/%s'!" % (predicate, arity), file=sys.stderr)
            
            values = list(zip(*knowledge.values(predicate, arity)))
            for tp, vals in zip(types, values):
                self.__values[tp] |= set(vals)
        
    def addTarget(self, predicate, arity) :
        self.__targets.append( (predicate, arity) )
        
    def addValue(self, typename, value) :
        self.__values[typename].add(value)
        
    def setArgumentTypes(self, literal) :
        self.__types[(literal.functor, literal.arity)] = literal.args
        
    def setArgumentModes(self, literal) :
        self.__modes[(literal.functor, literal.arity)] = literal.args
        
    def getArgumentTypes(self, literal=None, key=None) :
        if literal:
            key = (literal.functor, literal.arity)
        return self.__types.get(key,[])
            
    def getArgumentModes(self, literal=None, key=None) :
        if literal:
            key = (literal.functor, literal.arity)
        return self.__modes.get(key,[])
                
    def getTypeValues(self, typename):
        return self.__values.get(typename, [])
        
    def getArgumentValues(self, literal):
        types = self.getArgumentTypes(literal)
        return list( product( *[ self.getTypeValues(t) for t in types ] ) )

    def getTypedVariables(self, literal):
        types = self.getArgumentTypes(literal)
        result = []
        for vn, vt in zip(literal.args, types) :
            if is_variable(vn) or vn.is_var():
                result.append((vn, vt))
        return set(result)



        
    def newVar(self) :
        self.__varcount += 1
        return Var('Var_' + str(self.__varcount))
        
    def relationsByConstant(self, constant ) :
        kb = self.learning_problem.knowledge

        c,vt = constant
        
        res = []
        varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        goals = []
        for pred_id in self.__modes:
            if pred_id[1] < 2:
                continue
            index = [i for i, x in enumerate(self.getArgumentTypes(key=pred_id)) if x == vt]
            for i in index:
                vars = [varnames[j] for j in range(0, pred_id[1])]
                args = vars[:]
                args[i] = c
                goals.append(Term(pred_id[0], *args))
        return ( map( Literal.parse, kb.query_goals(goals) ) )

#        print (res)
#         
#         res = []
#         res += [ Literal('parent',(c,r[0])) for r in kb.query( Literal('parent', (c,'Y')), 'Y' ) ] 
#         res += [ Literal('parent',(r[0],c)) for r in kb.query( Literal('parent', ('Y',c)), 'Y' ) ] 
#         res += [ Literal('married',(c,r[0])) for r in kb.query( Literal('married', (c,'Y')), 'Y' ) ] 
#         res += [ Literal('married',(r[0],c)) for r in kb.query( Literal('married', ('Y',c)), 'Y' ) ] 
        with Log('nodes', constant=c, extra=res) : pass
#         print ('A',res)
        return res
        
    def rpf_eval_path(self, path, rule, prd ) :
        #print ('EVALUATING:', path)

        evaluation_process = 4

        if evaluation_process == 1 :
            predict = [0] * len(rule.examples) 
            qs = []
            for i, ex in enumerate(rule.examples) :
        
                if rule.score_correct[i] - prd[i] > 0 :
                    assign = dict( zip( rule.target.arguments, ex ) )   
                    qs.append( MultiLiteral(*path).withAssign(assign) )
        
            j = 0
            res = rule.knowledge.verify(qs)
            for i, ex in enumerate(rule.examples) :
                if rule.score_correct[i] - prd[i] > 0 :
                    if res[j] == '1' : predict[i] = 1
                    j+=1

            return predict
                
        elif evaluation_process == 2 :


            predict = [0] * len(rule.examples) 
            for i, ex in enumerate(rule.examples) :
                if prd[i] < 1 :
                    assign = dict( zip( rule.target.arguments, ex ) )
                    eval = rule.knowledge.query( MultiLiteral(*path).withAssign(assign), ('t') ) 
                    if eval : predict[i] = 1
            return predict

        elif evaluation_process == 3 :
            eval = (set(map(tuple, rule.knowledge.query( path, rule.target.arguments ))) )
            print (len(eval))
            predict = [0] * len(rule.examples) 
            for i, ex in enumerate(rule.examples) :
                if prd[i] < 1 :
                    if ex in eval :
                        predict[i] = 1
            return predict
    
        else :
            # TODO only evaluate rule for remaining positive examples
            new_rule = RuleHead(previous=rule) + MultiLiteral( *path)
            s = new_rule.score  # Force computation
            return new_rule.getScorePredict()
        
    def rpf_init(self, rule, num_paths=0, pos_threshold=0.1, max_level=2) :

     with Timer(category="rpf") :
      with Log('rpf_init') :
        R = RPF(rule)
        self.RPF_paths = list(R.paths)
        
        
    def refinements(self, variables, use_vars) :
    
        existing_variables = defaultdict(list)
        existing_variables_set = set([])
        for varname, vartype in variables :
            existing_variables[vartype].append( varname ) 
            existing_variables_set.add(varname)
    
        if use_vars is not None:
            use_vars = set(use_vars)  # set( [ varname for varname, vartype in use_vars ] )
    
        for pred_id in self.__modes:
            pred_name = pred_id[0]
            arg_info = list(zip(self.getArgumentTypes(key=pred_id), self.getArgumentModes(key=pred_id)))
            for args in self._build_refine(existing_variables, True, arg_info, use_vars):
                new_lit = Term(pred_name, *args)
                if new_lit.variables() & existing_variables_set:
                    yield new_lit
        
            if not self.learning_problem.NO_NEGATION:
                for args in self._build_refine(existing_variables, False, arg_info, use_vars):
                    new_lit = -Term(pred_name, *args)
                    if new_lit.variables() & existing_variables_set:
                        yield new_lit
    
    def _build_refine_one(self, existing_variables, positive, arg_type, arg_mode):
        arg_mode = str(arg_mode)
        if arg_mode in '+-':
            for var in existing_variables[arg_type] :
                yield var
        if arg_mode == '-' and positive and not self.learning_problem.RPF:
            yield '#'
        if arg_mode == 'c':
            if positive:
                for val in self.kb.types[arg_type]:
                    yield val
            else:
                yield '_'
    
    def _build_refine(self, existing_variables, positive, arg_info, use_vars) :
        if arg_info :
            for arg0 in self._build_refine_one(existing_variables, positive, arg_info[0][0], arg_info[0][1]) :
                if use_vars != None and arg0 in use_vars :
                    use_vars1 = None
                else :
                    use_vars1 = use_vars 
                for argN in self._build_refine(existing_variables, positive, arg_info[1:], use_vars1) :
                    if arg0 == '#':
                        yield [self.newVar()] + argN
                    else :
                        yield [arg0] + argN
        else :
            if use_vars == None :
                yield []

import os
def bin_path(relative) :
    return os.path.join( os.path.split( os.path.abspath(__file__) )[0], relative )

class RPF(object) :
    
    def __init__(self, rule) :
        self.rule = rule
        self.paths = None
        self.rule.knowledge._create_file_det()
        self.evaluate(self.rule)
                
    def evaluate(self, rule) :
        assert(rule.target.arity == 2)
        
        modes = str([ Literal( pred_id[0], ['_'] * pred_id[1]) for pred_id in rule.language.modes if pred_id[1] == 2 ])
        
        threshold = 0.1         # discard example if it's score falls below this value
        eval_cutoff = 20        # evaluate paths when less than this number of examples is removed => None means always evaluate
        stop_threshold = 0.8    # stop when this percentage of positive weight is covered by the rules
        maxl = 4

        stop_threshold = sum(rule.score_correct) * (1.0 - stop_threshold)

        examples = [ i for i, s in enumerate(rule.score_correct) if s > threshold ] 
        predicts = [ 0 ] * len(rule.examples)
        
        paths = set([])
        paths_to_eval = set([])
        paths_to_discard = set([])
        
        eval_rule = RuleHead(previous=rule)
        
        while examples :
        
            # 1) Order constants by total score of examples in which they occur
            scores = defaultdict(float)
            for i in examples :
                ex = rule.examples[i]
                s = rule.score_correct[i] - predicts[i]
                for j,e in enumerate(ex) :
                    scores[(e,j)] += s
            scores = list(sorted([ (s,) + e for e, s in scores.items() ]))
        
            # 2) Select the best constant
            score, constant, position = scores[-1]
        
            # 3) Construct target 
            args = ['_'] * rule.target.arity
            args[ position ] = constant
            target = rule.target.with_args(*args)
            varargs = rule.target.arguments[:]
            v1 = varargs.pop( position )
            varargs = [v1] + varargs
        
            # 4) Construct list of constants
            constants = []
            remaining_examples = []
            for i in examples :
                ex = rule.examples[i]
                if ex[position] == constant :
                    constants.append( ex[1-position] )
                else :
                    remaining_examples.append( i )
            if rule.learning_problem.VERBOSE > 9 :
                print ('RPF:', target, '-', len(examples), len(remaining_examples))
            with Log('step', target=target, before=len(examples), after=len(remaining_examples)) : pass   
                
            # 5) Find paths for the given constants
            current_paths = self._call_yap( target, varargs, modes, constants, maxl=maxl )
            # 6) Evaluate paths (probabilistically)
            # Only evaluate on remaining examples
            for path in current_paths :
                if not path in paths :
                    if rule.learning_problem.VERBOSE > 9 :
                        print ('PATH FOUND', path)
                    with Log('found', path=path) : pass
                    paths.add(path)
                    paths_to_eval.add(path)
                            
            eval_rule.example_filter = remaining_examples
            if eval_cutoff == None or len(examples) - len(remaining_examples) < eval_cutoff :                    
                for path in paths_to_eval :
                    new_rule = eval_rule + MultiLiteral( *path)
                    s = new_rule.score  # Force computation
                    s = sum(new_rule.getScorePredict())
                    if new_rule.invalid_scores : # Evaluation error
                        paths_to_discard.add(path)   
                        if rule.learning_problem.VERBOSE > 9 :
                            print ('PATH DISCARDED', path)

                                             
                    else :
                        for i in remaining_examples :
                            predicts[i] = max( predicts[i], new_rule.getScorePredict(i) )
                paths_to_eval = set([])

            examples = [ i for i in remaining_examples if rule.score_correct[i] - predicts[i] > threshold ] 
            
            remain_score = sum([ rule.score_correct[i] - predicts[i] for i in remaining_examples ])
            if remain_score < stop_threshold :
                break
        
        if rule.learning_problem.VERBOSE > 9 :
            print (paths)
        with Log('paths', paths='\n'.join( map(str,paths )), discard='\n'.join( map(str,paths_to_discard ))) : pass
        
        self.paths = list(paths - paths_to_discard)

        
    def _call_yap( self, target, args, modes, constants, maxl=-1) :

        DATAFILE = self.rule.knowledge.datafile_det
    
        RPF_PROLOG = bin_path('rpf.pl')
        import subprocess
#        try :
        #print (' '.join(['yap', "-L", RPF_PROLOG , '--', DATAFILE, str(maxl), "'" + str(target) + "'", "'" + modes + "'" ] + constants))

        output = subprocess.check_output(['yap', "-L", RPF_PROLOG , '--', DATAFILE, str(maxl), str(target), modes ] + constants )

#         print ('#### OUTPUT #####')
#         print (output)
#         print ('#################')
        

        paths = []
        for line in output.strip().split('\n') :
            parts = line.split('|')
            varname = parts[0]
            if len(parts) <= 1 or not '(' in parts[1] :
                continue
            path = tuple(sorted(map( lambda x : Literal.parse(x).withAssign( { varname : args[1], 'V_0' : args[0] } ), parts[1:] ) ))
            paths.append(path)
        return paths
        
#        except subprocess.CalledProcessError :
#            print ('Error during rpf', file=sys.stderr)
#            with Log('error', context='grounding') : pass

 