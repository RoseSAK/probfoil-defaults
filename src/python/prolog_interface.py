import sys

sys.path.append('../../')

import os
import time
import math

from util import Log


from collections import namedtuple, defaultdict
from itertools import product
from functools import total_ordering

chi2_cdf = lambda x : math.erf(math.sqrt(x/2))
def calc_significance(s, low=0.0, high=100.0, precision=1e-8) :
    v = (low+high)/2
    r = chi2_cdf(v)
    if -precision < r - s < precision :
        return v
    elif r > s :
        return calc_significance(s, low, v)
    else :
        return calc_significance(s, v, high)


#from core import PrologEngine, Clause, Conjunction

class Beam(object) :
    
    def __init__(self, size, allow_equivalent=False) :
        self.size = size
        self.content = []
        self.allow_equivalent = allow_equivalent
         
    def create(self) :
        return Beam(self.size, self.allow_equivalent) 
       
    def __iter__(self) :
        return iter(self.content)
        
    def push(self, obj, active) :
        if len(self.content) == self.size and obj < self.content[-1][0] : return False
        
        is_last = True
        
        p = len(self.content) - 1
        self.content.append( (obj, active) )
        while p >= 0 and (self.content[p][0] == None or self.content[p][0] < self.content[p+1][0]) :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
            is_last = False
        
        if not self.allow_equivalent and len(self.content) > 1 :
            r1, rf1 = self.content[p]
            r2, rf2 = self.content[p+1]
            
            r1scores = r1.score_predict
            r2scores = r2.score_predict
            
            if r1.localScore == r2.localScore and r1scores == r2scores :
                if rf1 != None and rf2 != None and len(rf1) > len(rf2) : #len(r1.variables) > len(r2.variables) :                
                    best, worst = r1, r2
                    self.content[p+1] = self.content[p]
                else :
                    best, worst = r2, r1
                with Log('beam_equivalent', best=best, worst=worst) : pass                
                self.content.pop(p)
        
        popped_last = False
        while len(self.content) > self.size :
            self.content.pop(-1)
            popped_last = True
            
        return not (is_last and popped_last)
    
    def peak_active(self) :
        i = 0
        while i < len(self.content) :
            if self.content[i][-1] :
                yield self.content[i]
                i = 0
            else :
                i += 1
                
    def has_active(self) :
        for r, act in self :
            if act != None : return True
        return False
    
    def pop(self) :
        self.content = self.content[1:]
        
    def __str__(self) :
        res = ''
        for c, r in self.content :
            res += str(c) + ': ' + str(c.score) +  ' | ' + str(r) + '\n'
        return res
        
    def toXML(self) :
        res = ''
        for c, r in self.content :
            if r == None :
                res +=  '<record rule="%s" score="%s" localScore="%s" refine="NO" />\n' % (c,c.score, c.localScore)
            else :
                res +=  '<record rule="%s" score="%s" localScore="%s" refine="%s" />\n' % (c,c.score, c.localScore, '|'.join(map(str,r)))
        return res


class LearningProblem(object) :
    
    def __init__(self, language, knowledge) :
        self.language = language
        self.knowledge = knowledge
        
        self.BEAM_SIZE = 5
        self.VERBOSE = True
        self.SIGNIFICANCE = calc_significance(0.99)
        
    def calculateScore(self, rule) :
        raise NotImplementedError('calculateScore')

    def learn(self, H) :
        """Core FOIL learning algorithm.
    
        H - initial hypothesis (Rule (e.g. FalseRule))
        """
    
        # Find clauses as long as stopping criterion is not met or until maximal score (1.0) is reached.
        while H.globalScore < 1.0 :   # this test is not required for correctness (alternative: while True)
       
          with Log('learn_rule', _timer=True):
            
            # Find best clause that refines this hypothesis
            new_H = self.best_clause( H )
            
            # Prune rule 
            #   Explanation: in case of equal score the search procedure prefers rules with more variables
            #                once the search is over we prefer shorter rules
            while new_H.parent and new_H.parent.localScore >= new_H.localScore :
                new_H = new_H.parent
        
            if self.VERBOSE : print ('RULE FOUND:', new_H, new_H.globalScore)
        
            # Log progress
            with Log('rule_found', rule=new_H, score=new_H.score) : 
                pass
            with Log('stopping_criterion', old_score=H.globalScore, new_score=new_H.globalScore, full_score=new_H.score) : 
                pass
        
            # TODO check significance level?
            # Check stopping criterion
            if (H.globalScore >= new_H.globalScore) :
                # Clause does not improve hypothesis => remove it and stop
                break
            else :
                # Clause improves hypothesis => continue
                H = new_H
        return H

    def best_clause(self, current_rule ) :
        """Find the best clause for this hypothesis."""
    
        # We use beam search; initialize beam
        beam = Beam(self.BEAM_SIZE)    
    
        # Create the new rule ( target <- )
        init_rule = RuleHead(previous=current_rule)
    
        # Calculate initial set of refinements
        refinements = list(init_rule.refine())
    
        # Add clause to beam (with empty score)
        beam.push( init_rule, refinements )
    
        try :
            # While there are untested refinements in the beam.
            while beam.has_active() :
              with Log('iteration', _timer=True) :
          
                # Create the next beam
                new_beam = beam.create()
        
                # Run through old beam and process its content
                for old_rule, refs in beam :
                
            
                  with Log('refining', rule=old_rule, score=old_rule.score, localScore=old_rule.localScore, _timer=True) :
            
                    # Add current rule to beam and mark it as tested (refinements=None)
                    new_beam.push( old_rule, None )
                        
                    # Update scores of available refinements and add new refinements if a new variable was introduced.
                    new_rules = self.update_refinements(old_rule, refs)
            
                    # Extract refinement literals
                    new_refs = [ r.literal for r in new_rules ]
                
                    #print ('NEW_REFS:', new_refs)
            
                    # Add rules to new beam (new_refs are ordered by score, descending)
                    for i, new_rule in enumerate(new_rules) :
                
                        if self.VERBOSE : print (new_rule, new_rule.score, new_rule.localScore)
                
                        if new_rule.score.FP == 0 and new_rule.score.FN == 0 :
                           return new_rule   # we found a rule with maximal score => no better rule can be found
                        # elif new_rule.localScore <= old_rule.localScore) :
                        #     break
                        # Attempt to add rule to beam
                        elif not new_beam.push( new_rule , new_refs[i+1:] ) : 
                            break  # current ref does not have high enough score -> next ones will also fail
            
                # Use new beam in next iteration
                beam = new_beam
        
                # Write beam to log
                with Log('beam', _child=beam.toXML()) : pass
        except KeyboardInterrupt :
            with Log('INTERRUPT') : pass
            print ("INTERRUPT")
            
            pass    # TODO LOG INTERRUPT
        
        # Return head of beam.
        return beam.content[0][0]

    def update_refinements(self, rule, refine) :
        #with Log('ref', rule=rule, refine=refine) : pass
        if refine == None :
            return []
    
        #print ('R',refine)
    
        if self.language.DISTINCT_VARS :
            new_refine = list(rule.refine(update=False))
            refine = new_refine
        else :
            # Calculate new refinements in case a variable was added by the previous literal
            new_refine = list(rule.refine(update=True))
            # Add new refinements
            refine += new_refine
            
        with Log('new_refine', new=new_refine, all=refine) : pass
#        print ('NEW_REFINE:', new_refine, refine, rule)
        

        
        #print ('nR',new_refine)
        # Update scores for all literals in batch
        refine = sorted((rule + ref for ref in refine), reverse = True)
        
        # Reject / accept literals based on a local stopping criterion
        result = []
        for r in refine :
        
            prev_score = r.previous.score
            parent_score = r.parent.score
            new_score = r.score
                
            if new_score.TP <= prev_score.TP :
                # Rule doesn't cover any true positive examples => it's useless
                with Log('rejected', reason="TP", literal=r.literal, score=r.score, localScore=r.localScore ) : pass
            elif r.max_significance < self.SIGNIFICANCE :
                # Rule cannot reach required significance => it's useless
                with Log('rejected', reason="s", literal=r.literal, score=r.score, max_significance=r.max_significance ) : pass
            elif r.score_predict == r.parent.score_predict and not r._new_vars :
                # Predictions are indentical and new rule does not introduce a new variable
                with Log('rejected', reason="same", literal=r.literal, score=r.score ) : pass
            else :
                # Accept the extension and add it to the output
                with Log('accepted', literal=r.literal, score=r.score, localScore=r.localScore, max_significance=r.max_significance ) : pass
                result.append( r )
    
        return result

    # localScore = property(_calc_local)
    # significance = property(_calc_significance)
    # max_significance = property(_calc_max_significance)
    # globalScore = property(lambda self: self.score.accuracy())
    # probability = property(lambda self: self.score.max_x )

class PF1Score(object) :

    def __init__(self, correct, predict) :
        self.max_x = 1
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.TN = 0.0
        self.P = 0.0
        self.N = 0.0
            
        for p, ph in zip(correct,predict) :
            n = 1-p
            nh = 1-ph
            tp = min(p,ph)
            tn = min(n,nh)
            fp = n - tn
            fn = p - tp
        
            self.TP += tp
            self.TN += tn
            self.FP += fp
            self.FN += fn
            self.P += p
            self.N += n
        self.maxTP = self.TP                                    
        
    def m_estimate(self, m) :
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
    
    def accuracy(self) :
        M = self.P + self.N
        return (self.TP + self.TN ) / M
        
    def __str__(self) :
        return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )
    
    # def __eq__(self, other) :
    #     if other == None : return False
    #     return (self.TP, self.FP, self.TN, self.FN) == (other.TP, other.FP, other.TN, other.FN)

class ProbFOIL(LearningProblem) :
    
    def calculateScore(self, rule) :
        return PF1Score(rule.score_correct, rule.score_predict)

class ProbFOIL2(LearningProblem) :
    
    def calculateScore(self, rule) :
        if not rule.previous :
            return PF1Score(rule.score_correct, rule.score_predict)
        else :
            return PF2Score(rule.score_correct, rule.score_predict, rule.previous.score_predict)

class PF2Score(PF1Score) :

    def _calc_y(self, p,l,u) :
        if l == u :
            return 0
        else :
            v = (p-l) / (u-l)
            if v < 0 :
                return 0
            elif v > 1 :
                return 1
            else :
                return v
                
                
    def __init__(self, correct, predict, predict_prev) :
    
        self.MIN_RULE_PROB = 0.01
        self.M_ESTIMATE_M = 10
    
        values = sorted( (self._calc_y(p,l,u), p,l,u) for p,l,u in zip(correct, predict_prev, predict) )
      # with Log('calcscore') :
      #   with Log('values', _child=values) : pass
    
        P = 0.0
        N = 0.0
        ys = set([])
        for y, p, l, u in values :
            ys.add(y)
            P += p
            N += (1-p)
        m = self.M_ESTIMATE_M
        
        # TODO incremental computation
        def score(x) :
            r = [ ((u-l)*x + l, p) for y,p,l,u in values ]
            TP = sum( ri for ri, pi in r if ri < pi ) + sum( pi for ri, pi in r if ri >= pi )
            #with Log('fp', lst=r, x=x) : pass
            FP = sum( ri - pi for ri, pi in r if ri > pi )
            TN = sum( 1 - pi for ri, pi in r if ri <= pi ) + sum ( 1-ri for ri, pi in r if ri > pi )
            FN = sum( pi - ri for ri, pi in r if ri <= pi )
            return TP, FP, TN, FN
    
        max_s = None
        max_x = None
    
        # for x1 in range(0,101) :
        #     x = float(x1) / 100.0
        for x in sorted(ys) :
            TP, FP, TN, FN = score(x)
            s = self._m_estimate(m, TP, TN, FP, FN, P, N)
            # with Log('candidate', x=x, score=s, TP=TP, FP=FP, TN=TN, FN=FN) : pass
            if x >= self.MIN_RULE_PROB and ( max_s == None or s > max_s ) :
                max_s = s
                max_x = x
        if max_x == None :
            max_x = 1
            TP, FP, TN, FN = score(max_x)
            max_s = self._m_estimate(m, TP, TN, FP, FN, P, N)

        self.max_s = max_s
        self.max_x = max_x
        self.TP, self.FP, self.TN, self.FN = score(max_x)
        self.P = P
        self.N = N
    
        self.maxTP = score(1.0)[0]
    
        # with Log('best', x=max_x, score=max_s, TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN, m_est=self.m_estimate(m)) : pass         

    # def __eq__(self, other) :
    #     if other == None : return False
    #     return (self.TP, self.FP, self.TN, self.FN) == (other.TP, other.FP, other.TN, other.FN)
    

    def _m_estimate(self, m, TP, TN, FP, FN, P, N) :
        #if (TP == 0 and FP == 0 and m == 0) : m = 1
        return (TP + m * (P / (N + P))) / (TP + FP + m) 

    def __str__(self) :
        return '%.3g %.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN, self.max_x )

@total_ordering
class Rule(object) :
    
    ids = 0
    
    def __init__(self, parent) :
        self.parent = parent
        self.literal = None
        self.__score = None
        self.score_predict = None
        self.__identifier = Rule.ids
        Rule.ids += 1
        
    def __add__(self, literal) :
        """Adds the given literal to the body of the rule and returns the new rule."""
        result = RuleBody(literal, self)
        self.knowledge.enqueue(result)
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
        # TODO allow overwriting identifier => combine queries for multiple rules
        return self.__identifier # id(self)
    
    def _get_typed_variables(self) :
        return self.target.getTypedVariables(self.language)
        
    def _get_variables(self) :
        return set(map(lambda x : x[0], self.typed_variables))

    def _get_body_variables(self) :
        return set(map(lambda x : x[0], self._body_vars))
        
    def _get_examples(self) :
        return self.previous.examples    
                
    def _get_learning_problem(self) :
        return self.previous.learning_problem
                
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
    
    def __str__(self) :
        parts = self._str_parts()
        lines = []
        for p in parts :
            body = ', '.join(p)
            if body :
                lines.append( str(self.target) + ' :- ' + body + '.' )
            else :
                lines.append( str(self.target) + '.' )
        return '\t'.join(lines)
        
    def __len__(self) :
        if self.parent :
            return len(self.parent) + 1
        else :
            return 1
    
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
        
    def _get_score(self) :
        if self.__score == None :
            if self.score_predict == None :
                self.knowledge.process_queue()
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
        s = self.score
        M = s.P + s.N
        
        dTP = s.maxTP
        dM = M
        dP = s.P
        if self.previous :
            dTP -= self.previous.score.TP
           # dP -= self.previous.score.TP
        
        ms = 2 * dTP * ( -math.log( dP / dM ) )
        return ms
    
    # Calculate actual significance
    def _calc_significance(self) :
        s = self.score
        C = s.TP + s.FP
        if C == 0 : return 0
            
        M = s.P + s.N
        p_pos_c = s.TP / C
        p_neg_c = 1 - p_pos_c
        
        p_pos = s.P / M
        p_neg = s.N / M
        
        pos_log = math.log(p_pos_c/p_pos) if p_pos_c > 0 else 0
        neg_log = math.log(p_neg_c/p_neg) if p_neg_c > 0 else 0
        
        l = 2*C * (p_pos_c * pos_log  + p_neg_c * neg_log  )
        
        return l
    
    score_correct = property( lambda s: s._get_score_correct() )
    score_predict = property( lambda s : s._get_score_predict(), lambda s,v : s._set_score_predict(v) )
        
    score = property ( lambda s : s._get_score() )
    
    globalScore = property( lambda s : s._get_score().accuracy() )
    localScore = property( lambda s : s._get_score().m_estimate(5) )    # TODO set m value
    max_significance = property( _calc_max_significance ) # TODO get real value
    significance = property(_calc_significance)
    
    
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
        par[-1].append(str(self.literal))
        return par
                
class RuleHead(Rule) :
    """Rule with empty body."""
    
    def __init__(self, previous) :
        """Adds a new empty rule with the given head to the given FOIL rule set."""
        super(RuleHead,self).__init__(None)
        self.__previous = previous
        
        current_vars = set(self.target.variables)
        self._all_vars = current_vars
        self._new_vars = set([])
        self._body_vars = set([])
    
        self.score_predict = [1] * len(self.score_correct)
                
    def _get_target(self) :
        return self.previous.target
    
    def _get_previous(self) :
        return self.__previous
        
    def _str_parts(self) :
        par = self.previous._str_parts()
        par.append([] )
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
        
    def initialize(self) :
        # 1) Determine types of arguments of self.target
        #   => requires access to 'language'
        # argument_types = self.language.getArgumentTypes( self.target ) 
        
        # 2) Retrieve values for types of arguments 
        #   => requires access to 'language'

        # 3) Populate list of examples
        #   => carthesian product of values
        self.__examples = self.language.getArgumentValues( self.target )
        
        # 4) Populate scores => fact probabilities
        #   => requires access to 'facts'
        #   => scores are 1 - p where p is probability of the fact
        
        
        scores = []
        for example in self.examples :
            scores.append( self.knowledge.getFact( self.target.withArgs(example) ) )
        self.__score_correct = scores
        self.score_predict = [0] * len(scores)
    
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
        not_sign = '\+' if self.is_negated else ''
        r = not_sign + self.functor
        if self.arguments :
            return r + '(' + ','.join(map(str,self.arguments)) + ')'
        else :
            return r
            
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
        self.__varcount = 0
        
        self.DISTINCT_VARS = False
        
    def initialize(self, knowledge) :
        # Load type definitions
        for context, hasNext, prob in knowledge.engine.normal_engine.executeQuery('base(X).') :
            X = context['X']
            self.setArgumentTypes( Literal(X.functor, X.arguments) )
        
        # Load values
        for pred, arity in self.__types :
            types = self.__types[(pred,arity)]
            for i, tp in enumerate(types) :
                args = ['_'] * arity
                args[i] = 'X'
                args = ', '.join(args)
                query = '%s(%s).' % (pred,args)
                for context, hasNext, prob in knowledge.engine.normal_engine.executeQuery(query) :
                    self.addValue(tp, context['X'])
#                    self.__values[tp] |= set(map(str,context['VALUES']))
        
    def addValue(self, typename, value) :
        self.__values[typename].add(value)
        
    def setArgumentTypes(self, literal) :
        self.__types[ literal.key ] = literal.arguments
        
    def setArgumentModes(self, literal) :
        self.__modes[ literal.key ] = literal.arguments
        
    def getArgumentTypes(self, literal=None, key=None) :
        if literal : key = literal.key
        return self.__types[ key ]
            
    def getArgumentModes(self, literal=None, key=None) :
        if literal : key = literal.key
        return self.__modes[ key ] 
                
    def getTypeValues(self, typename) :
        return self.__values[ typename ]
        
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
            #print ('USE_VARS',use_vars)
        
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
        if arg_mode == '-' and (positive or self.DISTINCT_VARS) :
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

class PrologInterface(object) :
    
    def __init__(self) :
        self.engine = self._createPrologEngine()
        self.last_id = 0
        self.__queue = []
        self.__prob_cache = {}
                
    def _getRuleQuery(self, identifier) :
        return 'query_' + str(identifier)
        
    def _getRuleQueryAtom(self, rule) :
        functor = self._getRuleQuery(rule.identifier)
        args = rule.body_variables
        return Literal(functor, args)

    def _getRuleSetQueryAtom(self, rule, arguments=None) :
        functor = self._getRuleSetQuery(rule.identifier)
        if arguments == None :
            args = rule.target.arguments
        else :
            args = arguments
        return Literal(functor, args)
        
    def _getRuleSetQuery(self, identifier) :
        return 'query_set_' + str(identifier)
        
    def _prepare_rule(self, rule) :
        """Generates clauses for this rule and adds them to the Prolog engine."""
        clause_head = self._getRuleSetQueryAtom(rule)
        
        if rule.previous.identifier :
            prev_head = self._getRuleSetQueryAtom(rule.previous)
        else :
            prev_head = None
        
        
        clause_body = self._getRuleQueryAtom(rule)
                
        if prev_head :  # not the first rule
            clause1 = self._toPrologClause(clause_head, prev_head )
            self.engine.addClause( clause1 )
        
        if rule.parent :
            if rule.parent.literal :
                clauseB = self._toPrologClause(clause_body, self._getRuleQueryAtom(rule.parent), rule.literal )
            else :
                clauseB = self._toPrologClause(clause_body, rule.literal )
            self.engine.addClause( clauseB )
        else :
            clause_body = None
        clause2 = self._toPrologClause(clause_head, clause_body )
        self.engine.addClause( clause2 )
        
        # current_case :- parent_rule(A,B,C,D), new_literal(A,D).
        # new_query(A,B) :- previous_cases(A,B).
        # new_query(A,B) :- current_case(A,B,C,D).
                
    def _ground_rule(self, rule, examples) :
        """Execute grounding procedure for the given rule with the given examples."""
        
        functor = self._getRuleSetQuery(rule.identifier)
        nodes = []
        requires_problog = False
        for ex_id, ex in enumerate(examples) :
            if not rule.parent or rule.parent.score_predict[ex_id] != 0 :
                query = self._toProlog( Literal( functor, ex ) )
                node_id = self.engine.groundQuery(query)  # => should return node id
            else :
                node_id = None  # rule parent predicted FALSE for this rule
            nodes.append(node_id)
            if node_id :    # not None or 0
                requires_problog = True
        #print (rule, nodes, self.engine.ground_cache)
        rule.nodes = nodes
        rule.requires_problog = requires_problog
            
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        
        prep_time = time.time()
        self._prepare_rule(rule)
        prep_time = time.time() - prep_time
        
        ground_time = time.time()
        self._ground_rule(rule, rule.examples)
        ground_time = time.time() - ground_time
        
        #with Log('enqueue', rule=str(rule), prep_time=prep_time, ground_time=ground_time) : pass
        
        self._add_to_queue(rule)
        
    def getFact(self, fact) :
        return self.engine.clauses.get_fact( self._toProlog(fact) )
        
        
        
    def _add_to_queue(self, rule) :
        self.__queue.append(rule)
        
    def process_queue(self) :
        
     #print ('PROCESS', self.engine.ground_cache)
        
      with Log('problog', _timer=True) :
        
        # Build CNF
        queries = []
        facts = {}
        for rule in self.__queue :
            if rule.requires_problog :
                if rule.score_predict == None :
                    for ex_id, example in enumerate(rule.examples) :
                        if rule.nodes[ex_id] and not rule.nodes[ex_id] in self.__prob_cache : 
                            queries.append( self._toProlog(self._getRuleSetQueryAtom(rule, example)) )
        
        if queries :
            cnf, facts = self.engine.ground_cache.toCNF( queries )
            if len(cnf) > 1 :
                print ('COMPILING CNF ', cnf)
                with Log('compile', _timer=True) :
                    ddnnf = self._compile_cnf(cnf, facts)
        
        Literal = namedtuple('Literal', ['atom'])
        for rule in self.__queue :
            if rule.score_predict == None :
                evaluations = []
                for ex_id, example in enumerate(rule.examples) :
                    q_node_id = rule.nodes[ex_id]
                    
#                        q = self._toProlog(self._getRuleSetQueryAtom(rule, example))
                    #q_node_id = self.engine.ground_cache.byName(q)
                    if q_node_id == None :
                        p = 0
                    else :
                        is_neg = q_node_id < 0
                        q_node_id = abs(q_node_id)
                    
                        if q_node_id == 0 :
                            p = 1
                        elif q_node_id in facts :
                            p = facts[q_node_id]
                        elif q_node_id in self.__prob_cache :
                            p = self.__prob_cache[q_node_id]
                        else :
                            p = ddnnf[1][q_node_id]
                            self.__prob_cache[q_node_id] = p
                    
                        if is_neg : p = 1 - p
                    evaluations.append(p)
                rule.score_predict = evaluations
        
        self.__queue = []
                     
    def _compile_cnf(self, cnf, facts) :
        import subprocess
        
        cnf_file = 'probfoil_eval.cnf'
        nnf_file = os.path.splitext(cnf_file)[0] + '.nnf'
        with open(cnf_file,'w') as f :
            for line in cnf :
                 print(line,file=f)
        subprocess.check_output(["dsharp", "-Fnnf", nnf_file , "-disableAllLits", cnf_file])
        
        from compilation.compile import DDNNFFile
        from evaluation.evaluate import FileOptimizedEvaluator, DDNNF
        
        ddnnf = DDNNFFile(nnf_file, None)
        ddnnf.atoms = lambda : list(range(1,len(self.engine.ground_cache)+1))   # OMFG what a hack
        
        return FileOptimizedEvaluator()(knowledge=ddnnf, probabilities=facts, queries=None, env=None)
        
    ### Prolog implementation specific code below    
    def query(self, fact) :
        """Get probability for given fact."""
        
        num_sols = 0
        for context, hasMore, probability in self.engine.normal_engine.executeQuery(self._toProlog(fact)) :
            num_sols += 1
        
        if num_sols == 0 :
            return 0
        elif num_sols > 1 :
            raise Exception('Fact defined multiple times!')
        elif probability == None :
            return 1
        else :
            return probability.computeValue(None)   ## Prolog specific
            
    def _createPrologEngine(self) :
#        import prolog.core as prolog
        return GroundingEngine()
         
    def _toPrologClause(self, head, *body) :
        import prolog.core as prolog
        pl_head = self._toProlog(head)
        i = len(body) - 1
        pl_body = self._toProlog(body[i])
        while i > 0 :
            i -= 1
            pl_body = prolog.Conjunction( self._toProlog(body[i]), pl_body )
        return prolog.Clause( pl_head, pl_body )
                
    def _toProlog(self, term) :
        import prolog.core as prolog
        if term == None :
            return prolog.Function('true')
        elif isinstance(term, Literal) :
            args = list(map(self._toProlog, term.arguments))
            func = prolog.Function(term.functor, args)
            if term.is_negated :
                return prolog.Negation(func)
            else :
                return func
        elif isinstance(term, prolog.Constant) or isinstance(term, prolog.Function) :
            return term
        else :
            # return parser.parseToken(x)
            try :
                x = int(term)
                return prolog.Constant(x)
            except ValueError :
                if term[0] in '_ABCDEFGHIJKLMNOPQRSTUVWXYZ' :
                    return prolog.Variable(term)
                else :
                    return prolog.Function(term)

from prolog.core import PrologEngine
class GroundingEngine(PrologEngine) :
    
    def __init__(self, **args) :
        super(GroundingEngine, self).__init__(**args)
        
        self.now_grounding = set([])
        self.ground_cache = Index()
        
        self.mode = 'grounding'

        # self.loadFile = self._base_engine.loadFile
        # self.addClause = self._base_engine.addClause
        # self.evaluate = self._base_engine.evaluate
        # self.listing = self._base_engine.listing
        
    normal_engine = property( lambda s : super() )
    
        
    #    def addClause(self, clause) :
    #         return self._base_engine.addClause(clause)
    #     
    # def evaluate(self, call_atom, context) :
    #     return self._base_engine
        
    def _query_normal(self, call_atom, context) :
        #print ('QN', call_atom.ground(context, destroy_vars=True))
        return super()._query(call_atom, context)
       # return self._base_engine._query(call_atom, context)
        
    def _query(self, call_atom, context) :
        ground_atom = call_atom.ground(context, destroy_vars=True)
        if ground_atom in self.now_grounding :
            # We are already grounding this query (cycle).
            print ('WARNING: cycle detected!')
            context.pushFact( ground_atom )
            yield 0
        elif not ground_atom.variables and ground_atom in self.ground_cache.failed :
            # This (ground) query was grounded before and failed.
            return  # failed
        elif not ground_atom.variables and ground_atom in self.ground_cache :
            # This (ground) query was grounded before and succeeded.
            node_id=self.ground_cache.byName(ground_atom)
            context.pushFact( node_id )
            yield 0
        elif ground_atom.variables and ground_atom in self.ground_cache.groups :
            # This non-ground query was grounded before.
            group_results = self.ground_cache.groups[ground_atom]
            for result_atom, node_id in group_results :
                with context :
                    if call_atom.unify(result_atom, context, context) :
                        context.pushFact( node_id )
                        yield 1
        else :
            # This query was not grounded before, we should evaluate it.
            self.now_grounding.add( ground_atom )
            results = defaultdict(list)
            success = False
            for x in self._query_normal(call_atom, context) :
                success = True
                result_atom = call_atom.ground(context, destroy_vars=True)
                result_atom.probability = call_atom.probability
                if result_atom.variables : 
                    raise Exception('Expected ground atom: \'%s\'' % result_atom)
                facts = context.getFacts()
                facts = list(filter(lambda s : s != 0, facts))
                
                if facts and result_atom.probability :
                    # Probabilistic clause!
                    # ADD new probabilistic fact
                    
                    facts.append( self.ground_cache.add('fact', None ) )
                    results[result_atom].append( self.ground_cache.add('and',tuple(facts)) )
#                    raise NotImplementedError('No support for probabilistic clauses yet!')
                elif facts :
                    results[result_atom].append( self.ground_cache.add('and',tuple(facts)) )
                elif result_atom.probability :
                    node_id = self.ground_cache.add('fact', result_atom )
                    results[result_atom].append( node_id )
                else :
                    results[result_atom].append(0)
            
            new_results = []
            for result_atom in results :
                facts = results[result_atom]
                if 0 in facts :
                    self.ground_cache.add( 'or', [], name=result_atom)
                    results[result_atom] = []
                    new_results.append( (result_atom, 0 ))
                else :
                    new_results.append( (result_atom, self.ground_cache.add( 'or',tuple(facts), name=result_atom) ) )
            self.now_grounding.remove( ground_atom )
        
            for result_atom, idx in new_results :
                with context :
                    call_atom.unify(result_atom, context, context)
                    call_atom.probability = result_atom.probability
                    if results[result_atom] : 
                        context.pushFact(idx)
                    elif call_atom.probability != None :
                        node_id = self.ground_cache.add('fact', result_atom, name=result_atom )
                        #print ("USE:", result_atom, node_id)
                        context.pushFact( node_id, result_atom )
                    yield 0 # there was a result
            if ground_atom.variables :
                self.ground_cache.groups[ ground_atom ] = new_results
            if not ground_atom.variables and not success :
                self.ground_cache.addFailed( ground_atom )


    def _not(self, operand, context) :
        # TODO BUG in grounding negation?   => (in case of multiple evaluations)
        with context.derive() as current_context :
            operand.unify(operand, context, current_context)
            
            assert( not operand.ground(context).variables )
            
            uf = []
            for x in operand.evaluate(self, current_context) :
                if not operand.isProbabilistic and not context.getFacts() or 0 in context.getFacts() :
                    return
                else :                    
                    for f in context.getFacts() :
                        uf.append(-f)
        for f in uf :
            context.pushFact(f)
        yield 0

    def groundQuery(self, querystring) :
        q = querystring
            
        from prolog.memory import MemoryContext
        from prolog.core import DoCut
        with MemoryContext() as context :
            success = False
            try :
                context.defineVariables(q.variables)
                for result in q.evaluate(self, context) :
                    success = True
            except DoCut :
                context.log('CUT ENCOUNTERED')
                #yield context, True
            if success :
                return self.ground_cache.byName(querystring)
            else :
                return self.ground_cache.byName(querystring)
        
class Index(object) :
    
    def __init__(self) :
        self.__content = {}
        self.__index = []
        self.__names = {}
        self.__facts = {}
        
        self.__refcount = []
        self.__unresolved = defaultdict(list)
        self.hasCycle = False
        self.groups = {}
        self.failed = set([])
        
    def __len__(self) :
        return len(self.__index)
        
    def _update_refcount(self, node_id, data, name=None) :
        # if name != None :
        #     self.__refcount[node_id] += 1
        
        for x in data :
            try :
                x = int(x)
                self.__refcount[x-1] += 1
            except TypeError :
                self.hasCycle = True
                self.__unresolved[x].append(node_id)
                
        if name != None and name in self.__unresolved :
            nodes = self.__unresolved[name]
            self.__refcount[node_id-1] += len(nodes)
            for n in nodes :
                old_key = self[n]
                tp, dt = old_key
                new_tuple = []
                for d in dt :
                    if d == name :
                        new_tuple.append(node_id)
                    else :
                        new_tuple.append(d)
                new_key = tp, tuple(new_tuple)   
                del self.__content[ old_key ]
                self.__content[ new_key ] = n
                self.__index[ n-1 ] = new_key
            del self.__unresolved[ name ]
        
    def addFailed(self, name) :
        self.failed.add(name)
        
    def _addNode(self, key) :
        result = len(self.__index) + 1
        self.__content[key] = result    # Add the node
        self.__index.append(key)
        self.__refcount.append(0)
        return result
        
    def addFact(self, probability, identifier=None) :
        if identifier == None :
            identifier = 'FACT_' + str(len(self.__facts))
        
        key = ('fact',identifier)
        result = self.__content.get(key, None)
        if result == None : result = self._addNode(key)
        
        self.__facts[ result ] = probability
        return result
        
    def addNode(self, nodetype, content, name=None) :
        if not content :
            # Attempt to add an empty node
            result = 0  # => link it to 0
        else :
            content = tuple(sorted(set(content))) # Eliminate duplicates and fix order
            if len(content) == 1 :
                # Node with one child => don't add, just return child.
                result = content[0]
                self._update_refcount(result, [], name)
            else :
                # Node with multiple children
                key = (nodetype, content) # Create node key
                result = self.__content.get(key, None)  # Lookup node
                # Node does not exist yet
                if result == None : result = self._addNode(key)
                self._update_refcount(result, content, name)
        
        # Store name
        if name : self.__names[name] = result
        
        return result
        
    def add(self, nodetype, data, name=None) :
        if (nodetype == 'and' or nodetype =='or') and not data : 
            if name :
                self.__names[name] = 0
            #print ('DISCARD', nodetype, data, name)
            return    # Empty node (i.e. no probabilistic children) => skip
        
        if (nodetype == 'and' or nodetype == 'or') : data = tuple(sorted(set(data)))
        
        if nodetype == 'fact' and data == None :
            data = 'fact_' + str(len(self.__facts)) 
            raise NotImplementedError('Probabilistic clauses are not yet supported!')
        
        # Compact and skip nodes with only one child
        if (nodetype == 'and' or nodetype == 'or') and len(data) == 1  :
            result = data[0]            
            self._update_refcount(result, [], name)
        else :        
            key = (nodetype, data)
            result = self.__content.get(key, None)
            # Node does not exist yet
            if result == None :
                result = len(self.__index) + 1
                self.__content[key] = result
                self.__index.append(key)
                self.__refcount.append(0)
                
            if nodetype in ('and','or') : 
                self._update_refcount(result, data, name)
            elif nodetype == 'not' :
                assert(False)
                self._update_refcount(result, [data], name)
            else :  # fact
                self.__facts[ data ] = result
        # Register node name if one is given.
        if name :
            self.__names[name] = result
        #print ('STORE', nodetype, data, name, result)
        return result
    
    def getIndex(self, data) :
        return self.__content.get(data, None)
        
    def __getitem__(self, index) :
        assert(index > 0)
        return self.__index[index - 1]
        
    def byName(self, name) :
        return self.__names.get(name, None)
        
    def __contains__(self, data) :
        return data in self.__names
        
    def __iter__(self) :
        return iter(self.__index)
        
    def __str__(self) :
        s = ''
        for k,v in enumerate(self.__index) :
            s += ( str(k+1) + ': ' + str(v)) + '\n'
        
        names = dict([ (k, self.__names[k]) for k in self.__names if self.__names[k] != 0 ])
            
        s += 'NAMES: [' + str(len(self.__names)) + '] ' + str(names) + '\n'
        s += 'FACTS: ' + str(self.__facts)
        return s
        
    def _selectNodes(self, queries, node_selection) :
        for q in queries :
            node_id = self.byName(q)
            if node_id :
                self._selectNode(abs(node_id), node_selection)
        
    def _selectNode(self, node_id, node_selection) :
        assert(node_id != 0)
        if not node_selection[node_id-1] :
            node_selection[node_id-1] = True
            nodetype, content = self[node_id]
            
            if nodetype in ('and','or') :
                for subnode in content :
                    if subnode :
                        self._selectNode(abs(subnode), node_selection)
        
    def toCNF(self, queries=None) :
        if self.hasCycle :
            raise NotImplementedError('The dependency graph contains a cycle!')
        
        if queries != None :
            node_selection = [False] * len(self.__index)    # selection table
            self._selectNodes(queries, node_selection)
        else :
            node_selection = [True] * len(self.__index)    # selection table
        
        # TODO offset by one
        lines = []
        facts = {}
        for k, sel in enumerate( node_selection ) :
          if sel :
            k += 1
            v = self[k]
            nodetype, content = v
            if nodetype == 'fact' :
                if content.probability:
                    facts[k] = content.probability.computeValue(None)
                else :
                    facts[k] = 1 
            elif nodetype == 'and' :
                line = str(k) + ' ' + ' '.join( map( lambda x : str(-(x)), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (-k, x) )
                # lines.append('')
            elif nodetype == 'or' :
                line = str(-k) + ' ' + ' '.join( map( lambda x : str(x), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (k, -x) )
                # lines.append('')
            elif nodetype == 'not' :
                # TODO
                raise NotImplementedError("Not!")
                # line = str(k) + '-' + str(content)
                # lines.append(line)
            else :
                raise ValueError("Unknown node type!")
                
        atom_count = len(self.__index)
        clause_count = len(lines)
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts
        
    def stats(self) :
        return namedtuple('IndexStats', ('atom_count', 'name_count', 'fact_count' ) )(len(self), len(self.__names), len(self.__facts)) 


def test(filename, target, *modes) :
    
    # from rule import Literal, RuleHead, FalseRule
    
    target_pred, target_arity = target.split('/')
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    target_args = [] 
    for x in range(0, int(target_arity)) :
        target_args.append(letters[x])
    target = Literal(target_pred, target_args)
    
    modes = list(map(lambda x : Literal(*x.split('/')), modes))
      
    with open('log.xml', 'w') as Log.LOG_FILE : 
     with Log('log') :
        import prolog.parser as pp
        parser = pp.PrologParser()
#        target = Literal('grandmother','XY')
    
        p = PrologInterface()
    
        p.engine.loadFile(filename)
    
        l = Language()
        
        l.initialize(p)  # ==> read language specification + values from p
        
        
        # l.setArgumentTypes( Literal('grandmother', ('person', 'person') ) )
        # l.setArgumentTypes( Literal('parent', ('person', 'person') ) )
        # l.setArgumentTypes( Literal('female', ('person',) ) )
        # l.setArgumentTypes( Literal('father', ('person', 'person') ) )    
        # l.setArgumentTypes( Literal('mother', ('person', 'person') ) )
        # l.setArgumentModes( Literal('father', ('+','-') ) )
        # l.setArgumentModes( Literal('mother', ('+','-') ) )
        
        for mode in modes :
            l.setArgumentModes( mode )
#        l.setArgumentModes( Literal('female', ('+') ) )
        
        
        # for v in  [ 'alice', 'an', 'esther', 'katleen', 'laura', 'lieve', 'lucy', 'rose', 'soetkin', 'yvonne', 
        #     'bart', 'etienne', 'leon', 'luc', 'pieter', 'prudent', 'rene', 'stijn', 'willem' ] :
        #     l.addValue('person',v)
    
        lp = ProbFOIL2(l, p)
        
        with Log('initialize', _timer=True) :
            r0 = RootRule(target, lp)
            r0.initialize()
        
        print(r0.refine())
        
        print (r0.score_correct)
        
        try :
            result = lp.learn(r0)
        except :
            with Log('grounding_stats', **vars(p.engine.ground_cache.stats())) : pass
            with Log('error') : pass
#            p.engine.listing()
            with open('probfoil.pl','w') as pl_out :
                print (p.engine.listing(), file=pl_out)

            raise Exception('ERROR')

        with Log('grounding_stats', **vars(p.engine.ground_cache.stats())) : pass
        print('RULE LEARNED:',result)
        
        with open('probfoil.pl','w') as pl_out :
            print (p.engine.listing(), file=pl_out)

        # print(p.engine.ground_cache)

if __name__ == '__main__' :
    test(*sys.argv[1:])    