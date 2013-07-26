#! /usr/bin/env python3

from __future__ import print_function               # for compatibility with Python 2

from collections import defaultdict, namedtuple
from itertools import product
from functools import total_ordering
from util import Log, Timer
import os, sys

##############################################################################
###                      PARSE COMMAND LINE ARGUMENTS                      ###
##############################################################################

from argparse import ArgumentParser

import math
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

parser = ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--beamsize', type=int, default=5, dest='BEAM_SIZE', help='size of search beam')
parser.add_argument('-m', type=int, dest='M_ESTIMATE_M',  default=10, help='value of m for m-estimate')
parser.add_argument('--distinct_vars', action='store_true', dest='DISTINCT_VARS', help='enable distinct variables (EXPERIMENTAL)')
parser.add_argument('--pvalue', type=float, dest='SIGNIFICANCE_P', default=0.99, help='minimal rule significance (p-value)')
parser.add_argument('-v', action='count', dest='VERBOSE', help='increase verbosity level')
parser.add_argument('--probfoil1', action='store_false', dest='PROBFOIL2', help='use traditional ProbFOIL scoring')
parser.add_argument('--min_rule_prob', type=float, dest='MIN_RULE_PROB', default=0.01, help='minimal probability of rule in Prob2FOIL')
parser.add_argument('--equiv_check', action='store_true', dest='EQUIV_CHECK', help='enable elimination of equivalent results in beam')

SETTINGS = parser.parse_args()
SETTINGS.SIGNIFICANCE= calc_significance(SETTINGS.SIGNIFICANCE_P)

##############################################################################
###                           LEARNING ALGORITHM                           ###
##############################################################################

def learn(H) :
    """Core FOIL learning algorithm.
    
    H - initial hypothesis (Rule (e.g. FalseRule))
    """
    
    # Find clauses as long as stopping criterion is not met or until maximal score (1.0) is reached.
    while H.globalScore < 1.0 :   # this test is not required for correctness (alternative: while True)
       
      with Log('learn_rule', _timer=True):
            
        # Find best clause that refines this hypothesis
        new_H = best_clause( H )
        
        if SETTINGS.VERBOSE : print ('RULE FOUND:', new_H, new_H.globalScore)
        
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

def best_clause( current_rule ) :
    """Find the best clause for this hypothesis."""
    
    # We use beam search; initialize beam
    beam = Beam(SETTINGS.BEAM_SIZE)    
    
    # Create the new rule ( target <- )
    init_rule = RuleHead(current_rule.target, previous=current_rule)
    
    # Calculate initial set of refinements
    refinements = list(init_rule.refine())
    
    # Add clause to beam (with empty score)
    beam.push( init_rule, refinements )
    
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
            new_rules = update_refinements(old_rule, refs)
            
            # Extract refinement literals
            new_refs = [ r.literal for r in new_rules ]
            
            # Add rules to new beam (new_refs are ordered by score, descending)
            for i, new_rule in enumerate(new_rules) :
                
                if SETTINGS.VERBOSE : print (new_rule, new_rule.score)
                
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
    
    # Return head of beam.
    return beam.content[0][0]

def update_refinements( rule, refine) :
    #with Log('ref', rule=rule, refine=refine) : pass
    if refine == None :
        return []
    
    if SETTINGS.DISTINCT_VARS :
        new_refine = list(rule.refine(update=False))
        refine = new_refine
    else :
        # Calculate new refinements in case a variable was added by the previous literal
        new_refine = list(rule.refine(update=True))
        # Add new refinements
        refine += new_refine
        
    
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
        elif r.max_significance < SETTINGS.SIGNIFICANCE :
            # Rule cannot reach required significance => it's useless
            with Log('rejected', reason="s", literal=r.literal, score=r.score, max_significance=r.max_significance ) : pass
        else :
            # Accept the extension and add it to the output
            with Log('accepted', literal=r.literal, score=r.score, localScore=r.localScore ) : pass
            result.append( r )
    
    return result

##############################################################################
###                          RULES AND EXTENSIONS                          ###
##############################################################################

@total_ordering
class Rule(object) :
    
    # literal   (Literal)               => current literal
    # parent    (Rule)                  => parent rule (i.e. without current literal)
    # previous  (Rule)                  => previous rule in the ruleset
    # kb        (KnowledgeBase)         => knowledge base
    # language  (Language)              => language specifier
    # variables {(name, type)}          => set of variables in the rule
    # examples
    
    def __init__(self, literal, parent) :
        self.parent = parent
        self.literal = literal
        self._score = None
    
        if parent :
            self.previous = parent.previous
            self.kb = parent.kb
            self.examples = parent.examples
            self.target = parent.target
            old_vars = parent.variables
        else :
            old_vars = set([])
        
        self._evaluation_cache = None
        
        current_vars = set(literal.get_vars(self.kb))
        self._all_vars = old_vars | current_vars
        self._new_vars = current_vars - old_vars
    
    # Calculate the local score
    def _calc_local(self) :
        m = SETTINGS.M_ESTIMATE_M 
        m_estimate_c = self.score.m_estimate(m)
        return m_estimate_c #- m_estimate_p
        
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
        
    # Calculate the Prob2FOIL score
    def _get_score2(self) :
        if self._score == None :
            predict = [ x[0] for x in self.evaluate() ]
            correct = [ x[0] for x in self.examples]
            if  self.previous :
                predict_prev = [ x[0] * self.previous.probability for x in self.previous.evaluate() ]
            else :
                predict_prev = [ 0 for x in self.examples ]
            self._score = Score2(correct, predict, predict_prev)
        return self._score
        
    # Calculate the ProbFOIL score
    def _get_score(self) :
        if self._score == None :
            predict = [ x[0] for x in self.evaluate() ]
            correct = [ x[0] for x in self.examples]
            self._score = Score(correct, predict)
        return self._score
    
    variables = property(lambda self : self._all_vars)
    language = property(lambda self : Language(self.kb) )
    
    if SETTINGS.PROBFOIL2 :
        score = property(_get_score2)
    else :
        score = property(_get_score)
    localScore = property(_calc_local)
    significance = property(_calc_significance)
    max_significance = property(_calc_max_significance)
    globalScore = property(lambda self: self.score.accuracy())
    probability = property(lambda self: self.score.max_x )
    
    def _toString(self) :
        parent_string, parent_sep = self.parent._toString()
        return parent_string + parent_sep + str(self.literal), ', '
        
    def __str__(self) :
        return self._toString()[0] + '.'
        
    def __add__(self, literal) :
        return Rule(literal, self)
        
    # Evaluate rule
    def evaluate(self) :
        if self._evaluation_cache == None :
            self._evaluation_cache = [None] * len(self.examples)
            parent_evaluation = self.parent.evaluate()
            
            for ex, parent_eval in enumerate(parent_evaluation) :
                current_cache = []
                
                restricts = False
                is_probabilistic = False
                score = 0.0
                
                if self.previous != None :
                    score = self.previous.evaluate()[ex][0] * self.previous.probability
                                
                parent_score, parent_matches, parent_used = parent_eval 
                used_facts = defaultdict(set)
                used_facts.update(parent_used)
                for parent_subst, query in parent_matches :
                    local_restricts = True
                    for subst, lit in self.kb.query([self.literal], parent_subst, used_facts, distinct=SETTINGS.DISTINCT_VARS) :
                        if query or lit :
                            if lit and -lit[0] in query :
                                pass
                            else :
                                local_restricts = False
                                if lit and lit[0] in query :
                                    lit = []
                                    break
                                else :
                                    is_probabilistic = True
                                    score = None
                                current_cache.append( (subst, query | set(lit) ) )
                        else :
                            local_restricts = False
                            score = 1.0
                            current_cache.append( (subst, set([]) ) )
                        
                    restricts |= local_restricts
                if not restricts and not is_probabilistic :
                    score = parent_score
                    
                self._evaluation_cache[ex] = (score, current_cache, used_facts)
        
            self._update_scores(self._evaluation_cache)
        
        return self._evaluation_cache
        
    # Generate refinement literals 
    def refine(self, update=False) :
        # generate refined clauses
        if update :
            # Only generate extensions that use a variable introduced by the last literal
            use_vars = self._new_vars
            if not use_vars : return [] # no variables available
        else :
            use_vars = None
        return [ literal for literal in self.language.refinements( self.variables, use_vars ) ]
    
    # Score evaluation using ProbLog
    def _update_scores(self, cache) : 
      with Log('scoreupdate', rule=self)   :
        queries = QueryPack(self.kb)
        
        for ex, cache_ex in enumerate(cache) :
            score, clauses, used_facts = cache_ex
            if score == None :
                # we need to call problog
                query = Query(ex)
                query.update_used_facts(used_facts)
                
                previous_rules = [self.previous]
                while previous_rules[-1] != None :
                    previous_rules.append( previous_rules[-1].previous )
                previous_rules = previous_rules[:-1]
                for prev_rule_i, prev_rule in enumerate(previous_rules) :
                    _a, prev_clauses, prev_used_facts = prev_rule.evaluate()[ex]
                    if prev_rule.probability < 1 and prev_rule.probability > 0 :
                        prev_used_facts[ (None, prev_rule_i) ].add(prev_rule.probability)
                    query.update_used_facts(prev_used_facts)
                    for _v, clause in prev_clauses :
                        if 0 < prev_rule.probability < 1 :
                            clause.add( Literal('prob_' + str(prev_rule_i), []))
                        query += clause
                for _v, clause in clauses :
                    query += clause
                
                queries += query
            
        scores = queries.execute()
        
        for ex in scores :
            score, current, used_facts = cache[ex]
            cache[ex] = scores[ex], current, used_facts
            
    def __len__(self) :
        if self.parent :
            return len(self.parent) + 1
        else :
            return 1
    
    # Count number of negative literals
    def count_negated(self) :
        if self.parent :
            return self.parent.count_negated() + self.literal.is_negated
        else :
            return self.literal.is_negated
            
    def __lt__(self, other) :
        return (self.localScore, -len(self), -self.count_negated(), str(self)) < (other.localScore, -len(other), -other.count_negated(), str(other))             
        
    def __eq__(self, other) :
        return str(self) == str(other)
        
    # Return a string representation of the entire ruleset
    def strAll(self) :
        r = self.previous.strAll() 
        if self.probability < 0.9999 :
            r += str(self) + ' %% (p=%.4g)\n' % self.probability
        else :
            r += str(self) + '\n' 
        return r

# A rule with an empty body.
class RuleHead(Rule) :
    
    def __init__(self, literal, previous) :
        self.previous = previous
        self.examples = previous.examples
        self.target = previous.target
        self.kb = previous.kb
        super(RuleHead, self).__init__(literal, None)
        
    def _toString(self) :
        return str(self.literal), ' :- '
        
    def refine(self, update=False) :
        if update :
            return []
        else :
            return [ literal for literal in self.language.refinements( self.variables, None ) ]
        
    
    def evaluate(self) :
        if self._evaluation_cache == None :
            self._evaluation_cache = [None] * len(self.examples)
            for i, ex_p in enumerate(self.examples) :
                p, ex = ex_p
                subst = self.literal.unify( ex )
                self._evaluation_cache[i] =  (1.0, [( subst, set([]) )], defaultdict(set) )
        return self._evaluation_cache
        
# A rule with a false body (base rule in a ruleset)
class FalseRule(Rule) :
    
    def __init__(self, kb, target) :
        self.kb = kb
        self.previous = None
        self.target = target
        super(FalseRule, self).__init__(target, None)
        self.examples = self._initExamples()
        
    def _toString(self) :
        return str(self.literal), ' :- fail.'
        
    def __str__(self) :
        return ''
        
    def _initExamples(self) :
        examples = list(product(*[ self.kb.types[t] for t in self.language.predicate_types(self.literal.identifier) ]))
        result = []
        for example in examples :
            p = self.kb.find_fact( Literal( self.literal.functor, example ) )
            result.append( (p, example) )
        return result        
        
    def evaluate(self) :
        if self._evaluation_cache == None :
            self._evaluation_cache = [None] * len(self.examples)
            for i, ex_p in enumerate(self.examples) :
                p, ex = ex_p
                self._evaluation_cache[i] =  (0.0, [], {} )
        return self._evaluation_cache
        
    def strAll(self) :
        return ''

# Language definition, defines refinement operator
class Language(object) :
    
    def __init__(self, kb) :
        self.kb = kb
        self.predicates = list(kb.modes)
        
    def newVar(self) :
        return self.kb.newVar()
    
    def predicate_types(self, literal) :
        return self.kb.argtypes(*literal)
        
    def predicate_modes(self, literal) :
        return self.kb.modes[literal]
        
    def refinements(self, variables, use_vars) :
        existing_variables = defaultdict(list)
        for varname, vartype in variables :
            existing_variables[vartype].append( varname ) 
        
        if use_vars != None :
            use_vars = set( [ varname for varname, vartype in use_vars ] )
        
        for pred_id in self.predicates :
            pred_name = pred_id[0]
            arg_info = list(zip(self.predicate_types(pred_id), self.predicate_modes(pred_id)))
            for args in self._build_refine(existing_variables, True, arg_info, use_vars) :
                yield Literal(pred_name, args)
            for args in self._build_refine(existing_variables, False, arg_info, use_vars) :
                yield Literal(pred_name, args, False)
    
    def _build_refine_one(self, existing_variables, positive, arg_type, arg_mode) :
        if arg_mode in ['+','-'] :
            for var in existing_variables[arg_type] :
                yield var
        if arg_mode == '-' and (positive or SETTINGS.DISTINCT_VARS) :
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


# ProbFOIL/mFOIL score calculation
class Score(object) :
    
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
        return (self.TP + self.TN ) / (self.TP + self.TN + self.FP + self.FN)
            
    def __str__(self) :
        return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )
     
# Prob2FOIL score calculation   
class Score2(Score) :
    
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
        m = SETTINGS.M_ESTIMATE_M

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
            if x >= SETTINGS.MIN_RULE_PROB and ( max_s == None or s > max_s ) :
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
        
    
    def _m_estimate(self, m, TP, TN, FP, FN, P, N) :
        #if (TP == 0 and FP == 0 and m == 0) : m = 1
        return (TP + m * (P / (N + P))) / (TP + FP + m) 
    


##############################################################################
###                           PROBLOG EVALUATION                           ###
##############################################################################

# Query pack for ProbLog queries
class QueryPack(object) :
     
    # This field is shared by all objects for this class!
    query_cache = {}
     
    def __init__(self, kb) :
        self.__queries = {}
        self.__kb = kb
        self.used_facts = defaultdict(set)
        
    def update_used_facts(self, update) :
        for pred in update :
            self.used_facts[pred] |= update[pred]
        
    def addQuery(self, query) :
        key = query.key
        val = query.value
        
        if not key in self.__queries :
            self.__queries[key] = (query, [])
        
        self.__queries[key][1].append(val)
        self.update_used_facts(query.used_facts)
        
    def __iadd__(self, query) :
        """Add a query."""
        self.addQuery(query)
        return self
    
    def __len__(self) :
        return len(self.__queries)
        
    def __iter__(self) :
        return iter(self.__queries)
        
    def __getitem__(self, key) :
        return self.__queries[key][1]
        
    def __delitem__(self, key) :
        del self.__queries[key]
        
    def facts(self) :
        result = set([])
        for queries in self.__queries.values() :
            for _a, _b, facts in queries :
                result |= facts
        return result
        
    def split(self) :
        for key in self :
            qp = QueryPack(self.__kb)
            qp.__queries[key] = self[key]
            qp.used_facts = self.used_facts
            yield qp
     
     
    @classmethod
    def lit_to_problog_lit(cls, lit) :
        import core
        
        if lit.is_negated :
            return core.Literal( str(-lit), False )
        else :
            return core.Literal( str(lit), True )
        
     
    def _to_dependency_graph(self) :
        import core
        
        GN = namedtuple('GraphNode', ['type', 'content'])
        
        depGraph = {}
        
        and_nodes = {}
        
        and_index = 0
        for qi, key in enumerate(self) :
            query, _ids = self.__queries[key]
            
            or_branches = [] 
            for clause in query.clauses :
                if len(clause) == 1 :
                    or_branches.append( QueryPack.lit_to_problog_lit(list(clause)[0]) )
                else :
                    clause = tuple(clause)
                    if clause in and_nodes :
                        and_name = and_nodes[clause]
                    else :
                        and_name = 'and' + str(len(and_nodes))
                        depGraph[and_name] = GN('and_node', list(map(QueryPack.lit_to_problog_lit, clause )))
                        and_nodes[clause] = and_name
                    or_branches.append( QueryPack.lit_to_problog_lit(Literal(and_name,[])  ) )
            
            depGraph['pfq_' + str(qi) ] = GN('or_node', or_branches)            
        return depGraph

    def get_probs(self) :
     
        facts = self.used_facts
     
        result = {}
        for identifier in facts :
            name, arity = identifier
            if name == None :
                values = []
                for p in facts[identifier] :
                    result[ 'prob_' + str(arity) ] = p
            else :
                _index, values, _types, probs = self.__kb.predicates[identifier]
    
                for i in facts[identifier] :
                    args = ','.join(values[i])
                    prob = probs[i]
         
                    lit =  '%s(%s)' % (name, args)
                    result[ lit ] = prob
                    self.query_cache[lit] = prob
                    self.query_cache['\+' + lit] = 1-prob
        return result
        
    def toProbLog(self) :
        r = ''
        probs = self.get_probs()
        
        for at in probs :
            r += '%s::%s.\n' % (at, probs[at]) 
         
        for qi, key in enumerate(self) :
            query, _ids = self.__queries[key]
            
            query_head = 'pfq_' + str(qi)
            for clause in query.clauses :
                
                r += '%s :- %s.\n' % (query_head, ','.join(map(str,clause)))
                
            r += 'query(%s).\n' % query_head
        return r
        
     
    def execute( self ) :
        scores = {}
        
        if not len(self) :
            return scores
        
        probabilities = self.get_probs()     # get from used facts
        
        # Lookup queries in cache
        cnt_cache = 0
        skipped = []
        for key in self :
            p = self.query_cache.get(key, None)
            if p != None :
                cnt_cache += 1
                for ex_i, _a in self[key] :
                    scores[ex_i] = p
                skipped.append(key)
        for key in skipped :
            del self[key]
                
        dependency_graph = self._to_dependency_graph() 
        create_dependency_graph = lambda *s : dependency_graph
        ground_program = None
        queries = [ QueryPack.lit_to_problog_lit(Literal('pfq_%s' % i, [])) for i, _q in enumerate(self) ]
        evidence = []
        
        
        q_cnt = len(self)
        with Log('problog', nr_of_queries=q_cnt+cnt_cache, cache_hits=cnt_cache, _timer=(q_cnt > 0)) as l :
        
            
            if q_cnt == 0 :
                return scores
        
            # with Log('program', _child=self.toProbLog()) : pass
            #     
            # with Log('dpgraph', _child=dependency_graph, size=len(dependency_graph)) :
            #     pass
            #     
            # with Log('probs', _child=probabilities) :
            #     pass
            #     
            # with Log('queries', _child=queries) :
            #     pass
                
            import problog
            engine_base = problog.ProbLogEngine.create([])
                
            GroundProgram = namedtuple('GroundProgram', ('ground_program', 'evidence', 'probabilities', 'queries'))
            
            engine = problog.ProbLogEngine(
                lambda *s : GroundProgram(ground_program, evidence, probabilities, queries),
                engine_base.converter,
                engine_base.compiler,
                engine_base.evaluator
            )
            
            engine.converter._create_dependency_graph = create_dependency_graph
            
            logger = problog.Logger(0)
            dir_out = '/tmp/pf/'
            with problog.WorkEnv(dir_out,logger, persistent=problog.WorkEnv.ALWAYS_KEEP) as env :
                probs = engine.execute('a', env) # TODO call problog
        
        for i, b in enumerate(self) :
            q = 'pfq_%s' % i
            p = probs[q]            
            for e, _x in self[b] :
                scores[e] = p
                self.query_cache[b] = p
                
        return scores
         
# Query for ProbLog
class Query(object) :
     
    def __init__(self, example_id, parent=None) :
        self.example_id = example_id
        self.used_facts = defaultdict(set)
        if parent :
            self.clauses = parent.clauses[:]
        else :
            self.clauses = []
        
    def update_used_facts(self, update) :
        for pred in update :
            self.used_facts[pred] |= update[pred]
        
        
    key = property( lambda s : str(s) )
    value = property( lambda s : (s.example_id, s.facts() ) )
        
    def __iadd__(self, clause_body) :
        self.addClause(clause_body)
        return self
        
    def addClause(self, new_clause) :
        if not new_clause : return
        new_clause = set(new_clause)
        i = 0
        while i < len(self.clauses) :
            existing_clause = self.clauses[i]
            if self._test_subsume(existing_clause, new_clause) :
                # existing clause subsumes new clause => discard new clause
                return
            elif self._test_subsume(new_clause, existing_clause) :
                # new clause subsumes existing clause => discard existing clause
                self.clauses.pop(i)
            else :
                i += 1
        self.clauses.append( new_clause )
        self.clauses = sorted(self.clauses)
        
    def _test_subsume(self, clause1, clause2) :
        return clause1 <= clause2
        
    def __str__(self) :
        return ';'.join( ','.join( map(str, clause) ) for clause in self.clauses )
        
    def facts(self) :
        result = set([])
        for clause in self.clauses :
            result |= clause
        return result
        
##############################################################################
###                              PROLOG CORE                               ###
##############################################################################

class KnowledgeBase(object) :
    
    def __init__(self, idtypes=[]) :
        self.predicates = {}
        self.idtypes = idtypes
        self.__examples = None
        self.types = defaultdict(set)
        self.modes = {}
        self.learn = []
        self.newvar_count = 0
        self.probabilistic = set([])
        self.prolog_file = None
        self.query_cache = {}
        
    def newVar(self) :
        self.newvar_count += 1
        return 'X_' + str(self.newvar_count)
            
    def register_predicate(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(set) for i in range(0,arity) ] 
            self.predicates[identifier] = (index, [], args, [])
            
    def is_probabilistic(self, literal) :
        return literal.identifier in self.probabilistic
            
    def add_mode(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        self.modes[identifier] = args
        
    def add_learn(self, name, args) :
        self.learn.append( (name, args) )
        
    def add_fact(self, name, args, p=1.0) :
        arity = len(args)
       # self.register_predicate(name, arity)
        identifier = (name, arity)
        index, values, types, probs = self.predicates[identifier]
        
        if p < 1.0 :
            self.probabilistic.add(identifier)
        
        arg_index = len(values)
        values.append( args )
        probs.append( p )
        
        for i, arg in enumerate(args) :
            self.types[types[i]].add(arg)
            index[i][arg].add(arg_index)
        
    def argtypes_l(self, literal) :
        return self.predicates[literal.identifier][2]
        
    def argtypes(self, name, arity) :
        identifier = (name, arity)
        return self.predicates[identifier][2]
        
    def ground_fact(self, literal, used_facts=None, restricts=[]) :
        
        index, values, types, probs = self.predicates[ literal.identifier ]
        
        # Initial result set = all literals for this predicate
        result = set(range(0, len(values))) 
        
        # Restrict set for each argument
        for i,arg in enumerate(literal.args) :
            if not is_var(arg) :
                result &= index[i][arg]
            elif restricts and literal.is_negated :
                # value of argument should not be one of restricts
                for r in restricts :
                    result -= index[i][r]
            if not result : break 
        
        result_maybe = set( i for i in result if 0 < probs[i] < 1 )
        result_exact = result - result_maybe
        
        if literal.is_negated :
            if result_exact : 
                # There are facts that are definitely true
                return []   # Query fails
            else :
                # Query might succeed
                if used_facts != None and result_maybe :
                    # Add maybe facts to used_facts
                    used_facts[ literal.identifier ] |= result_maybe
                if result_maybe :
                    is_det = False
                else :
                    is_det = True
                return [ (is_det, literal.args) ]
        else :
            if used_facts != None and result_maybe :
                # Add maybe facts to used_facts
                used_facts[ literal.identifier ] |= result_maybe
            return [ (probs[i] == 1, values[i]) for i in result_maybe | result_exact ]
        
    def find_fact(self, literal) :
        
        index, values, types, probs = self.predicates[literal.identifier]
        
        result = set(range(0, len(values))) 
        for i,arg in enumerate(literal.args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break        
        
        probability = sum( probs[i] for i in result )        
        if literal.is_negated :
            return 1.0 - probability
        else :
            return probability
            
    def __str__(self) :
        s = ''
        for name, arity in self.predicates :
            for args in self.predicates[(name,arity)][1] :
                s += '%s(%s)' % (name, ','.join(args)) + '.\n'
        return s
          
    def reset_examples(self) :          
        self.__examples = None
        
    def _get_examples(self) :
        if self.__examples == None :
            self.__examples = list(product(*[ self.types[t] for t in self.idtypes ]))
        return self.__examples
    
    examples = property(_get_examples)
        
    def __getitem__(self, index) :
        return self.examples[index]
        
    def __len__(self) :
        return len(self.examples)
        
    def __iter__(self) :
        return iter(range(0,len(self.examples)))
    
    def query(self, literals, substitution, facts_used=None, distinct=False) :
        if not literals :  # reached end of query
            yield substitution, []
        else :
            head, tail = literals[0], literals[1:]
            head_ground = head.assign(substitution)     # assign known variables
            
            if distinct :
                distincts = substitution.values()
            else :
                distincts = []
            
            for is_det, match in self.ground_fact(head_ground, facts_used, distincts) :
                # find match for head
                new_substitution = dict(substitution)
                new_substitution.update( head_ground.unify( match ) )
                
                if distinct and not distinct_values(new_substitution) :
                    continue
                    
                if is_det :
                    my_lit = []
                else :
                    my_lit = [ head_ground.assign(new_substitution) ]
                for sol, sub_lit in self.query(tail, new_substitution, facts_used) :
                    if not my_lit or my_lit[0] in sub_lit :
                        next_lit = sub_lit
                    else :
                        next_lit = my_lit + sub_lit
                    
                    yield sol, next_lit
    
    def facts_string(self, facts) :
        result = ''
        for identifier in facts :
            name, arity = identifier
            index, values, types, probs = self.predicates[identifier]
            
            for i in facts[identifier] :
                args = ','.join(values[i])
                prob = probs[i]
                if prob == 1 :
                    result += '%s(%s).\n' % (name, args)
                else :
                    result += '%s::%s(%s).\n' % (prob, name, args)
        return result
                    
    def toPrologFile(self) :
        if self.prolog_file == None :
            self.prolog_file = '/tmp/pf.pl'
            with open(self.prolog_file, 'w') as f :
                for pred in self.predicates :
                    index, values, types, probs = self.predicates[pred]
                    for args in values :
                        print('%s(%s).' % (pred[0], ', '.join(args) ), file=f)
        return self.prolog_file
        

def is_var(arg) :
    return arg == None or arg[0] == '_' or (arg[0] >= 'A' and arg[0] <= 'Z')

def distinct_values(subst) :
    return len(subst.values()) == len(set(subst.values()))

class Literal(object) :
    
    def __init__(self, functor, args, sign=True) :
        self.functor = functor
        self.args = list(args)
        self.sign = sign
    
    identifier = property(lambda s : (s.functor, len(s.args) ) )
    is_negated = property(lambda s : not s.sign)
                
    def __str__(self) :  
        return repr(self)
        
    def __repr__(self) :
        sign = '\+' if not self.sign else ''
        args = '(%s)' % ','.join(self.args) if self.args else ''
        return '%s%s%s' % (sign, self.functor, args)
        
    def __neg__(self) :
        return Literal(self.functor, self.args, not self.sign)
        
    def __hash__(self) :
        return hash(str(self))
        
    def __eq__(self, other) :
        return self.functor == other.functor and self.args == other.args and self.sign == other.sign
                
    def __lt__(self, other) :
        return (self.sign, len(self.args), self.functor, self.args) < (other.sign, len(other.args), other.functor, other.args)
                        
    def unify(self, ground_args) :
        """Unifies the arguments of this literal with the given list of literals and returns the substitution.
            Only does one-way unification, only the first literal should have variables.
            
            Returns the substitution as a dictionary { variable name : value }.
        """
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if is_var(ARG) :
                if not ARG == '_' :
                    result[ARG] = arg
            elif is_var(arg) :
                raise ValueError("Unexpected variable in second literal : '%s'!" % (ARG))
            elif ARG == arg :
                pass    # default case
            else :
                raise ValueError("Literals cannot be unified: '%s' '%s!'" % (arg, ARG))
        return result  
        
    def assign(self, subst) :
        """Creates a new literal where the variables are assigned according to the given substitution."""
        return Literal(self.functor, [ subst.get(arg, arg) for arg in self.args ], self.sign)
        
    def get_vars(self, kb) :
        return [ (arg, typ) for arg,typ in zip(self.args, kb.argtypes_l(self)) if is_var(arg) and not arg == '_' ]
        
                
    @classmethod
    def parse(cls, string) :
        """Parse a literal from a string."""
        
        # TODO does not support literals of arity 0.
        
        regex = re.compile('\s*(?P<name>[^\(]+)\((?P<args>[^\)]+)\)[,.]?\s*')
        result = []
        for name, args in regex.findall(string) :
            if name.startswith('\+') :
                name = strip_negation(name)
                sign = False
            else :
                sign = True
            result.append( Literal( name , map(lambda s : s.strip(), args.split(',')), sign) )
        return result

##############################################################################
###                               UTILITIES                                ###
##############################################################################

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
        
        if SETTINGS.EQUIV_CHECK and len(self.content) > 1 :
            r1, rf1 = self.content[p]
            r2, rf2 = self.content[p+1]
            
            r1scores = list(zip(*r1.evaluate()))[0]
            r2scores = list(zip(*r2.evaluate()))[0]
            
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

##############################################################################
###                            SCRIPT EXECUTION                            ###
##############################################################################

def read_file(filename) :
    if filename.endswith('.arff') :
        return read_arff(filename)
    else :
        return read_prolog(filename)

def read_prolog(filename) :
    import re 
    
    line_regex = re.compile( "((?P<type>(base|modes|learn))\()?((?P<prob>\d+[.]\d+)::)?\s*(?P<name>\w+)\((?P<args>[^\)]+)\)\)?." )
    
    kb = KnowledgeBase([])
    
    with open(filename) as f :
        for line in f :        
            if line.strip().startswith('%') :
                continue
            m = line_regex.match(line.strip())
            if m : 
                ltype, pred, args = m.group('type'), m.group('name'), list(map(lambda s : s.strip(), m.group('args').split(',')))
                prob = m.group('prob')
                
                if ltype == 'base' :
                    kb.register_predicate(pred, args)  
                elif ltype == 'modes' :
                    kb.add_mode(pred,args)                    
                elif ltype == 'learn' :
                    kb.add_learn(pred,args)                
                else :
                    if not prob :
                        prob = 1.0
                    else :
                        prob = float(prob)
                    kb.add_fact(pred,args,prob)
                                
    return kb
    
def read_arff(filename) :
    kb = KnowledgeBase([])
    
    atts = []
    with open(filename) as f :
        reading_data = False
        i = 0
        for line in f :
            line = line.strip()
            if line.startswith('#') :
                continue    # comment
            elif line.startswith('@attribute') :
                _a, name, tp = line.split()
                assert(tp == 'numeric')
                atts.append( name )
                kb.register_predicate(name, ['ex'])
            elif line.startswith('@data') :
                reading_data = True
            elif line and reading_data :
                data = map(float,line.split(','))
                
                for a,v in zip(atts,data) :
                    kb.add_fact(a,[str(i)],v)
                i += 1

        for a in atts[:-1] :
            kb.add_mode(a, ['+'])
        kb.add_learn(atts[-1], ['ex'])
    return kb

def main(files=[]) :

    LSETTINGS = dict(vars(SETTINGS))
    del LSETTINGS['files']

    
    for filename in files :
        
        kb = read_file(filename)
        
        varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        targets = kb.learn
        
        filename = os.path.split(filename)[1]
        
        with open(filename+'.xml', 'w') as Log.LOG_FILE :
            
            with Log('log') :
                for pred, args in targets :
                  with Timer('learning time') as t :
                    
                    kb.idtypes = args
                    kb.reset_examples()
                    
                    target = Literal(pred, varnames[:len(args)] )
                    
                    print('==> LEARNING CONCEPT:', target)
                    with Log('learn', input=filename, target=target, _timer=True, **LSETTINGS ) :
                        H = learn(FalseRule(kb, target))   
                        
                        print (H.strAll())
                        print(H.score, H.globalScore)
                        
                        with Log('result') as l :
                            if l.file :
                                print(H.strAll(), file=l.file)
                                


if __name__ == '__main__' :
    
    main(SETTINGS.files)
  