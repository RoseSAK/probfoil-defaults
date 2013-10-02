import sys

sys.path.append('../../')

import os

import prolog.core as prolog

from collections import namedtuple, defaultdict
from itertools import product

#from core import PrologEngine, Clause, Conjunction

class LearningProblem(object) :
    
    def __init__(self, language, knowledge) :
        self.language = language
        self.knowledge = knowledge
        
    def calculateScore(self, rule) :
        raise NotImplementedError('calculateScore')

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
    

class ProbFOIL(LearningProblem) :
    
    def calculateScore(self, rule) :
        return PF1Score(rule.score_correct, rule.score_predict)

class ProbFOIL2(LearningProblem) :
    
    def calculateScore(self, rule) :
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


class Rule(object) :
    
    def __init__(self, parent) :
        self.parent = parent
        self.literal = None
        self.__score = None
        self.score_predict = None
        
    def __add__(self, literal) :
        """Adds the given literal to the body of the rule and returns the new rule."""
        return RuleBody(literal, self)
        
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
        return id(self)
    
    def _get_variables(self) :
        return self.target.variables
        
    def _get_examples(self) :
        return self.previous.examples    
                
    def _get_learning_problem(self) :
        return self.previous.learning_problem
                
    target = property( lambda s : s._get_target() )
    previous = property( lambda s : s._get_previous() )
    identifier = property( lambda s : s._get_identifier() )
    variables = property( lambda s : s._get_variables() )
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
                lines.append( str(self.target) + ' <- ' + body + '.' )
            else :
                lines.append( str(self.target) + '.' )
        return '\t'.join(lines)
    
    def refine(self, update=False) :
        """Generate refinement literals for this rule."""
        # generate refined clauses
        if update :
            # Only generate extensions that use a variable introduced by the last literal
            use_vars = self._new_vars
            if not use_vars : return [] # no variables available
        else :
            use_vars = None
        return [ literal for literal in self.language.refinements( self.variables, use_vars ) ]
        
    def _get_score(self) :
        if self.__score == None :
            self.__score = self.learning_problem.calculateScore(self)
        return self.__score
    
    def _get_score_predict(self) :
        return self.__score_predict
        
    def _set_score_predict(self, predictions) :
        self.__score_predict = predictions
    
    def _get_score_correct(self) :
        return self.previous.score_correct
    
    score_correct = property( lambda s: s._get_score_correct() )
    score_predict = property( lambda s : s._get_score_predict(), lambda s,v : s._set_score_predict(v) )
        
    score = property ( lambda s : s._get_score() )
    
    
class RuleBody(Rule) :
    """Rule with at least one literal in the body."""
    
    def __init__(self, literal, parent) :
        """Create a new FOIL rule by adding the given literal to the give parent rule."""
        super(RuleBody, self).__init__(parent)
        self.literal = literal
        
        old_vars = parent.variables
            
        current_vars = set(literal.variables)
        self._all_vars = old_vars | current_vars
        self._new_vars = current_vars - old_vars
        
    def _get_variables(self) :
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
        self._new_vars = current_vars
    
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
            scores.append( self.knowledge.query( self.target.withArgs(example) ) )
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
        
    variables = property( _get_variables )  
    arity = property (lambda s : len(s.arguments) )
    key = property( lambda s : ( s.functor, s.arity ) )
    
    def __str__(self) :
        not_sign = '\+' if self.is_negated else ''
        r = not_sign + self.functor
        if self.arguments :
            return r + '(' + ','.join(self.arguments) + ')'
        else :
            return r

class Language(object) :
    
    def __init__(self) :
        self.__types = {}
        self.__values = defaultdict(set)
        
    def addValue(self, typename, value) :
        self.__values[typename].add(value)
        
    def setArgumentTypes(self, literal) :
        self.__types[ literal.key ] = literal.arguments
        
    def getArgumentTypes(self, literal) :
        return self.__types[ literal.key ]
                
    def getTypeValues(self, typename) :
        return self.__values[ typename ]
        
    def getArgumentValues(self, literal) :
        types = self.getArgumentTypes(literal)
        return list( product( *[ self.getTypeValues(t) for t in types ] ) )

class PrologInterface(object) :
    
    def __init__(self) :
        self.engine = prolog.PrologEngine()
        self.last_id = 0
        self.__queue = []
        self.__prob_cache = {}
                
    def _getRuleQuery(self, identifier) :
        return 'query_' + str(identifier)
        
    def _getRuleQueryAtom(self, rule) :
        functor = self._getRuleQuery(rule.identifier)
        args = [ prolog.Variable(x) for x in rule.variables ]
        return prolog.Function( functor, args )

    def _getRuleSetQueryAtom(self, rule, arguments=None) :
        functor = self._getRuleSetQuery(rule.identifier)
        if arguments == None :
            args = list(map(self._toProlog, rule.target.arguments))
        else :
            args = list(map(self._toProlog, arguments))
        return prolog.Function( functor, args )
        
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
            clause1 = prolog.Clause(clause_head, prev_head)
            self.engine.addClause( clause1 )
        
        if rule.parent :
            if rule.parent.literal :
                clauseB = prolog.Clause(clause_body, prolog.Conjunction( self._getRuleQueryAtom(rule.parent), self._toProlog(rule.literal) ) )
            else :
                clauseB = prolog.Clause(clause_body, self._toProlog(rule.literal) )
            self.engine.addClause( clauseB )
        else :
            clause_body = None
        clause2 = prolog.Clause(clause_head, clause_body )
        self.engine.addClause( clause2 )
        
        # current_case :- parent_rule(A,B,C,D), new_literal(A,D).
        # new_query(A,B) :- previous_cases(A,B).
        # new_query(A,B) :- current_case(A,B,C,D).
                
    def _ground_rule(self, rule, examples) :
        """Execute grounding procedure for the given rule with the given examples."""
        
        functor = self._getRuleSetQuery(rule.identifier)
        for ex in examples :
            args = [ self._toProlog(x) for x in ex ]
            query = prolog.Function(functor, args)
            self.engine.groundQuery(query)  # => should return node id
            
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        self._prepare_rule(rule)
        self._ground_rule(rule, rule.examples)
        self._add_to_queue(rule)
        
    def _add_to_queue(self, rule) :
        self.__queue.append(rule)
        
    def process_queue(self) :
        queries = []
        for rule in self.__queue :
            if rule.score_predict == None :
                for example in rule.examples :
                    queries.append( self._getRuleSetQueryAtom(rule, example) )
            
        cnf, facts = self.engine.ground_cache.toCNF( queries )
        ddnnf = self._compile_cnf(cnf, facts)
        
        Literal = namedtuple('Literal', ['atom'])
        for rule in self.__queue :
            if rule.score_predict == None :
                evaluations = []
                for example in rule.examples :
                    q = self._getRuleSetQueryAtom(rule, example)
                    q_node_id = self.engine.ground_cache.byName(q)
                    #print (q, q_node_id)
                    if q_node_id == 0 :
                        p = 1
                    elif q_node_id == None :
                        p = 0
                    elif q_node_id in facts :
                        p = facts[q_node_id]
                    elif q_node_id in self.__prob_cache :
                        p = self.__prob_cache[q_node_id]
                    else :
                        res = ddnnf.evaluate({},[ Literal(str(q_node_id)) ])
                        p = res[1][str(q_node_id)]
                        self.__prob_cache[q_node_id] = p
                    evaluations.append(p)
                rule.score_predict = evaluations
        
        self.__queue = []
                
    def _toProlog(self, term) :
        if isinstance(term, Literal) :
            args = list(map(self._toProlog, term.arguments))
            func = prolog.Function(term.functor, args)
            if term.is_negated :
                return prolog.Not(func)
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
        
    def query(self, literal) :
        num_sols = 0
        for context, hasMore, probability in self.engine.executeQuery(self._toProlog(literal)) :
            num_sols += 1
        
        if num_sols == 0 :
            return 0
        elif num_sols > 1 :
            raise Exception('Fact defined multiple times!')
        elif probability == None :
            return 1
        else :
            return probability.computeValue(None)
            
    def _compile_cnf(self, cnf, facts) :
        import subprocess
        
        cnf_file = 'probfoil_eval.cnf'
        nnf_file = os.path.splitext(cnf_file)[0] + '.nnf'
        with open(cnf_file,'w') as f :
            for line in cnf :
                 print(line,file=f)
        subprocess.check_output(["dsharp", "-Fnnf", nnf_file , "-smoothNNF", "-disableAllLits", cnf_file])
        
        from compilation.compile import DDNNFFile
        from evaluation.evaluate import DDNNF
        
        return DDNNF(DDNNFFile(nnf_file, None), facts)    

def test(filename) :
    
    # from rule import Literal, RuleHead, FalseRule
     
    import prolog.parser as pp
    parser = pp.PrologParser()
    target = Literal('parent','XY')
    
    p = PrologInterface()
    
    p.engine.loadFile(filename)
    
    l = Language()
    l.setArgumentTypes( Literal('grandmother', ('person', 'person') ) )
    l.setArgumentTypes( Literal('parent', ('person', 'person') ) )
    l.setArgumentTypes( Literal('father', ('person', 'person') ) )    
    l.setArgumentTypes( Literal('mother', ('person', 'person') ) )
    
    for v in  [ 'alice', 'an', 'esther', 'katleen', 'laura', 'lieve', 'lucy', 'rose', 'soetkin', 'yvonne', 
        'bart', 'etienne', 'leon', 'luc', 'pieter', 'prudent', 'rene', 'stijn', 'willem' ] :
        l.addValue('person',v)
    
    lp = ProbFOIL(l, p)
    
    r0 = RootRule(target, lp)
    r0.initialize()
    
    r1_1 = RuleHead(r0)
    r1_2 = r1_1 + Literal('father', 'XY')
    
    r2_1 = RuleHead(r1_2)
    r2_2 = r2_1 + Literal('mother', 'XY')

    print(r0, r0.score_correct)
            
    p.enqueue(r1_1)
    p.process_queue()
    
    print(r1_1, r1_1.score_predict, r1_1.score)

    p.enqueue(r1_2)
    p.process_queue()

    print(r1_2, r1_2.score_predict, r1_2.score)
    
    p.enqueue(r2_1)
    p.process_queue()

    print(r2_1, r2_1.score_predict, r2_1.score)
    
    p.enqueue(r2_2)
    p.process_queue()
    
    print(r2_2, r2_2.score_predict, r2_2.score)
    

    
#    p.engine.listing()
#    print(p.engine.ground_cache)

    # for ex, sc in zip(r2_2.examples, r2_2.scores) :
    #     if sc > 0 :
    #         print (ex,sc)
    
#    p.engine.listing()
    
    # fs = ['stress(%s)' % x for x in range(1,6) ]
    # for x in fs :
    #     print (x, p.query( parser.parseToken(x) ))

if __name__ == '__main__' :
    test(*sys.argv[1:])    