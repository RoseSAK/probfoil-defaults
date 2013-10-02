import sys

sys.path.append('../../')

import os

import prolog.core as prolog

from collections import namedtuple
from itertools import product

#from core import PrologEngine, Clause, Conjunction

class Rule(object) :
    
    def __init__(self, parent) :
        self.parent = parent
        self.literal = None
                
    def __add__(self, literal) :
        """Adds the given literal to the body of the rule and returns the new rule."""
        return RuleBody(literal, self)
        
    def _get_target(self) :
        return self.previous.target
        
    def _get_previous(self) :
        assert(self.parent)
        return self.parent.previous
        
    def _get_identifier(self) :
        return id(self)
    
    def _get_variables(self) :
        return self.target.variables
        
    def _get_examples(self) :
        return self.previous.examples    
        
    def setEvaluation(self, example_scores) :
        self.scores = example_scores
                
    target = property( lambda s : s._get_target() )
    previous = property( lambda s : s._get_previous() )
    identifier = property( lambda s : s._get_identifier() )
    variables = property( lambda s : s._get_variables() )
    language = property( lambda s : s._get_language() )
    knowledge = property( lambda s : s._get_knowledge() )
    
    examples = property ( lambda s : s._get_examples() )
    
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
    
class RuleBody(Rule) :
    """Rule with at least one literal in the body."""
    
    def __init__(self, literal, parent) :
        """Create a new FOIL rule by adding the given literal to the give parent rule."""
        super(RuleBody, self).__init__(parent)
        self.literal = literal
        
        if parent :
            old_vars = parent.variables
        else :
            old_vars = set([])
            
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
        self.__previous = previous
        super(RuleHead,self).__init__(None)
                
    def _get_target(self) :
        return self.previous.target
    
    def _get_previous(self) :
        return self.__previous
        
    def _str_parts(self) :
        par = self.previous._str_parts()
        par.append([] )
        return par
        
class FalseRule(Rule) :
    """Rule with fail in the body."""
    
    def __init__(self, target, language, knowledge) :
        self.__target = target
        self.__language = language
        self.__knowledge = knowledge
        self.__examples = []
        super(FalseRule,self).__init__(None)
    
    def _get_language(self) :
        return self.__language
    
    def _get_target(self) :
        return self.__target
    
    def _get_previous(self) :
        return None
        
    def _get_identifier(self) :
        return None
        
    def _get_knowledge(self) :
        return self.__knowledge
        
    def _get_examples(self) :
        return self.__examples
        
    def _get_root(self) :
        return self
        
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
            scores.append( 1 - self.knowledge.query( self.target.withArgs(example) ) )
        self.setEvaluation(scores)
    
    def _str_parts(self) :
        return []

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
    
    def __str__(self) :
        not_sign = '\+' if self.is_negated else ''
        r = not_sign + self.functor
        if self.arguments :
            return r + '(' + ','.join(self.arguments) + ')'
        else :
            return r

class Language(object) :
    
    def __init__(self) :
        pass
        
    def getArgumentTypes(self, literal) :
        return ['person', 'person']
        
    def getTypeValues(self, typename) :
        if typename == 'person' :
            return [ 'alice', 'an', 'esther', 'katleen', 'laura', 'lieve', 'lucy', 'rose', 'soetkin', 'yvonne', 'bart', 'etienne', 'leon', 'luc', 'pieter', 'prudent', 'rene', 'stijn', 'willem' ]
        else :
            raise ValueError('Unknown type \'%s\'!' % typename)
        
    def getArgumentValues(self, literal) :
        types = self.getArgumentTypes(literal)
        return list( product( *[ self.getTypeValues(t) for t in types ] ) )
        

class PrologInterface(object) :
    
    def __init__(self) :
        self.engine = prolog.PrologEngine()
        self.last_id = 0
        self.__queue = []
                
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
            for example in rule.examples :
                queries.append( self._getRuleSetQueryAtom(rule, example) )
            
        cnf, facts = self.engine.ground_cache.toCNF( queries )
        ddnnf = self._compile_cnf(cnf, facts)
        
        Literal = namedtuple('Literal', ['atom'])
        for rule in self.__queue :
            evaluations = []
            for example in rule.examples :
                q = self._getRuleSetQueryAtom(rule, example)
                q_node_id = self.engine.ground_cache.byName(q)
                print (q, q_node_id)
                if q_node_id == 0 :
                    p = 1
                elif q_node_id == None :
                    p = 0
                elif q_node_id in facts :
                    p = facts[q_node_id]
                else :
                    res = ddnnf.evaluate({},[ Literal(str(q_node_id)) ])
                    p = res[1][str(q_node_id)]
                evaluations.append(p)
            rule.setEvaluation(evaluations)
        
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
    
    r0 = FalseRule(target, Language(), p)
    r0.initialize()
    
    r1_1 = RuleHead(r0)
    r1_2 = r1_1 + Literal('father', 'XY')
    
    r2_1 = RuleHead(r1_2)
    r2_2 = r2_1 + Literal('mother', 'XY')
            
    p.enqueue(r1_1)
    
    p.enqueue(r1_2)
    
    p.enqueue(r2_1)
    
    p.enqueue(r2_2)
    
    p.process_queue()

    print(r0, r0.scores)
    print(r1_1, r1_1.scores)
    print(r1_2, r1_2.scores)
    print(r2_1, r2_1.scores)
    print(r2_2, r2_2.scores)

    for ex, sc in zip(r2_2.examples, r2_2.scores) :
        if sc > 0 :
            print (ex,sc)
    
#    p.engine.listing()
    
    # fs = ['stress(%s)' % x for x in range(1,6) ]
    # for x in fs :
    #     print (x, p.query( parser.parseToken(x) ))

if __name__ == '__main__' :
    test(*sys.argv[1:])    