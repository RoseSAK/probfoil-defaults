import sys

sys.path.append('../../')


import prolog.core as prolog

#from core import PrologEngine, Clause, Conjunction

class Rule(object) :
    
    def __init__(self, parent) :
        self.parent = parent
        self.literal = None
                
    def __add__(self, literal) :
        """Adds the given literal to the body of the rule and returns the new rule."""
        return RuleBody(literal, self)
        
    def _get_target(self) :
        return self.parent.target
        
    def _get_previous(self) :
        assert(self.parent)
        return self.parent.previous
        
    def _get_identifier(self) :
        return id(self)
    
    def _get_variables(self) :
        return self.target.variables
        
    def _get_examples(self) :
        return []
        
    def setEvaluation(self, example_scores) :
        pass
        
    target = property( lambda s : s._get_target() )
    previous = property( lambda s : s._get_previous() )
    identifier = property( lambda s : s._get_identifier() )
    variables = property( lambda s : s._get_variables() )
    
    examples = property ( lambda s : s._get_examples() )
    
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
        
class FalseRule(Rule) :
    """Rule with fail in the body."""
    
    def __init__(self, target) :
        self.__target = target
        super(FalseRule,self).__init__(None)
        
    def _get_target(self) :
        return self.__target
    
    def _get_previous(self) :
        return None
        
    def _get_identifier(self) :
        return None

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

    def _getRuleSetQueryAtom(self, rule) :
        functor = self._getRuleSetQuery(rule.identifier)
        args = rule.target.arguments
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
                clauseB = prolog.Clause(clause_body, prolog.Conjunction( self._getRuleQueryAtom(rule.parent), rule.literal ) )
            else :
                clauseB = prolog.Clause(clause_body, rule.literal )
            self.engine.addClause( clauseB )
        else :
            clause_body = None
        clause2 = prolog.Clause(clause_head, clause_body )
        self.engine.addClause( clause2 )
        
        # current_case :- parent_rule(A,B,C,D), new_literal(A,D).
        # new_query(A,B) :- previous_cases(A,B).
        # new_query(A,B) :- current_case(A,B,C,D).
        
    def _toIdentifier(self, x) :
        try :
            x = int(x)
            return prolog.Constant(x)
        except ValueError :
            return prolog.Function(x)
        
    def _ground_rule(self, rule, examples) :
        functor = self._getRuleSetQuery(rule.identifier)
        for ex in examples :
            args = [ self._toIdentifier(x) for x in ex ]
            query = prolog.Function(functor, args)
            self.engine.groundQuery(query)
            
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        self._prepare_rule(rule)
        self._ground_rule(rule, rule.examples)
        self._add_to_queue(rule)
        
    def _add_to_queue(self, rule) :
        self.__queue.append(rule)
        
    def process_queue(self) :

        for rule in self.__queue :
            pass
            # TODO run evaluation
        
        # 1) Construct CNF from Index() ( easy if loop-free )
        # 2) ProbLog: NF = compile(cnf_form, [])  # there is no evidence => also bypass ProbLog here
        # 3) ProbLog: results = evaluate(NF, probabilities, queries)
        
        self.__queue = []
        
    # def _compile(self) :
    #     'dsharp' : (lambda cnf_file : [bin_path('dsharp', True), '-Fnnf', cnf_file + '.nnf', '-Fgraph', cnf_file + '.nnf.graph', '-smoothNNF', '-disableAllLits', cnf_file ])
        
    # def _extract_depgraph(self, dg, queries) :
    #     nodes = set([])
    #     for query in queries :
    #         nodes = dg.dependents(query, nodes)
    

def test(filename) :
    
    # from rule import Literal, RuleHead, FalseRule
     
    import prolog.parser as pp
    parser = pp.PrologParser()
    target = parser.parseToken('smokes(X)')
    
    p = PrologInterface()
    
    p.engine.loadFile(filename)
    
    r0 = FalseRule(target)
    
    r1_1 = RuleHead(r0)
    r1_2 = r1_1 + parser.parseToken('\+stress_free(X)')
    
    r2_1 = RuleHead(r1_2)
    r2_2 = r2_1 + parser.parseToken('friend(X,Y)')
    r2_3 = r2_2 + parser.parseToken('influences(Y,X)')
    r2_4 = r2_3 + parser.parseToken('smokes(Y)')
        
    #p.prepare_rule(r1)
    p._prepare_rule(r1_1)
    p._ground_rule(r1_1, ['1','2','3','4'])
    
    p._prepare_rule(r1_2)
    p._ground_rule(r1_2, ['1','2','3','4'])
    
    p._prepare_rule(r2_1)
    p._ground_rule(r2_1, ['1','2','3','4'])
    
    p._prepare_rule(r2_2)
    p._ground_rule(r2_2, ['1','2','3','4'])
    
    p._prepare_rule(r2_3)
    p._ground_rule(r2_3, ['1','2','3','4'])
    
#    p._prepare_rule(r2_4)
#    p._ground_rule(r2_4, ['1','2','3','4'])
    
    print(p.engine.ground_cache)
    
    p.engine.listing()
    
    print ('\n'.join(p.engine.ground_cache.toCNF()))

if __name__ == '__main__' :
    test(*sys.argv[1:])    