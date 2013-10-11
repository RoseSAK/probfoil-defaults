import sys

from prolog_interface import PrologInterface, Grounding
from collections import defaultdict, namedtuple
from language import Literal

class PyPrologInterface(PrologInterface) :
    
    def __init__(self) :
        super().__init__()
       
    def query(self, query, variables) :
        q = self._toProlog(query)
        for context, hasNext, prob in self.engine.executeQuery(q) :
            yield [ context[X] for X in variables ]
        
    def _createPrologEngine(self) :
#        import prolog.core as prolog
        return GroundingEngine()
         
    def _toPrologClause(self, head, *body, probability=1) :
        import prolog.core as prolog
        pl_head = self._toProlog(head)
        if probability != 1 : 
            pl_head.probability = prolog.Constant(probability)
            functor = '<-'
        else :
            functor = ':-'
            
        i = len(body) - 1
        pl_body = self._toProlog(body[i])
        while i > 0 :
            i -= 1
            pl_body = prolog.Conjunction( self._toProlog(body[i]), pl_body )
        return prolog.Clause( pl_head, pl_body, functor=functor )
                
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
        self.ground_cache = Grounding()
        
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
                    facts.append( self.ground_cache.addFact( result_atom.probability.computeValue(context), 'PROB_' + str(result_atom) ) )
                    results[result_atom].append( self.ground_cache.addNode('and',tuple(facts)) )
                elif facts :
                    results[result_atom].append( self.ground_cache.addNode('and',tuple(facts)) )
                elif result_atom.probability :
                    node_id = self.ground_cache.addFact(result_atom.probability.computeValue(context), str(result_atom) )
                    results[result_atom].append( node_id )
                else :
                    results[result_atom].append(0)
            
            new_results = []
            for result_atom in results :
                facts = results[result_atom]
                if 0 in facts :
                    self.ground_cache.addNode( 'or', [], name=result_atom)
                    results[result_atom] = []
                    new_results.append( (result_atom, 0 ))
                else :
                    new_results.append( (result_atom, self.ground_cache.addNode( 'or',tuple(facts), name=result_atom) ) )
            self.now_grounding.remove( ground_atom )
        
            for result_atom, idx in new_results :
                with context :
                    call_atom.unify(result_atom, context, context)
                    call_atom.probability = result_atom.probability
                    if results[result_atom] : 
                        context.pushFact(idx)
                    elif call_atom.probability != None :
                        node_id = self.ground_cache.addFact(result_atom.probability.computeValue(context), str(result_atom), result_atom )
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
                
    def getGrounding(self) :
        return self.ground_cache
    
        
