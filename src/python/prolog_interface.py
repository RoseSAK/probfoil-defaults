import sys
sys.path.append('../../')

import os
import time
import math

from util import Log
from language import Literal, Language, RootRule
from learn import ProbFOIL, ProbFOIL2

from collections import namedtuple, defaultdict


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
            clause1 = self._toPrologClause(clause_head, prev_head, probability=rule.previous.probability )
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
        
        cnf_file = '/tmp/probfoil_eval.cnf'
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
        
class Grounding(object) :
    
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
        
    def addFact(self, probability, identifier=None, name=None) :
        if identifier == None :
            identifier = 'FACT_' + str(len(self.__facts))
        
        key = ('fact', identifier)
        result = self.__content.get(key, None)
        if result == None : result = self._addNode(key)
        
        self.__facts[ identifier ] = probability
        
        if name : self.__names[name] = result
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
                facts[k] = self.__facts[ content ]
                # if content.probability:
                #     facts[k] = content.probability.computeValue(None)
                # else :
                #     facts[k] = 1 
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


