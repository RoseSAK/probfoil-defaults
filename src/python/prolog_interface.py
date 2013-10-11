
import os
import time
import math

from util import Log
from language import Literal, Language, RootRule
from learn import ProbFOIL, ProbFOIL2

from collections import namedtuple, defaultdict


# Goal: use existing ProbLog grounder
#   but keep incremental grounding
# Idea:
#   - add already grounded information as facts for future
#       => potential problem => predicates not grounded for all variables


class PrologEngine(object) :
    """Expected interface for PrologEngine."""
    
    def __init__(self) :
        pass
        
    def loadFile(self, filename) :
        """Load a Prolog source file."""
        raise NotImplementedError('Calling an abstract method!')

    def listing(self) :
        """Return a string listing the content of the current Prolog database."""
        raise NotImplementedError('Calling an abstract method!')
        
    def addClause(self, clause) :
        """Add a clause to the database."""
        raise NotImplementedError('Calling an abstract method!')

    def query(self, literal) :
        """Execute a query."""
        raise NotImplementedError('Calling an abstract method!')

    def groundQuery(self, literal) :
        """Ground a query."""
        raise NotImplementedError('Calling an abstract method!')
    
    def getFactProbability(self, literal) :
        """Retrieve the fact probability."""
        raise NotImplementedError('Calling an abstract method!')
    
    def getGrounding(self) :
        """Get grounding information."""
        raise NotImplementedError('Calling an abstract method!')

    grounding = property(lambda s : s.getGrounding() )

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
        
        ground_time = time.time()
        functor = self._getRuleSetQuery(rule.identifier)
        nodes = []
        requires_problog = False
        
        queries_ran = 0
        to_run = []
        for ex_id, ex in enumerate(examples) :
            if not rule.parent or rule.parent.score_predict[ex_id] != 0 :
                query = self._toProlog( Literal( functor, ex ) )
                to_run.append(query)                
#                node_id = self.engine.groundQuery(query)  # => should return node id
#                queries_ran += 1
                node_id = 'RUN'
            else :
                node_id = None  # rule parent predicted FALSE for this rule
            nodes.append(node_id)
            if node_id :    # not None or 0
                requires_problog = True
        
        print('Grounding', len(to_run), 'queries.')
        node_ids = self.engine.groundQuery(*to_run)
        i = 0
        for j, n in enumerate(nodes) :
            if n == 'RUN' :
                nodes[j] = node_ids[i]
                i += 1
        
        rule.nodes = nodes
        rule.requires_problog = requires_problog
        ground_time = time.time() - ground_time
        
        Log.TIMERS['grounding'] += ground_time
        #with Log('enqueue', rule=rule.literal, queries=queries_ran, ground_time=ground_time) : pass
            
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        
        self._prepare_rule(rule)
        
        self._ground_rule(rule, rule.examples)
        
        self._add_to_queue(rule)
        
    def getFact(self, fact) :
        return self.engine.getFactProbability( self._toProlog(fact) )
        
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
            cnf, facts = self.engine.getGrounding().toCNF( queries )
            if len(cnf) > 1 :
                compile_time = time.time()
                print ('COMPILING CNF ', cnf)
                with Log('compile', _timer=True) :
                    ddnnf = self._compile_cnf(cnf, facts)
                compile_time = time.time() - compile_time
                Log.TIMERS['compiling'] += compile_time
        for rule in self.__queue :
            if rule.score_predict == None :
                evaluations = []
                for ex_id, example in enumerate(rule.examples) :
                    q_node_id = rule.nodes[ex_id]
                    
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
                            #print('FROM CACHE', rule.literal, q_node_id, p)
                        else :
                            
                            p = ddnnf[1][q_node_id]
                            self.__prob_cache[q_node_id] = p
                            #print('EVALUATE', rule.literal, q_node_id, p)
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
        ddnnf.atoms = lambda : list(range(1,len(self.engine.getGrounding())+1))   # OMFG what a hack
        
        return FileOptimizedEvaluator()(knowledge=ddnnf, probabilities=facts, queries=None, env=None)


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
        
    names = property(lambda s : s.__names)
        
    def getProbability(self, name) :
        return self.__facts[name]
        
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
        
        if probability == 1 :
            result = 0
        else :
            key = ('fact', identifier)
            result = self.__content.get(key, None)
            if result == None : result = self._addNode(key)
        
            self.__facts[ identifier ] = probability
        if name : self.__names[name] = result
        return result
    
    def updateNode(self, nodeid, nodetype, content, name=None) :
        key = self[nodeid]
        newkey = (nodetype,content)
        del self.__content[key]
        self.__content[newkey] = nodeid
        self.__index[nodeid-1] = newkey
        self._update_refcount(nodeid, content, name)
        if name : self.__names[name] = nodeid
        return nodeid
        
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

    def reserveNode(self, name) :
        return self._addNode(name)

    def integrate(self, nodes) :
        old_count = len(self)
        
        translation = {}
        for line_id, line in enumerate(nodes) :
            if line_id > 0 :
                self._integrate_node(line_id, line[0], line[1], line[2], nodes, translation)
                
        return len(self) - old_count
            
    def _integrate_node(self, line_id, line_type, line_content, name, lines, translation) :
        if name and name in self.__names :
            node_id = self.byName(name)
            translation[line_id] = node_id
            return node_id
        elif line_id and line_id in translation : 
            node_id = translation[line_id]
            if node_id == None :
                self.hasCycle = True
                node_id = self.reserveNode(name)
                translation[line_id] = node_id
                return node_id
            else :
                return translation[line_id] # node already integrated
            
        if line_id != None : translation[line_id] = None

        if line_type == 'fact' :
            # 
            node_id = self.addFact(line_content, name, name)
        else :
            # line_type in ('or', 'and')
            subnodes = []
            for subnode in line_content :
                if type(subnode) == tuple :
                    subnodes.append(self._integrate_node(None, subnode[0], subnode[1], None, lines, translation))
                else :
                    subnode_id = int(subnode)
                    neg = -1 if subnode_id < 0 else 1
                    subnode_id = abs(subnode_id)
                    subnode = lines[subnode_id]
                    subnodes.append(neg * self._integrate_node(subnode_id, subnode[0], subnode[1], subnode[2], lines, translation))

            if line_id != None and translation[ line_id ] != None :
                node_id = translation[ line_id ]
                self.updateNode(node_id, line_type, tuple(subnodes), name )
            else :
                node_id = self.addNode(line_type, tuple(subnodes), name)
            
        if line_id != None : 
            translation[ line_id ] = node_id
        return node_id
    
        
    
            