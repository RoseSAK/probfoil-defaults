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

import os
import time
import math
import sys

from util import Log
from language import Literal

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
        
        if not rule.previous : return # Don't do anything when rule is root 
        
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
        
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        self._prepare_rule(rule)
        self._add_to_queue(rule)
        
    def getFact(self, fact) :
        return self.engine.getFactProbability( self._toProlog(fact) )
        
    def _add_to_queue(self, rule) :
        
        debug_case_counters = [0] * 3
        
        for ex_id, example in enumerate(rule.examples) :
            if rule.parent and rule.parent.score_predict[ex_id] == 0 :
                # parent rule predicts 0 => current rule will also predict 0
                pass 
                debug_case_counters[0] += 1
#            elif rule.previous and rule.previous.score_predict[ex_id] == 1 :
                # previous rule predicts 1 => current rule irrelevant
#                pass
            elif not rule.previous : # evaluating root rule
                query_head = self._toProlog(rule.target.withArgs(example))
                self.__queue.append( ( rule, ex_id, query_head ) )
                debug_case_counters[1] += 1
            else :
                # rule requires evaluation for this example
                query_head = self._toProlog(self._getRuleSetQueryAtom(rule, example))
                self.__queue.append( ( rule, ex_id, query_head ) )
                debug_case_counters[2] += 1
        
    def _ground_queue(self, ground_queue) :
        ground_time = time.time()
        
        # Separate tuples in queue
        rules, ex_ids, queries = zip(*ground_queue)
        
        # Ground all queries simultaneously (returns node_ids for each individual query)
        node_ids = self.engine.groundQuery(*queries)
        
        # Initialize evaluation queue
        evaluation_queue = defaultdict(list)
        
        # For debugging / logging
        debug_case_counters = [0] * 4
        
        # Run through results and build evaluation queue
        for rule, ex_id, node_id in zip(rules, ex_ids, node_ids) :
            if rule.score_predict == None : rule.score_predict = [0] * len(rule.examples)
            
            if node_id == None:
                # Grounding failed (query failed)
                pass    # Don't do anything (score_predict is 0 by default)
                debug_case_counters[0] += 1
            elif node_id == 0 :
                # Grounding is empty (query deterministically true)
                # Set score for this examples and rule
                rule.score_predict[ex_id] = 1
                debug_case_counters[1] += 1
            else :
                negated = node_id < 0
                node_id = abs(node_id)
                if node_id in self.__prob_cache :
                    # This node was evaluated earlier => reuse
                    p = self.__prob_cache[node_id]
                    if negated : p = 1-p
                    rule.score_predict[ex_id] = p
                    debug_case_counters[2] += 1
                else :
                    # Add node to evaluation queue
                    evaluation_queue[node_id].append( (rule, ex_id, negated) )
                    debug_case_counters[3] += 1
        
        ground_time = time.time() - ground_time
        with Log('grounding', time=ground_time, queue_size=len(ground_queue), fail=debug_case_counters[0], true=debug_case_counters[1], cache=debug_case_counters[2], queue=debug_case_counters[3] ) : pass
        Log.TIMERS['grounding'] += ground_time
        
        # Return evaluation queue
        return evaluation_queue
        
    def _evaluate_queue(self, evaluation_queue) :
        
        # Get list of nodes to evaluate
        nodes = list(evaluation_queue.keys())
        
        if not nodes : return # Nothing to do
        
        # Convert grounding to CNF
        cnf, facts = self.engine.getGrounding().toCNF( nodes )
        
        # If cnf is not empty (possible if all nodes are facts)
        if len(cnf) > 1 :
            # Compile the CNF
            compile_time = time.time()
            with Log('compile', _timer=True) :
                compiled_cnf = self._compile_cnf(cnf, facts)
            compile_time = time.time() - compile_time
            Log.TIMERS['compiling'] += compile_time
        
        for node_id in evaluation_queue :
            assert(node_id != 0 and node_id != None)
            assert(not node_id in self.__prob_cache)
            
            # Retrieve probability p for this node
            if node_id in facts :
                # Node is a fact
                p = facts[node_id]
            else :
                # Node was evaluated
                p = compiled_cnf[node_id]
                self.__prob_cache[node_id] = p
                
            # Store probability in rule
            for rule, ex_id, negated in evaluation_queue[node_id] :
                if negated :
                    rule.score_predict[ex_id] = 1-p
                else :
                    rule.score_predict[ex_id] = p
        
        
    def process_queue(self) :
        
        # Ground stored queries => returns nodes to be evaluated
        evaluations = self._ground_queue(self.__queue)

        # Clear queue
        self.__queue = []
            
        # Evaluate nodes
        self._evaluate_queue(evaluations)
        
    def _compile_cnf(self, cnf, facts) :
        import subprocess
        
        # Compile CNF to DDNNF
        cnf_file = '/tmp/probfoil_eval.cnf'
        nnf_file = os.path.splitext(cnf_file)[0] + '.nnf'
        with open(cnf_file,'w') as f :
            for line in cnf :
                 print(line,file=f)
                 
        executable = os.environ['PROBLOGPATH'] + '/assist/linux_x86_64/dsharp'
        subprocess.check_output([executable, "-Fnnf", nnf_file , "-disableAllLits", cnf_file])
        
        if sys.path[-1] != os.environ['PROBLOGPATH'] + '/src/' :
            sys.path.append(os.environ['PROBLOGPATH'] + '/src/')
        # Evaluate DDNNF
        from compilation.compile import DDNNFFile
        from evaluation.evaluate import FileOptimizedEvaluator, DDNNF
        
        ddnnf = DDNNFFile(nnf_file, None)
        ddnnf.atoms = lambda : list(range(1,len(self.engine.getGrounding())+1))   # OMFG what a hack
        
        return FileOptimizedEvaluator()(knowledge=ddnnf, probabilities=facts, queries=None, env=None)[1]


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
            if 0 in content : # Certainly true in node
                if nodetype == 'or' :   
                    return 0    # Or is certainly true
                else :  # nodetype == 'and'
                    # filter out 0
                    content = tuple(filter(lambda x : x != 0, content))
            
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
            node_id = q
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
    
        
    
            