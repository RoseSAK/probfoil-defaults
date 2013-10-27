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
import re
import subprocess

from util import Log, Timer
from language import Literal

from collections import namedtuple, defaultdict


class PrologInterface(object) :
    
    def __init__(self, env) :
        self.env = env
        self.engine = self._createPrologEngine()
        self.last_id = 0
        self.__queue = []
        
        self.__prob_cache = {}
        
        self.preground = None
        
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
            prev_head = Literal('pf_prev', rule.target.arguments)
            
            if self.preground == None :
                self.preground = []
                self.__prob_cache = {}
                
                # There is a previous rule
                for ex_id, example in self._get_examples_for_queue(rule.previous) :
                    if ex_id == None : ex_id = 0
                    prev_node = rule.previous.eval_nodes[ex_id]
                    if prev_node == 0 :
                        self.preground.append( 'pf_prev( %s )' % ','.join(example) )
                    elif prev_node != None :
                        if prev_node < 0 :
                            prev_node = '9' + str(abs(prev_node)) + '1'
                        else :
                            prev_node = '0' + str(prev_node) + '1'
                        self.preground.append('0.%s::pf_prev( %s )' % ( prev_node, ','.join(example) ) )
                for c in self.preground :
                    self.engine.addClause(c)
        else :
            prev_head = None
            
        clause_body = self._getRuleQueryAtom(rule)
            
        if prev_head :  # not the first rule
            clause1 = self._toPrologClause(clause_head, prev_head )
            self.engine.addClause( clause1 )
            
        if rule.parent :
            clause_body = rule.literals
        else :
            clause_body = [None]
        clause2 = self._toPrologClause(clause_head, *clause_body ) # , probability=0.8 )
        self.engine.addClause( clause2 )
        

        
    def enqueue(self, rule) :
        """Enqueue rule for evaluation."""
        with Timer(category='enqueue') :
            with Timer(category='enqueue_prepare') :
                self._prepare_rule(rule)
            with Timer(category='enqueue_add') :
                self._add_to_queue(rule)
        
    def getFact(self, fact) :
        return self.engine.getFactProbability( self._toProlog(fact) )
                
    def _add_to_queue(self, rule) :
        
        debug_case_counters = [0] * 3
        
        for ex_id, example in self._get_examples_for_queue(rule) : # enumerate(rule.examples) :
            if not rule.previous : # evaluating root rule
                query_head = self._toProlog(rule.target.withArgs(example))
                self.__queue.append( ( rule, ex_id, query_head ) )
                debug_case_counters[1] += 1
            else :
                # rule requires evaluation for this example
                query_head = self._toProlog(self._getRuleSetQueryAtom(rule, example))
                self.__queue.append( ( rule, ex_id, query_head ) )
                debug_case_counters[2] += 1
        
    def _ground_queue(self, ground_queue) :
        
      with Timer(category='evaluate_grounding') as tmr : 
        
        # Separate tuples in queue
        rules, ex_ids, queries = zip(*ground_queue)
        
        rules_dict = {}
        for rule, query in zip(rules,queries) :
            rules_dict[query] = rule
        
        # Ground all queries simultaneously (returns node_ids for each individual query)
        node_ids = list(self.engine.groundQuery(queries, rules_dict))
        
        self.preground = None
        self.engine.clearClauses()
        
        # Initialize evaluation queue
        evaluation_queue = defaultdict(list)
                
        # Run through results and build evaluation queue
        for rule, ex_id, node_id in zip(rules, ex_ids, node_ids) :
            if rule.score_predict == None : rule.score_predict = [0] * len(rule.examples)
            if rule.eval_nodes == None : rule.eval_nodes = [None] * len(rule.examples)
            
            p = self.engine.getGrounding().getProbability(node_id)
            rule.score_predict[ex_id] = p
            rule.eval_nodes[ex_id] = node_id
            
            if p == None :
                evaluation_queue[abs(node_id)].append( (rule, ex_id, node_id < 0) )
                    
        # Return evaluation queue
        return evaluation_queue
        
    def _evaluate_queue(self, evaluation_queue) :
        
        # Get list of nodes to evaluate
        nodes = list(evaluation_queue.keys())
        
        if not nodes : return # Nothing to do
        
        # Convert grounding to CNF
        with Timer(category='evaluate_converting') :
            cnf, facts = self.engine.getGrounding().toCNF( nodes )
        
        # Compile the CNF
        evaluator = self._compile_cnf(cnf, facts)
        
        for node_id in evaluation_queue :
            assert(node_id != 0 and node_id != None)
            assert(not node_id in self.__prob_cache)
            
            with Timer(category='evaluate_evaluating') :
                
                p = evaluator.evaluate(node_id)
                
                # Store probability in rule
                for rule, ex_id, negated in evaluation_queue[node_id] :
                    if ex_id == None :
                        for ex_id, example in rule.enum_examples() :
                            p = evaluator.evaluate(node_id, rule, ex_id)
                            if negated :
                                rule.score_predict[ex_id] = 1-p
                            else :
                                rule.score_predict[ex_id] = p
                    else :
                        self.__prob_cache[ node_id ] = p
                        if negated :
                            rule.score_predict[ex_id] = 1-p
                        else :
                            rule.score_predict[ex_id] = p
            
    def process_queue(self) :
        
      with Timer(category='evaluate') :
        
        # Ground stored queries => returns nodes to be evaluated
        evaluations = self._ground_queue(self.__queue)

        # Clear queue
        self.__queue = []
            
        # Evaluate nodes
        self._evaluate_queue(evaluations)
        
    def _compile_cnf(self, cnf, facts) :
        if len(cnf) <= 1 :
            return self._construct_evaluator(None, facts)
        else :
          #with Log('compile', _timer=True) :
        
          with Timer(category='evaluate_compiling') :
            #print ('compiling CNF', len(cnf), cnf[0])

            # Compile CNF to DDNNF
            cnf_file = self.env.tmp_path('probfoil_eval.cnf')
            nnf_file = os.path.splitext(cnf_file)[0] + '.nnf'
            with open(cnf_file,'w') as f :
                for line in cnf :
                    print(line,file=f)
                 
            with Timer('Compiling %s:' % cnf[0], verbose=self.env['verbose']>1) :
                executable = self.env['PROBLOGPATH'] + '/assist/linux_x86_64/dsharp'
                subprocess.check_output([executable, "-Fnnf", nnf_file , "-disableAllLits", cnf_file])
        
            if sys.path[-1] != self.env['PROBLOGPATH'] + '/src/' :
                sys.path.append(self.env['PROBLOGPATH'] + '/src/')
            # Evaluate DDNNF
            from compilation.compile import DDNNFFile
            from evaluation.evaluate import FileOptimizedEvaluator
        
          with Timer(category='evaluate_evaluating') :
            ddnnf = DDNNFFile(nnf_file, None)
            ddnnf.atoms = lambda : list(range(1,len(self.engine.getGrounding())+1))   # OMFG what a hack
            
            return self._construct_evaluator(ddnnf, facts)
    
    def _rewrite_facts(self, facts) :
        return facts
    
    def _get_examples_for_queue(self, rule) :
        return rule.enum_examples()
            
    def _construct_evaluator(self, ddnnf, facts) :
        return DefaultEvaluator(ddnnf, self._rewrite_facts(facts))
        
    def loadData(self, filename) :
        self.engine.loadFile(filename)

        
class DefaultEvaluator(object) :
    
    def __init__(self, knowledge, facts) :
        self.__knowledge = knowledge
        self.__facts = facts
        
        if knowledge :
            from evaluation.evaluate import FileOptimizedEvaluator
            self.__base = FileOptimizedEvaluator()
            self.__result = self.__base(knowledge=self.__knowledge, probabilities=self.__facts, queries=None, env=None)[1]
        else :
            self.__result = {}
        self.__result.update(facts)
        
    def example_dependent(self) :
        return False
        
    def evaluate(self, node_id, rule=None, ex_id=None) :
        return self.__result[node_id]
        

    

class Grounding(object) :
    
    # Invariant: stored nodes do not have TRUE or FALSE in their content.
    
    TRUE = 0
    FALSE = None
    
    def __init__(self, parent=None) :
        if parent :
            self.__offset = len(parent)
        else :
            self.__offset = 0
        self.__parent = parent
        self.__nodes = []
        self.__fact_names = {}
        self.__nodes_by_content = {}
        self.__probabilities = []
        self.__usedfacts = []
        
    def _getUsedFacts(self, index) :
        if index < 0 :
            return self.__usedfacts[-index-1]
        else :
            return self.__usedfacts[index-1]
        
    def _setUsedFacts(self, index, value) :
        if index < 0 :
            self.__usedfacts[-index-1] = frozenset(value)
        else :
            self.__usedfacts[index-1] = frozenset(value)
        
    def _negate(self, t) :
        if t == self.TRUE :
            return self.FALSE
        elif t == self.FALSE :
            return self.TRUE
        else :
            return -t
            
    def addChoice(self, rule) :
        return self._addNode('choice', rule)
        
    def addFact(self, name, probability) :
        """Add a named fact to the grounding."""
        assert(not name.startswith('pf_'))
        node_id = self.__fact_names.get(name,None)
        if node_id == None : # Fact doesn't exist yet
            node_id = self._addNode( 'fact', (name, probability) )
            self.__fact_names[name] = node_id
            # TODO check whether it is a meta-fact
            self.setProbability(node_id, probability)
            self._setUsedFacts(node_id,[abs(node_id)])
        return node_id
        
    def addNode(self, nodetype, content) :
        if nodetype == 'or' :
            return self.addOrNode(content)
        elif nodetype == 'and' :
            return self.addAndNode(content)
        else :
            raise Exception("Unknown node type '%s'" % nodetype)
        
    def addOrNode(self, content) :
        """Add an OR node."""
        return self._addCompoundNode('or', content, self.TRUE, self.FALSE)
        
    def addAndNode(self, content) :
        """Add an AND node."""
        return self._addCompoundNode('and', content, self.FALSE, self.TRUE)
        
    def _addCompoundNode(self, nodetype, content, t, f) :
        assert( content )   # Content should not be empty
        
        # If there is a t node, (true for OR, false for AND)
        if t in content : return self.TRUE
        
        # Eliminate unneeded node nodes (false for OR, true for AND)
        content = filter( lambda x : x != f, content )

        # Put into fixed order
        content = tuple(sorted(content))
        
        # Empty OR node fails, AND node is true
        if not content : return f
                
        # Contains opposites: return 'TRUE' for or, 'FALSE' for and
        if len(set(content)) > len(set(map(abs,content))) : return t
            
        # If node has only one child, just return the child.
        if len(content) == 1 : return content[0]
        
        # Lookup node for reuse
        key = (nodetype, content)
        node_id = self.__nodes_by_content.get(key, None)
        
            
        if node_id == None :    
            # Node doesn't exist yet
            node_id = self._addNode( *key )
            self.__nodes_by_content[ key ] = node_id

            facts = set([])
            disjoint_facts = True
            for child in content :
                child_facts = self._getUsedFacts(child)
                if facts & child_facts : disjoint_facts = False
                facts |= child_facts
            self._setUsedFacts(node_id, facts)
        
            if disjoint_facts :
                if nodetype == 'or' :
                    f = lambda a, b : a*(1-b)
                    p = 1
                elif nodetype == 'and' :
                    f = lambda a, b : a*b
                    p = 1
                for child in content :
                    p_c = self.getProbability(child)
                    if p_c == None :
                        p = None
                        break
                    else :
                        p = f(p,p_c)
                if p and nodetype == 'or' :
                    p = 1 - p
                self.setProbability(node_id, p)

        return node_id
        
    def _addNode(self, nodetype, content) :
        node_id = len(self) + 1
        self.__nodes.append( (nodetype, content) )
        self.__probabilities.append(None)
        self.__usedfacts.append(frozenset([]))
        return node_id
        
    def getNode(self, index) :
        if index <= self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset-1]
    
    def _calculateProbability(self, nodetype, content) :
        if nodetype == 'or' :
            f = lambda a, b : a+b
            p = 0
        elif nodetype == 'and' :
            f = lambda a, b : a*b
            p = 1
            for child in content :
                p_c = self.getProbability(child)
                if p_c == None :
                    p = None
                    break
                else :
                    p = f(p,p_c)
        return p
        
    def getProbability(self, index) :
        if index == 0 :
            return 1
        elif index == None :
            return 0
        elif index < 0 :
            p = self.getProbability(-index)
            if p == None :
                return None
            else :
                return 1 - p
        # elif self.getNode(index)[0] == 'choice' :
        #     return self.getNode(index)[1].probability
        else :
            return self.__probabilities[index-1]
    
    def setProbability(self, index, p) :
        #print ('SP', index, p, self.getNode(index))
        if index == 0 or index == None :
            pass
        elif index < 0 :
            self.__probabilities[-index-1] = 1 - p
        else :
            self.__probabilities[index-1] = p
        
    def integrate(self, lines, rules) :
        # Dictionary query_name => node_id
        result = {}
        
        ln_to_ni = ['?'] * (len(lines) + 1)   # line number to node id
        line_num = 0
        for line_type, line_content, line_alias in lines[1:] :
            line_num += 1
            node_id = self._integrate_line(line_num, line_type, line_content, line_alias, lines, ln_to_ni, rules)
            if node_id != None :
                result[line_alias] = node_id
        return result
        
    def _integrate_line(self, line_num, line_type, line_content, line_alias, lines, ln_to_ni, rules) :
        # TODO make it work for cycles
        
        if line_num != None :
            node_id = ln_to_ni[line_num]
            if node_id != '?' : return node_id
        
        if line_type == 'fact' :
            if re.match('ad\d+_query_set_', line_alias) :
                query = line_alias[len(re.match('ad\d+_', line_alias).group()):]
                node_id = self.addChoice(rules[query])
            elif line_alias.startswith('pf_') :
                node_id = str(line_content)[2:]
                if node_id[0] == '9' :
                    node_id = -int(node_id[1:-1])
                else :
                    node_id = int(node_id[1:-1])
            else :
                node_id = self.addFact(line_alias, line_content)
        else :
            # Compound node => process content recursively
            subnodes = []
            for subnode in line_content :
                if type(subnode) == tuple :
                    subnodes.append(self._integrate_line(None, subnode[0], subnode[1], None, lines, ln_to_ni, rules))
                else :
                    subnode_id = int(subnode)
                    neg = subnode_id < 0
                    subnode_id = abs(subnode_id)
                    subnode = lines[subnode_id]
                    tr_subnode = self._integrate_line(subnode_id, subnode[0], subnode[1], subnode[2], lines, ln_to_ni, rules)
                    if neg :
                        tr_subnode = self._negate(tr_subnode)
                    subnodes.append(tr_subnode)
                    
        if line_type == 'or' :
            node_id = self.addOrNode(tuple(subnodes))    
        elif line_type == 'and' :
            node_id = self.addAndNode(tuple(subnodes))    
            
        # Store in translation table
        if line_num != None : ln_to_ni[line_num] = node_id
        
        return node_id
        
    def _selectNodes(self, queries, node_selection) :
        for q in queries :
            node_id = q
            if node_id :
                self._selectNode(abs(node_id), node_selection)
        
    def _selectNode(self, node_id, node_selection) :
        assert(node_id != 0)
        if not node_selection[node_id-1] :
            node_selection[node_id-1] = True
            nodetype, content = self.getNode(node_id)
            
            if nodetype in ('and','or') :
                for subnode in content :
                    if subnode :
                        self._selectNode(abs(subnode), node_selection)
        
    def __len__(self) :
        return len(self.__nodes) + self.__offset
        
    def toCNF(self, queries=None) :
        # if self.hasCycle :
        #     raise NotImplementedError('The dependency graph contains a cycle!')
        
        if queries != None :
            node_selection = [False] * len(self)    # selection table
            self._selectNodes(queries, node_selection)
        else :
            node_selection = [True] * len(self)    # selection table
            
        lines = []
        facts = {}
        for k, sel in enumerate( node_selection ) :
          if sel :
            k += 1
            v = self.getNode(k)
            nodetype, content = v
            
            if nodetype == 'fact' :
                facts[k] = content[1]
            elif nodetype == 'and' :
                line = str(k) + ' ' + ' '.join( map( lambda x : str(-(x)), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (-k, x) )
            elif nodetype == 'or' :
                line = str(-k) + ' ' + ' '.join( map( lambda x : str(x), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (k, -x) )
                # lines.append('')
            elif nodetype == 'choice' :
                if content.hasScore() :
                    facts[k] = content.probability
                else :
                    facts[k] = 1.0
            else :
                raise ValueError("Unknown node type!")
                
        atom_count = len(self)
        clause_count = len(lines)
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts
        
    def stats(self) :
        return namedtuple('IndexStats', ('atom_count', 'name_count', 'fact_count' ) )(len(self), 0, len(self.__fact_names))
        
    def __str__(self) :
        return '\n'.join('%s: %s' % (i+1,n) for i, n in enumerate(self.__nodes))   
         