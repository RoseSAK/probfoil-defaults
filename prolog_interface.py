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
        self.last_id = 0
        self.toGround = []
        self.toScore = []
        self.toEvaluate = set([])
        
        self.grounding = Grounding()
        self.data = []
                    
    def enqueue(self, rule) :
        # Put the rule in queue for evaluation.
        # Already computes as much grounding and evaluation as it can.
        
        with Timer(category="enqueue") :
            if not rule.literal : 
                # Without literal there is nothing to do, except initialize data structures.
                if not rule.eval_nodes :
                    rule.eval_nodes = [None] * len(rule.examples)
                    rule.self_nodes = [None] * len(rule.examples)
                    rule.score_predict = [0] * len(rule.examples)
                    self.toGround.append((rule,[ x for x,y in rule.enum_examples() ]))
                    self.toScore.append((rule,[ x for x,y in rule.enum_examples() ]))
            else :
                # Initialize eval_nodes and score_predict structures in rule
                rule.eval_nodes = [None] * len(rule.examples)
                rule.self_nodes = [None] * len(rule.examples)
                rule.score_predict = [0] * len(rule.examples)
        
                # Set up toGround, toScore and toEvaluate queues
                toGround = []
                toScore = []
                toEvaluate = set([])
        
                success = True
                # Check whether we can do some simple pre-grounding
                if rule.literal.arguments == rule.target.arguments :
                    # Example fully defines literal instance, check whether we grounded it before.
                    for ex_id, example in rule.enum_examples() :
                        fact_id = self.grounding.getFact( str( Literal( rule.literal.functor, example) ) )
                        if fact_id == None :
                            # Fact wasn't found, we'll still need to do normal grounding
                            toGround.append( ex_id )
                            #toScore.append( ex_id )
                        else :
                            # Fact was found! We can do a bit of grounding here.
                            if rule.literal.is_negated : fact_id = -fact_id
                            
                            # Get the node id.
                            parent_node = rule.parent.self_nodes[ex_id]
                            if parent_node != None and parent_node > 0 :
                                p_type, p_content = self.grounding.getNode(parent_node)
                                if p_type == 'and' :
                                    node_id = self.grounding.addAndNode( p_content + (fact_id,) )
                                else :
                                    node_id = self.grounding.addAndNode( (parent_node, fact_id ) )
                            else :
                                node_id = self.grounding.addAndNode( (parent_node, fact_id ) )
                            
                            # Store the self_node (i.e. the node used to evaluate just this rule)
                            rule.self_nodes[ex_id] = node_id
                            
                            if rule.previous and rule.previous.eval_nodes :   # Previous rule has nodes                
                                prev_node = rule.previous.eval_nodes[ ex_id ]
                                if node_id == None :    # Rule fail
                                    new_node = prev_node    # => new theory predicts same as old theory
                                elif node_id == 0 :     # Rule true 
                                    new_node = 0
                                else :
                                    new_node = self.grounding.addOrNode( ( prev_node, node_id ) ) 
                                    
                                    if self.grounding.getProbability(new_node) == None :
                                        success = False
                                    
                            else :
                                new_node = node_id
                        
                            rule.eval_nodes[ex_id] = new_node
                    
                            # And get the probability.
                            p = self.grounding.getProbability(new_node)
                            if p == None :
                                # Calculation failed: needs advanced evaluation
                                self.toEvaluate.add( new_node )
                                toScore.append( ex_id )
                            else :
                                # Probability available: store it, this rule+ex_id has been completely evaluated
                                rule.score_predict[ex_id] = p
                    if not success and self.env['verbose'] > 3 :
                         print ('Requires evaluation:', '\t'.join(rule.getTheory()))
                else :
                    # We can't do any grounding yet
                    toGround = [ x for x,y in rule.enum_examples() ]
                    toScore = [ x for x,y in rule.enum_examples() ]
                    
                # Add information to global queues, but only if any action is needed.    
                if toGround : self.toGround.append( (rule, toGround) )
                
                if toScore : self.toScore.append( (rule, toScore) )
                self.toEvaluate |= toEvaluate
            
    def process_queue(self) :
        with Timer(category="grounding") :
            with Timer(category="grounding_generateprogram") :
                gp = []
                # Create ground program.
                for rule_id, rule_exids in enumerate(self.toGround) :
                    rule, ex_ids = rule_exids
            
                    if not rule.parent :
                        # We are evaluating the target => everything we need is in the data.
                        clause_pred =  str(rule.target.functor)
                    else :
                        # We are evaluating a candidate rule => add it to the ground program.
                        clause_pred =  'pf_rule_%s' % rule_id
                        clause = str(Literal( clause_pred, rule.target.arguments))
                        clause += ':-' + ','.join(map(str,rule.literals) )
                        clause += '.'
                        gp.append( clause )
            
                    # Add queries to ground program.
                    for ex_id in ex_ids :
                        query = Literal('pf_query_%s_%s' % (rule_id, ex_id), [])
                        real_query = Literal( clause_pred, rule.examples[ex_id] )
                        gp.append( '%s :- %s.' % (query, real_query) )
                        gp.append( 'query(%s).' % (query, ) )

            # Call grounder
            ground_result = []
            names_nodes = self._ground(gp)            
            for name, value in names_nodes.items() :
                if name.startswith('pf_query_') :
                    rule, ex_id = name.split('_')[2:]
                    ground_result.append( ( self.toGround[int(rule)][0], int(ex_id), int(value) ))

            with Timer(category="grounding_process") :
                # Process the grounding
                for rule, ex_id, node_id in ground_result :
                    # Add nodes for previous rules.
                    if rule.previous and rule.previous.eval_nodes :   # Previous rule has nodes
                        prev_node = rule.previous.eval_nodes[ ex_id ]
                        if node_id == None :    # Rule fail
                            new_node = prev_node    # => new theory predicts same as old theory
                        elif node_id == 0 :     # Rule true 
                            new_node = 0            # => new theory predicts true
                        else :
                            new_node = self.grounding.addOrNode( ( prev_node, node_id ) ) 
                    else :
                        new_node = node_id
            
                    # Store node information in rule and determine which nodes still need to be evaluated
                    rule.self_nodes[ex_id] = node_id
                    rule.eval_nodes[ex_id] = new_node
                    p = self.grounding.getProbability(new_node)
                    if p == None :
                        # Calculation failed: needs advanced evaluation
                        self.toEvaluate.add( new_node )
                        self.toScore.append( (rule, (ex_id,)) )
                    else :
                        # Probability available: store it, this rule/example combination has been completely evaluated
                        rule.score_predict[ex_id] = p
        
        with Timer(category="evaluate") :       
            # Evaluate nodes in toEvaluate queue
            if self.toEvaluate : 
                # Convert grounding to CNF
                with Timer(category='evaluate_converting') :
                    cnf, facts = self.grounding.toCNF( self.toEvaluate )
    
                # Compile the CNF
                evaluator = self._compile_cnf(cnf, facts)
    
                for node_id in self.toEvaluate :            
                    with Timer(category='evaluate_evaluating') :
                        p = evaluator.evaluate(node_id)
                        self.grounding.setProbability(node_id, p)
                
            # Score
            for rule, ex_ids in self.toScore :
                for ex_id in ex_ids :
                    node_id = rule.eval_nodes[ex_id]
                    p = self.grounding.getProbability(node_id)
                    rule.score_predict[ex_id] = p
    
        # Clear queues
        self.toEvaluate = set([])        
        self.toGround = []
        self.toScore = []

    def _ground(self, program) :
        if program :
            with Timer(category='grounding_writing') as tmr : 
                pl_filename = self.env.tmp_path('probfoil.pl')
                with open(pl_filename, 'w') as pl_file : 
                    print ('\n'.join(self.data), file=pl_file)
                    print ('\n'.join(program), file=pl_file)
                
            with Timer(category='grounding_grounding') as tmr : 
                # 2) Call grounder in Yap
                grounder_result = self._call_grounder( pl_filename)
    
            with Timer(category='grounding_integrating') as tmr : 
                # 3) Read in grounding and insert new things into Grounding data structure
                return self.grounding.integrate(grounder_result)
        else :
            return {}
            
    def _call_grounder(self, in_file) :
        PROBLOG_GROUNDER=self.env['PROBLOGPATH'] + '/assist/ground_compact.pl'
                
        # 2) Call yap to do the actual grounding
        ground_program = self.env.tmp_path('probfoil.ground')
        # Remove output file
        if os.path.exists(ground_program) : os.remove(ground_program)
        
        evidence = '/dev/null'
        queries = '/dev/null'
        
        if sys.path[-1] != self.env['PROBLOGPATH'] + '/src/' :
            sys.path.append(self.env['PROBLOGPATH'] + '/src/')
        
        import subprocess
        self.env['PROBLOGPATH'] + '/assist/linux_x86_64/dsharp'
        output = subprocess.check_output(['yap', "-L", PROBLOG_GROUNDER , '--', in_file, ground_program, evidence, queries ])
                
        return self._read_grounding(ground_program)
    
    def query(self, literal, variables) :
        """Execute a query."""
        
        program_file = self.env.tmp_path('probfoil.pl')
        
        import re
        regex = re.compile('\d([.]\d+)?\s*::')
        
        def make_det(line) :
            line = regex.sub('', line)
            line = line.replace('<-', ':-')
            return line
        
        with open(program_file, 'w') as f :
            print( '\n'.join(map(make_det, self.data)), file=f)  
            
            writes = ", write('|'), ".join( ('write(%s)' % v ) for v in variables )
            f.write( '\n')
            f.write( 'write_all :- %s, %s, nl, fail.\n' % (literal, writes) )
            f.write( 'write_all. \n')
            f.write( ':- write_all.')
        
        import subprocess as sp
        result = sp.check_output( ['yap', '-L', program_file ]).decode("utf-8").split('\n')[:-1]
        return map(lambda s : s.split('|'), result)
            
    def _read_grounding(self, filename) :
        lines = []
        with open(filename,'r') as f :
            for line in f :
                line = line.strip().split('|', 1)
                name = None
            
                if len(line) > 1 : name = line[1].strip()
                line = line[0].split()
                line_id = int(line[0])
                line_type = line[1].lower()
            
                while line_id >= len(lines) :
                    lines.append( (None,[],None) )
                if line_type == 'fact' :
                    line_content = float(line[2])
                else :
                    line_content = lines[line_id][1] + [(line_type, line[2:])]
                    line_type = 'or'
                
                lines[line_id] = ( line_type, line_content, name )
        return lines
            
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
                 
            with Timer('Compiling %s' % cnf[0], verbose=self.env['verbose']>1) :
                executable = self.env['PROBLOGPATH'] + '/assist/linux_x86_64/dsharp'
                subprocess.check_output([executable, "-Fnnf", nnf_file , "-disableAllLits", cnf_file])
        
            if sys.path[-1] != self.env['PROBLOGPATH'] + '/src/' :
                sys.path.append(self.env['PROBLOGPATH'] + '/src/')
            # Evaluate DDNNF
            from compilation.compile import DDNNFFile
            from evaluation.evaluate import FileOptimizedEvaluator
        
          with Timer(category='evaluate_evaluating') :
            ddnnf = DDNNFFile(nnf_file, None)
            ddnnf.atoms = lambda : list(range(1,len(self.grounding)+1))   # OMFG what a hack
            
            return self._construct_evaluator(ddnnf, facts)
    
    def _rewrite_facts(self, facts) :
        return facts
    
    def _get_examples_for_queue(self, rule) :
        return rule.enum_examples()
            
    def _construct_evaluator(self, ddnnf, facts) :
        return DefaultEvaluator(ddnnf, self._rewrite_facts(facts))
        
    def loadData(self, filename) :
        
        with open(filename) as datafile :
            for line in datafile :
                line = line.strip()
                if line and not line[0] in '%' :
                    self.data.append(line)

        
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
        
    def getFact(self, name) :
        return self.__fact_names.get(name, None)
        
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
        node_id = self.getFact(name)
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
            #self.__nodes_by_content[ key ] = node_id
            
            facts = set([])
            disjoint_facts = True
            cf = []
            for child in content :
                child_facts = self._getUsedFacts(child)
                if facts & child_facts : disjoint_facts = False
                facts |= child_facts
                cf.append(child_facts)
            self._setUsedFacts(node_id, facts)
            
            if disjoint_facts :
                p = self.calculateProbability(nodetype, content)
                self.setProbability(node_id, p)
                
        return node_id
        
    def _addNode(self, nodetype, content) :
        node_id = len(self) + 1
        self.__nodes.append( (nodetype, content) )
        self.__probabilities.append(None)
        self.__usedfacts.append(frozenset([]))
        return node_id
        
    def getNode(self, index) :
        assert (index != None and index > 0)
        if index <= self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset-1]
    
    def calculateProbability(self, nodetype, content) :
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
        if p != None and nodetype == 'or' :
            p = 1 - p
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
        
    def integrate(self, lines, rules=None) :
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
         