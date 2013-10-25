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


from prolog_interface import PrologInterface, Grounding
from collections import defaultdict, namedtuple
from language import Literal
from util import Timer
import sys
import os

# TODO randomize temporary filenames

class PrologEngine(object) :
    
    def __init__(self, env) :
        self.__sources = []
        self.__fixed_clauses = []
        self.__clauses = []
        self.__grounding = Grounding()
        self.__env = env
        
    def loadFile(self, filename) :
        """Load a Prolog source file."""
        self.__sources.append(os.path.abspath(filename))

    def listing(self) :
        """Return a string listing the content of the current Prolog database."""
        #raise NotImplementedError('Calling an abstract method!')
        return 'LISTING NOT SUPPORTED'
        
    def addFixedClause(self, clause) :
        """Add a clause to engine."""
        self.__fixed_clauses.append(str(clause) + '.')
        
    def addClause(self, clause) :
        """Add a clause to engine."""
        self.__clauses.append(str(clause) + '.')
        
    def clearClauses(self) :
        self.__clauses = []
        
    def query(self, literal, variables) :
        """Execute a query."""
        
        program_file = self.__env.tmp_path('probfoil.pl')
        with open(program_file, 'w') as f :
            for sourcefile in self.__sources :
                with open(sourcefile) as fsrc :
                    self._write_file(fsrc, f, True)
            
            writes = ", write('|'), ".join( ('write(%s)' % v ) for v in variables )
            f.write( '\n')
            f.write( 'write_all :- %s, %s, nl, fail.\n' % (literal, writes) )
            f.write( 'write_all. \n')
            f.write( ':- write_all.')
        
        import subprocess as sp
        result = sp.check_output( ['yap', '-L', program_file ]).decode("utf-8").split('\n')[:-1]
        return map(lambda s : s.split('|'), result)
        
    def _write_file(self, file_in, file_out, deterministic=False) :
        if deterministic :
            import re
            regex = re.compile('\d([.]\d+)?\s*::')
        
        for line in file_in :
            if deterministic : 
                line = regex.sub('', line)
                line = line.replace('<-', ':-')
            file_out.write(line)
        
    def groundQuery(self, literals, rules) :
        """Ground a query."""
        
        # 1) Write out input file for grounder
        #   - source file
        #   - (relevant) content of grounder
        #   - query( ... )
        
        #   How to get content of grounder?
        #       => For each name defined in the grounder: write out fact 0.5::name.
        
        lines = []
#        lines += [ ('0.5::%s.' % name) for name in self.grounding.names ]
        
        with Timer(category='evaluate_grounding_writing') as tmr : 
            pl_filename = self.__env.tmp_path('probfoil.pl')
            with open(pl_filename, 'w') as pl_file : 
                for sourcefile in self.__sources :
                    with open(sourcefile) as in_file :
                        pl_file.write( in_file.read() )
            
                print ('\n'.join(lines), file=pl_file)          # Previous groundings
                print ('\n'.join(self.__clauses), file=pl_file) # Stored clauses => only use new ones!!!!!
                print ('\n'.join(self.__fixed_clauses), file=pl_file) # Stored clauses => only use new ones!!!!!
            
                # print ('\n'.join(self.__fixed_clauses)) # Stored clauses => only use new ones!!!!!
                # print (' ==== ')
                # print ('\n'.join(self.__clauses)) # Stored clauses => only use new ones!!!!!
            
                for literal in literals :
                    print ( 'query(%s).' % literal, file=pl_file ) # Query

        with Timer(category='evaluate_grounding_grounding') as tmr : 
            # 2) Call grounder in Yap
            grounder_result = self._call_grounder( pl_filename)
        
        with Timer(category='evaluate_grounding_integrating') as tmr : 
            # 3) Read in grounding and insert new things into Grounding data structure
            names_nodes = self.getGrounding().integrate(grounder_result, rules)
        
        return [ names_nodes.get(lit,None) for lit in literals ]
    
    def _call_grounder(self, in_file) :
        PROBLOG_GROUNDER=self.__env['PROBLOGPATH'] + '/assist/ground_compact.pl'
                
        # 2) Call yap to do the actual grounding
        ground_program = self.__env.tmp_path('probfoil.ground')
        # Remove output file
        if os.path.exists(ground_program) : os.remove(ground_program)
        
        evidence = '/dev/null'
        queries = '/dev/null'
        
        if sys.path[-1] != self.__env['PROBLOGPATH'] + '/src/' :
            sys.path.append(self.__env['PROBLOGPATH'] + '/src/')
        
        import subprocess
        self.__env['PROBLOGPATH'] + '/assist/linux_x86_64/dsharp'
        output = subprocess.check_output(['yap', "-L", PROBLOG_GROUNDER , '--', in_file, ground_program, evidence, queries ])
                
        return self._read_grounding(ground_program)
        
    def getGrounding(self) :
        """Get grounding information."""
        return self.__grounding

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

class YapPrologInterface(PrologInterface) :
    
    def __init__(self, env) :
        super().__init__(env)
        
    def query(self, query, variables) :
        return self.engine.query(query, variables)
        
    def _createPrologEngine(self) :
        return PrologEngine(self.env)
         
    def _toPrologClause(self, head, *body, probability=1) :
        head = str(head)
        body = ','.join(map(str,body))
        
        if probability != 1 :
            operator = '<-'
            prob_str = str(probability) + '::'
        else :
            operator = ':-'
            prob_str = ''
        
        if not body : operator = ''
            
        return prob_str + head + operator + body
                
    def _toProlog(self, term) :
        return str(term)

class PropPrologInterface(YapPrologInterface) :
    
    def __init__(self, env) :
        super().__init__(env)
        self.__facts = defaultdict(dict)
        
    def _addFact(self, fact_name, ex_id, prob) :
        fact_name = '%s(var_1)' % fact_name
        c = len(self.engine.getGrounding())
        fact_id = self.engine.getGrounding().addFact(fact_name, 0.5)
        if ex_id == 0 :
            # New fact name
            self.engine.addFixedClause('0.5::%s' % fact_name)
        
        self.__facts[ex_id][fact_id] = prob
   
    def query(self, query, variables) :
        if query.functor == 'base' :
            return [ ('id',) ]
        else :
            return [ (x,) for x in range(0, len(self.__facts )) ]
   
    def loadData(self, filename) :
        with open(filename) as f :
            ex_id = 0
            for line in f :
                line = line.strip()

                if line and not line.startswith('@') and not line.startswith('#') :
                    line = list(map(float,line.split(',')))
                    for att, val in enumerate(line) :
                        self._addFact( 'att%s' % att, ex_id, val )
                    ex_id += 1
        #self.engine.loadFile(filename)
        
    def _rewrite_facts(self, facts) :
        return facts
    
    def _get_examples_for_queue(self, rule) :
        return [ (None, ('var_1',) ) ]
            
    def _construct_evaluator(self, ddnnf, facts) :
        return PropositionalEvaluator(ddnnf, facts, self.__facts)
    
class PropositionalEvaluator(object) :
    
    def __init__(self, knowledge, facts, base_facts) :
        self.__knowledge = knowledge
        self.__facts = facts
        self.__base_facts = base_facts
                
    def evaluate(self, node_id, rule=None, ex_id=None) :
        if ex_id == None : return 0
        facts = self.__base_facts[ex_id]
        if node_id in facts :
            return facts[node_id]
        else :
            from evaluation.evaluate import FileOptimizedEvaluator
            base = FileOptimizedEvaluator()
            result = base(knowledge=self.__knowledge, probabilities=facts, queries=None, env=None)[1]
            return result[node_id]
    
    def example_dependent(self) :
        return True


def test1(*files) :
    g = Grounding()
    
    for filename in files :
        nodes = read_grounding(filename)
        update = g.integrate( nodes )
        print('>>>>>>> PROCESSED', filename, '<<<<<<<', update)
        print(g)

def test2(filename, *queries) :
    p = PrologEngine()
    
    p.loadFile(filename)
    
    for q in queries :
        p.groundQuery(q)
        print('>>>> GROUNDED QUERY', q)
        print(p.getGrounding())

if __name__ == '__main__' :
    test2(*sys.argv[1:])
