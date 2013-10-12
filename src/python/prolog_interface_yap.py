from prolog_interface import PrologInterface, Grounding
from collections import defaultdict, namedtuple
from language import Literal

import sys
import os

class PrologEngine(object) :
    """Expected interface for PrologEngine."""
    
    def __init__(self) :
        self.__sources = []
        self.__clauses = []
        self.__grounding = Grounding()
        
    def loadFile(self, filename) :
        """Load a Prolog source file."""
        self.__sources.append(os.path.abspath(filename))

    def listing(self) :
        """Return a string listing the content of the current Prolog database."""
        #raise NotImplementedError('Calling an abstract method!')
        return 'LISTING NOT SUPPORTED'
        
    def addClause(self, clause) :
        """Add a clause to the database."""
        self.__clauses.append(str(clause) + '.')
        
    def query(self, literal, variables) :
        """Execute a query."""
        
        program_file = '/tmp/probfoil.pl'
        with open(program_file, 'w') as f :
            for sourcefile in self.__sources :
                with open(sourcefile) as fsrc :
                    self._write_file(fsrc, f, True)
            
            writes = ", write('|'), ".join( ('write(%s)' % v ) for v in variables )
        
            f.write( 'write_all :- %s, %s, nl, fail.\n' % (literal, writes) )
            f.write( 'write_all. \n')
            f.write( ':- write_all.')
        
        import subprocess as sp
        result = sp.check_output( ['yap', '-L', program_file ]).decode("utf-8").split('\n')[:-1]
        return map(lambda s : s.split('|'), result)
        
    def _write_file(self, file_in, file_out, deterministic=False) :
        if deterministic :
            import re
            regex = re.compile('\d([.]\d)?\s*::')
        
        for line in file_in :
            if deterministic : 
                line = regex.sub('', line)
                line = line.replace('<-', ':-')
            file_out.write(line)
        
    def groundQuery(self, *literals) :
        """Ground a query."""
        
        # 1) Write out input file for grounder
        #   - source file
        #   - (relevant) content of grounder
        #   - query( ... )
        
        #   How to get content of grounder?
        #       => For each name defined in the grounder: write out fact 0.5::name.
        
        lines = []
#        lines += [ ('0.5::%s.' % name) for name in self.grounding.names ]
        
        pl_filename = '/tmp/probfoil.pl'
        with open(pl_filename, 'w') as pl_file : 
            for sourcefile in self.__sources :
                with open(sourcefile) as in_file :
                    pl_file.write( in_file.read() )
            
            print ('\n'.join(lines), file=pl_file)          # Previous groundings
            print ('\n'.join(self.__clauses), file=pl_file) # Stored clauses => only use new ones!!!!!
            
            for literal in literals :
                print ( 'query(%s).' % literal, file=pl_file ) # Query
        
        # 2) Call grounder in Yap
        grounder_result = self._call_grounder( pl_filename)
        
        # 3) Read in grounding and insert new things into Grounding data structure
        self.getGrounding().integrate(grounder_result)
        
        return [ self.getGrounding().byName(lit) for lit in literals ]
    
    def _call_grounder(self, in_file) :
        PROBLOG_GROUNDER=os.environ['PROBLOGPATH'] + '/assist/ground_compact.pl'
                
        # 2) Call yap to do the actual grounding
        ground_program = '/tmp/probfoil.ground'
        evidence = '/dev/null'
        queries = '/dev/null'
        
        import core
        core.call_yap(PROBLOG_GROUNDER, args=[in_file, ground_program, evidence, queries])
        
        return read_grounding(ground_program)
    
    def getFactProbability(self, literal) :
        """Retrieve the fact probability."""
        
        self.groundQuery(literal)
        
        index = self.getGrounding().byName(str(literal))
        if index == None :
            return 0
        elif index == 0 :
            return 1
        else :
            return self.getGrounding().getProbability(str(literal))
    
    def getGrounding(self) :
        """Get grounding information."""
        return self.__grounding

def read_grounding(filename) :
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
    
    def __init__(self) :
        super().__init__()
        
    def query(self, query, variables) :
        return self.engine.query(query, variables)
        
    def _createPrologEngine(self) :
        return PrologEngine()
         
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
