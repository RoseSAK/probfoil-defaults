#! /usr/bin/env python3

from __future__ import print_function

from collections import namedtuple, defaultdict
from util import Timer, Log

import re, os

def is_var(arg) :
    return arg == None or arg[0] == '_' or (arg[0] >= 'A' and arg[0] <= 'Z')

def strip_negation(name) :
    if name.startswith('\+') :
        return name[2:].strip()
    else :
        return name

def true(f) :
    for x in f :
        return True
    return False

class Literal(object) :
    
    def __init__(self, kb, functor, args) :
        self.functor = functor
        self.args = args
        self.kb = kb
    
    identifier = property(lambda s : (strip_negation(s.functor), len(s.args) ) )
                
    def __str__(self) :
        if self.args :
            return '%s(%s)' % (self.functor, ','.join(self.args))
        else :
            return '%s' % (self.functor)

    def __repr__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)
        
    def __ne__(self, other) :
        return str(self) != str(other)
        
    def __lt__(self, other) :
        return str(self) < str(other)
        
    def __cmp__(self, other) :
        if self.isNegated() and not other.isNegated() :
            return -1
        elif other.isNegated() and not other.isNegated() :
            return 1
        else :
            return 0
        
    def isNegated(self) :
        return self.functor.startswith('\+')
        
    def unify(self, ground_args) :
        
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if is_var(ARG) and not ARG == '_' :
                result[ARG] = arg
        return result  
        
    def assign(self, subst) :
        return Literal(self.kb, self.functor, [ subst.get(arg, arg) for arg in self.args ])
        
    def variables(self) :
        result = defaultdict(set)
        types = self.kb.argtypes(strip_negation(self.functor), len(self.args))
        for tp, arg in zip(types, self.args) :
            if is_var(arg) and not arg == '_' :
                result[tp].add(arg)
        return result
        
    @classmethod
    def parse(cls, kb, string) :
        regex = re.compile('\s*(?P<name>[^\(]+)\((?P<args>[^\)]+)\)[,.]?\s*')
        result = []
        for name, args in regex.findall(string) :
            result.append( Literal( kb, name , map(lambda s : s.strip(), args.split(','))) )
        return result
        
class FactDB(object) :
    
    def __init__(self, idtypes=[]) :        
        self.predicates = {}
        self.idtypes = idtypes
        self.__examples = None
        self.types = defaultdict(set)
        self.modes = {}
        self.learn = []
        self.newvar_count = 0
        self.probabilistic = set([])
        self.prolog_file = None
        
    def newVar(self) :
        self.newvar_count += 1
        return 'X_' + str(self.newvar_count)
            
    def register_predicate(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(set) for i in range(0,arity) ] 
            self.predicates[identifier] = (index, [], args, [])
            
    def is_probabilistic(self, literal) :
        return literal.identifier in self.probabilistic
            
    def add_mode(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        self.modes[identifier] = args
        
    def add_learn(self, name, args) :
        self.learn.append( (name, args) )
        
    def add_fact(self, name, args, p=1.0) :
        arity = len(args)
       # self.register_predicate(name, arity)
        identifier = (name, arity)
        index, values, types, probs = self.predicates[identifier]
        
        if p < 1.0 :
            self.probabilistic.add(identifier)
        
        arg_index = len(values)
        values.append( args )
        probs.append( p )

        for i, arg in enumerate(args) :
            self.types[types[i]].add(arg)
            index[i][arg].add(arg_index)
        #index[-1][tuple(arg)].add(arg_index)
            
    def argtypes(self, name, arity) :
        identifier = (name, arity)
        return self.predicates[identifier][2]
      
    def ground_fact(self, literal, used_facts=None) :
        
        index, values, types, probs = self.predicates[ literal.identifier ]
        
        # Initial result set = all literals for this predicate
        result = set(range(0, len(values))) 
        
        # Restrict set for each argument
        for i,arg in enumerate(literal.args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break 
        
        result_maybe = set( i for i in result if 0 < probs[i] < 1 )
        result_exact = result - result_maybe
        
        if literal.isNegated() :
            if result_exact : 
                # There are facts that are definitely true
                return []   # Query fails
            else :
                # Query might succeed
                if used_facts != None and result_maybe :
                    # Add maybe facts to used_facts
                    used_facts[ literal.identifier ] |= result_maybe
                if result_maybe :
                    is_det = False
                else :
                    is_det = True
                return [ (is_det, literal.args) ]
        else :
            if used_facts != None and result_maybe :
                # Add maybe facts to used_facts
                used_facts[ literal.identifier ] |= result_maybe
            return [ (probs[i] == 1, values[i]) for i in result_maybe | result_exact ]
        
    def find_fact(self, name, args) :
        if name.startswith('\+') :
           # raise NotImplementedError("No support for negative literals yet!")
            negated = True
            name = strip_negation(name)
        else :
            negated = False

        arity = len(args)
        identifier = (name, arity)
        
        index, values, types, probs = self.predicates[identifier]
        
        result = set(range(0, len(values))) 
        for i,arg in enumerate(args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break        
        
        probability = sum( probs[i] for i in result )        
        if negated :
            return 1.0 - probability
        else :
            return probability

    def __str__(self) :
        s = ''
        for name, arity in self.predicates :
            for args in self.predicates[(name,arity)][1] :
                s += '%s(%s)' % (name, ','.join(args)) + '.\n'
        return s
          
    def reset_examples(self) :          
        self.__examples = None
        
    def _get_examples(self) :
        if self.__examples == None :
            from itertools import product
            self.__examples = list(product(*[ self.types[t] for t in self.idtypes ]))
        return self.__examples
    
    examples = property(_get_examples)
        
    def __getitem__(self, index) :
        return self.examples[index]
        
    def __len__(self) :
        return len(self.examples)
        
    def __iter__(self) :
        return iter(range(0,len(self.examples)))
        
    def clause(self, head, body, args) :
        subst = head.unify( args )
        for sol in self.query(body, subst) :
            yield sol

    def query_single(self, literals, substitution) :
        return true(self.query(literals, substitution))

    def query(self, literals, substitution, facts_used=None) :
        if not literals :  # reached end of query
            yield substitution, []
        else :
            head, tail = literals[0], literals[1:]
            head_ground = head.assign(substitution)     # assign known variables
            
            for is_det, match in self.ground_fact(head_ground, facts_used) :

                # find match for head
                new_substitution = dict(substitution)
                new_substitution.update( head_ground.unify( match ) )
                if is_det :
                    my_lit = []
                else :
                    my_lit = [ head_ground.assign(new_substitution) ]
                for sol, sub_lit in self.query(tail, new_substitution, facts_used) :
                    yield sol, my_lit + sub_lit
    
    def printFacts(self, facts, out) :
        for identifier in facts :
            name, arity = identifier
            index, values, types, probs = self.predicates[identifier]
            
            for i in facts[identifier] :
                args = ','.join(values[i])
                prob = probs[i]
                if prob == 1 :
                    print( '%s(%s).' % (name, args),  file=out)
                else :
                    print( '%s::%s(%s).' % (prob, name, args),  file=out)    
                    
    def toPrologFile(self) :
        if self.prolog_file == None :
            self.prolog_file = '/tmp/pf.pl'
            with open(self.prolog_file, 'w') as f :
                for pred in self.predicates :
                    index, values, types, probs = self.predicates[pred]
                    for args in values :
                        print('%s(%s).' % (pred[0], ', '.join(args) ), file=f)
        return self.prolog_file                

class Rule(object) :
    """Class representing a learned rule.
    
    It has the following parts:
        - head  - the head literal
        - body  - a list of body literals
        
        - score - scoring statistics in its current context (data and hypothesis)
    """
    
    variables = property(lambda s : s._get_varinfo()[0])
    varnames = property(lambda s : s._get_varinfo()[1])

    def __init__(self, head, body, score) :
        self.head = head
        self.body = body
        self.varinfo = None
        self.score = score

    def _get_varinfo(self) :
        if self.varinfo == None :
            self.varinfo = self._extract_variables([self.head] + self.body)
        return self.varinfo
        
    def _ground_extensions(self, kb, examples, extensions) :
        used_facts = defaultdict(set)
        new_clauses = []
        
        scores = [ [] for ext in extensions ]
        queries = []
        
        for ex_id, example in enumerate(examples) :
            literal_scores = self._ground_extensions_example(kb, ex_id, example, extensions, used_facts, new_clauses)
            for lit_i, score in enumerate(literal_scores) :
                if score == None :
                    queries.append(  (lit_i, ex_id, 'pfq_%s_%s' % (ex_id, lit_i)) )
                elif score == 1.0 :  
                    # literal is deterministically true, reuse rule score
                    score = 1.0
#                    score = self.score.covered[ex_id][1]
                scores[lit_i].append(score)     
        
        return scores, used_facts, new_clauses, queries

    def _ground_extensions_example(self, kb, ex_id, example, extensions, used_facts, new_clauses) :
        subst = self.head.unify( example )

        literal_score = [ 0.0 ] * len(extensions)    # default 0
        for body_vars, old_clause in kb.query(self.body, subst, used_facts) :
            for lit_i, lit in enumerate(extensions) :
                head = 'pfq_%s_%s' % (ex_id, lit_i)
                if literal_score[lit_i] == 1 :
                    continue    # literal was already proven to be deterministically true
                for lit_val, lit_new in kb.query([lit],body_vars, used_facts) :
                    if old_clause + lit_new :   # TODO remove old_clause and reuse old score
                        # probabilistic query
                        new_clauses.append( Rule( Literal(kb, head, []), old_clause + lit_new , None ) )
                        literal_score[lit_i] = None  # probabilistic
                    else :
                        # non-probabilistic => p = 1.0
                        literal_score[lit_i] = 1.0    # deterministically true
        return literal_score
        
    def _callProbLog( self, kb, scores, used_facts, clauses, queries ) :
        
        print('Calling problog', len(queries))
        
        outfile = '/tmp/pf.pl'
        with open(outfile, 'w') as out :
            kb.printFacts(used_facts, out)
            print('\n'.join( map(str,clauses)), file=out)            
            for l,e,q in queries :
                print('query(%s).' % q, file=out)
        
        import problog
        engine = problog.ProbLogEngine.create([])
        logger = problog.Logger(False)
        dir_out = '/tmp/pf/'
        with problog.WorkEnv(dir_out,logger, persistent=problog.WorkEnv.ALWAYS_KEEP) as env :
            probs = engine.execute(outfile, env) # TODO call problog
            
        for l,e,q in queries :
            scores[l][e] = probs[q]
            
       # return scores

    def evaluate_extensions(self, kb, examples, extensions) :
        scores, used_facts, clauses, queries = self._ground_extensions(kb, examples, extensions )         
        
        if queries :
            self._callProbLog(kb, scores, used_facts, clauses, queries)
        
        return zip(extensions,scores)
        
    def _extract_variables(self, literals) :
        names = set([])
        result = defaultdict(set)
        for lit in literals :
            lit_vars = lit.variables()
            for t in lit_vars :
                result[t] |= lit_vars[t]
                names |= lit_vars[t]
        return result, names

    def _build_refine_one(self, kb, positive, arg_type, arg_mode, defined_vars) :
        if arg_mode in ['+','-'] :
            for var in self.variables[arg_type] :
                yield var
        if arg_mode == '-' and positive :
            defined_vars.append(True)
            yield kb.newVar()
            defined_vars.pop(-1)
        if arg_mode == 'c' :
            if positive :
                for val in kb.types[arg_type] :
                    yield val
            else :
                yield '_'

    def _build_refine(self, kb, positive, arg_info, defined_vars, use_vars) :
        if arg_info :
            for arg0 in self._build_refine_one(kb, positive, arg_info[0][0], arg_info[0][1], defined_vars) :
                if use_vars != None and arg0 in use_vars :
                    use_vars1 = None
                else :
                    use_vars1 = use_vars 
                for argN in self._build_refine(kb, positive, arg_info[1:], defined_vars, use_vars1) :
                    yield [arg0] + argN
        else :
            if use_vars == None :
                yield []
        
    def refine(self, kb, update=False) :
        if update :
            if not self.body : return
            d, old_vars = self._extract_variables([self.head] + self.body[:-1])
            d, new_vars = self._extract_variables([self.body[-1]])
            use_vars = new_vars - old_vars
            if not use_vars : return
            #use_vars = None
            #print >> sys.stderr, use_vars
        else :
            use_vars = None
        
       # print 'VARS', self.variables        
        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, True, arg_info, [], use_vars) :
                yield Literal(kb, pred_name, args)

        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, False, arg_info, [], use_vars) :
                yield Literal(kb, '\+' + pred_name, args)
            
    def __str__(self) :
        if self.body :
            return str(self.head) + ' :- ' + ', '.join(map(str,self.body)) + '.'
        else :
            return str(self.head) + '.'
        
    def __len__(self) :
        return len(self.body)
        
    def countNegated(self) :
        return sum( b.isNegated() for b in self.body )    

def read_file(filename, idtypes=[]) :

    import re 
    
    line_regex = re.compile( "((?P<type>(base|modes|learn))\()?((?P<prob>\d+[.]\d+)::)?\s*(?P<name>\w+)\((?P<args>[^\)]+)\)\)?." )
    
    kb = FactDB(idtypes)

    with open(filename) as f :
        for line in f :        
            if line.strip().startswith('%') :
                continue
            m = line_regex.match(line.strip())
            if m : 
                ltype, pred, args = m.group('type'), m.group('name'), list(map(lambda s : s.strip(), m.group('args').split(',')))
                prob = m.group('prob')
                
                if ltype == 'base' :
                    kb.register_predicate(pred, args)  
                elif ltype == 'modes' :
                    kb.add_mode(pred,args)                    
                elif ltype == 'learn' :
                    kb.add_learn(pred,args)                
                else :
                    if not prob :
                        prob = 1.0
                    else :
                        prob = float(prob)
                    kb.add_fact(pred,args,prob)
                                
    return kb

def test(args=[]) :

    kb = read_file(args[0])
    from learn import learn, RuleSet

    print ("# loaded dataset '%s'" % (args[0]))
#    print "# ", kb.modes
    
    stop = False
    while not stop :
        try :
            s = raw_input('?- ')
            l = Literal.parse(kb, s)
            print ('\n'.join(map(str,kb.query(l, {}))))
        except EOFError :
            break
        except KeyboardInterrupt :
            break
        
def main(args=[]) :



    for filename in args :

        kb = read_file(filename)

        from learn import learn, Hypothesis, SETTINGS, Score

        varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        targets = kb.learn
    
        filename = os.path.split(filename)[1]
    
    
    
        with open(filename+'.xml', 'w') as Log.LOG_FILE :

            with Log('log') :
                for pred, args in targets :
                  with Timer('learning time') as t :
        
                    kb.idtypes = args
                    kb.reset_examples()
        
                    target = Literal(kb, pred, varnames[:len(args)] )
        
                    print('==> LEARNING CONCEPT:', target)
                    with Log('learn', input=filename, target=target, **vars(SETTINGS)) :
                        H = learn(Hypothesis(Rule, Score, target, kb))   
    
                        print(H)
                        print(H.score, H.score.globalScore)
    
                        with Log('result', time=t.elapsed_time) as l :
                            if l.file :
                                print(H, file=l.file)
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    