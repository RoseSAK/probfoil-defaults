#! /usr/bin/env python3

from __future__ import print_function

from collections import namedtuple, defaultdict
from util import Timer, Log

import re, os
import time

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
    
def test_fail(clause) :
    for i, literal in enumerate(clause) :
        if -literal in clause[i+1:] :
            return True
    return False
    
def distinct_values(subst) :
    return len(subst.values()) == len(set(subst.values())) 


class Literal(object) :
    
    def __init__(self, kb, functor, args, sign=True) :
        self.functor = functor
        self.args = list(args)
        self.sign = sign
        self.kb = kb
    
    identifier = property(lambda s : (s.functor, len(s.args) ) )
    is_negated = property(lambda s : not s.sign)
                
    def __str__(self) :  
        return repr(self)
        
    def __repr__(self) :
        sign = '\+' if not self.sign else ''
        args = '(%s)' % ','.join(self.args) if self.args else ''
        return '%s%s%s' % (sign, self.functor, args)

    def __neg__(self) :
        return Literal(self.kb, self.functor, self.args, not self.sign)

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return self.functor == other.functor and self.args == other.args and self.sign == other.sign
                
    def __lt__(self, other) :
        return (self.sign, len(self.args), self.functor, self.args) < (other.sign, len(other.args), other.functor, other.args)
                        
    def unify(self, ground_args) :
        """Unifies the arguments of this literal with the given list of literals and returns the substitution.
            Only does one-way unification, only the first literal should have variables.
            
            Returns the substitution as a dictionary { variable name : value }.
        """
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if is_var(ARG) :
                if not ARG == '_' :
                    result[ARG] = arg
            elif is_var(arg) :
                raise ValueError("Unexpected variable in second literal : '%s'!" % (ARG))
            elif ARG == arg :
                pass    # default case
            else :
                raise ValueError("Literals cannot be unified: '%s' '%s!'" % (arg, ARG))
        return result  
        
    def assign(self, subst) :
        """Creates a new literal where the variables are assigned according to the given substitution."""
        return Literal(self.kb, self.functor, [ subst.get(arg, arg) for arg in self.args ], self.sign)
        
    def get_vars(self, kb) :
        return [ (arg, typ) for arg,typ in zip(self.args, kb.argtypes_l(self)) if is_var(arg) and not arg == '_' ]
        
        
        
    def variables(self, kb=None) :
        """Creates a dictionary of variables in the literal by variable type."""
        if kb == None : kb = self.kb
        result = defaultdict(set)
        types = kb.argtypes(strip_negation(self.functor), len(self.args))
        for tp, arg in zip(types, self.args) :
            if is_var(arg) and not arg == '_' :
                result[tp].add(arg)
        return result
        
    @classmethod
    def parse(cls, kb, string) :
        """Parse a literal from a string."""
        
        # TODO does not support literals of arity 0.
        
        regex = re.compile('\s*(?P<name>[^\(]+)\((?P<args>[^\)]+)\)[,.]?\s*')
        result = []
        for name, args in regex.findall(string) :
            if name.startswith('\+') :
                name = strip_negation(name)
                sign = False
            else :
                sign = True
            result.append( Literal( kb, name , map(lambda s : s.strip(), args.split(',')), sign) )
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
        self.query_cache = {}
        
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

    def argtypes_l(self, literal) :
        return self.predicates[literal.identifier][2]
            
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
        
        if literal.is_negated :
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
        
    def find_fact(self, literal) :
        
        index, values, types, probs = self.predicates[literal.identifier]
        
        result = set(range(0, len(values))) 
        for i,arg in enumerate(literal.args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break        
        
        probability = sum( probs[i] for i in result )        
        if literal.is_negated :
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

    def query(self, literals, substitution, facts_used=None, distinct=False) :
        if not literals :  # reached end of query
            yield substitution, []
        else :
            head, tail = literals[0], literals[1:]
            head_ground = head.assign(substitution)     # assign known variables
            
            for is_det, match in self.ground_fact(head_ground, facts_used) :
                # find match for head
                new_substitution = dict(substitution)
                new_substitution.update( head_ground.unify( match ) )
                
                if distinct and not distinct_values(new_substitution) :
                    continue
                    
                if is_det :
                    my_lit = []
                else :
                    my_lit = [ head_ground.assign(new_substitution) ]
                for sol, sub_lit in self.query(tail, new_substitution, facts_used) :
                    if not my_lit or my_lit[0] in sub_lit :
                        next_lit = sub_lit
                    else :
                        next_lit = my_lit + sub_lit
                    
                    yield sol, next_lit
    
    def facts_string(self, facts) :
        result = ''
        for identifier in facts :
            name, arity = identifier
            index, values, types, probs = self.predicates[identifier]
            
            for i in facts[identifier] :
                args = ','.join(values[i])
                prob = probs[i]
                if prob == 1 :
                    result += '%s(%s).\n' % (name, args)
                else :
                    result += '%s::%s(%s).\n' % (prob, name, args)
        return result
                    
    def toPrologFile(self) :
        if self.prolog_file == None :
            self.prolog_file = '/tmp/pf.pl'
            with open(self.prolog_file, 'w') as f :
                for pred in self.predicates :
                    index, values, types, probs = self.predicates[pred]
                    for args in values :
                        print('%s(%s).' % (pred[0], ', '.join(args) ), file=f)
        return self.prolog_file
        
        


class QueryPack(object) :
     
    # This field is shared by all objects for this class!
    query_cache = {}
     
    def __init__(self, kb) :
        self.__queries = defaultdict(list)
        self.__kb = kb
        self.used_facts = defaultdict(set)
        
    def addQuery(self, query) :
        if query.clauses : 
            key = query.key
            val = query.value
            self.__queries[key].append(val)
        
    def __iadd__(self, query) :
        """Add a query."""
        self.addQuery(query)
        return self
    
    def __len__(self) :
        return len(self.__queries)
        
    def __iter__(self) :
        return iter(self.__queries)
        
    def __getitem__(self, key) :
        return self.__queries[key]
     
    def execute( self, scores, extensions ) :
        outfile = '/tmp/pf.pl'
        cl_cnt = 0
        q_cnt = 0
        cache_cnt = 0
    
        queries = []
#        problog_data = []
        problog_queries = []
    
        with open(outfile, 'w') as out :
            facts = self.__kb.facts_string(self.used_facts)
            print(facts, file=out)
        
            for q_i, body in enumerate(self) :
                q_id = 'pfq_%s' % q_i 
            
            
                p = self.query_cache.get(body, None)
                if p != None :
                    cache_cnt += 1
                    for lit_i, ex_i in self[body] :
                        scores[lit_i][ex_i] = p
                    continue

                # print(body, clauses[body])
            
                problog_queries.append('')

                for lit_i, ex_i in self[body] :
                    queries.append( (lit_i, ex_i, q_id, body ) )
        
                for b in body.split(';') :
                    cl_cnt += 1
                
                    problog_queries[-1] += '%s :- %s.\n' % (q_id, b)
                
                    print('%s :- %s.' % (q_id, b), file=out)
                q_cnt += 1
                print('query(%s).\n\n' % (q_id), file=out)
                problog_queries[-1] += 'query(%s).\n\n' % (q_id)
    
    
        with Log('problog', nr_of_queries=q_cnt, nr_of_clauses=cl_cnt, cache_hits=cache_cnt, _timer=True) as l :

            if q_cnt == 0 :
                return
        
            with Log('facts', _child=facts) : 
                pass
            
            for q in problog_queries :
                with Log('query') as lq :
                    if lq.file :
                        print ( q, file=lq.file) 
                
            if q_cnt :
                import problog
                engine = problog.ProbLogEngine.create([])
                logger = problog.Logger(False)
                dir_out = '/tmp/pf/'
                with problog.WorkEnv(dir_out,logger, persistent=problog.WorkEnv.ALWAYS_KEEP) as env :
                    probs = engine.execute(outfile, env) # TODO call problog
        
        
        for l,e,q,b in queries :
            p = probs[q]
            scores[l][e] = p
            self.query_cache[b] = p
 
         
class Query(object) :
     
    def __init__(self, example_id, extension_id, parent=None) :
        self.example_id = example_id
        self.extension_id = extension_id
        if parent :
            self.clauses = parent.clauses[:]
        else :
            self.clauses = []
        
    key = property( lambda s : str(s) )
    value = property( lambda s : (s.extension_id, s.example_id ) )
        
    def __iadd__(self, clause_body) :
        self.addClause(clause_body)
        return self
        
    def addClause(self, new_clause) :
        new_clause = set(new_clause)
        i = 0
        while i < len(self.clauses) :
            existing_clause = self.clauses[i]
            if self._test_subsume(existing_clause, new_clause) :
                # existing clause subsumes new clause => discard new clause
                return
            elif self._test_subsume(new_clause, existing_clause) :
                # new clause subsumes existing clause => discard existing clause
                self.clauses.pop(i)
            else :
                i += 1
        self.clauses.append( new_clause )
        self.clauses = sorted(self.clauses)

    def _test_subsume(self, clause1, clause2) :
        return clause1 <= clause2
        
    def __str__(self) :
        return ';'.join( ','.join( map(str, clause) ) for clause in self.clauses )

class Rule(object) :
    """Class representing a learned rule.
    
    It has the following parts:
        - head  - the head literal
        - body  - a list of body literals
        
        - score - scoring statistics in its current context (data and hypothesis)
    """
    
    variables = property(lambda s : s._get_varinfo()[0])
    varnames = property(lambda s : s._get_varinfo()[1])

    def __init__(self, head, body, score, comment = None) :
        self.head = head
        self.body = body
        self.varinfo = None
        self.score = score
        self.comment = comment

    def _get_varinfo(self) :
        if self.varinfo == None :
            self.varinfo = self._extract_variables([self.head] + self.body)
        return self.varinfo
        
    def _ground_extensions(self, H, examples, extensions) :
        kb = H.data
        queries = QueryPack(kb)
        
        scores = [ [] for ext in extensions ]
        
        for ex_i, example in enumerate(examples) :
            p, ph, ex_id = example
            
            subst = self.head.unify( kb[ex_id] )
            
            with Log('example', example=kb[ex_id], exid=ex_id, exi = ex_i) :
                pass
            
            previous_query = Query(ex_i, None)
            previous_rules = H.rules[:-1]
            for prev_rule in previous_rules :
                for body_vars, old_clause in kb.query(prev_rule.body, subst, queries.used_facts) :
                    previous_query += old_clause
        
            literal_scores = self._ground_extensions_example(kb, ex_i, ph, subst, extensions, queries, previous_query)
            for lit_i, score in enumerate(literal_scores) :
                scores[lit_i].append(score)     
        
        return scores, queries

    def _ground_extensions_example(self, kb, ex_id, ph, subst, extensions, queries, previous_query) :
        
        literal_score = [ 0.0 ] * len(extensions)    # default 0
        literal_ext = []

        literal_clauses = [ [] for ext in extensions ]

        literal_restricts = [ False ] * len(extensions)
        literal_isdet = [ True ] * len(extensions)
        
        literal_query = [ Query(ex_id, ext_i, previous_query) for ext_i, ext in enumerate(extensions) ]
        
        for body_vars, old_clause in kb.query(self.body, subst, queries.used_facts) :
            
            if test_fail(old_clause) :
                continue
            
            for lit_i, lit in enumerate(extensions) :
                head = 'pfq_%s_%s' % (ex_id, lit_i)
                old_rule = Rule( Literal(kb, head, []), old_clause , None )
                if literal_score[lit_i] == 1 :
                    continue    # literal was already proven to be deterministically true

                # DONE if new literal is deterministic and does not eliminate options => reuse old score
                # TODO if new literal is not deterministic and is used in previous calculation => reuse old score
                # TODO if new literal is not deterministic and was not used in previous calculation => old score * probability
            
                current_literal_restricts = True
                new_query = Query(ex_id, lit_i)
                for lit_val, lit_new in kb.query([lit],body_vars, queries.used_facts) :
                    
                                        
                    if old_clause + lit_new : 
                        # probabilistic query
                        valid = 0
                        if lit_new : 
                            valid = old_rule.valid_extension(lit_new[0])
                        
                        if valid == -1 :    # new literal is negation of another literal in the body
                            #print(old_rule, lit_new, literal_query[lit_i].clauses)
                            pass    # clause = False
                        else :
                            current_literal_restricts = False
                            if valid == 1 :
                                lit_new = []
                                # we remove the new literal => ALL other clauses for this literal are subsumed !
                                break
                            else :
                                literal_isdet[lit_i] = False
                                literal_score[lit_i] = None  # probabilistic
                            literal_query[lit_i] += (old_clause + lit_new)
                    else :
                        current_literal_restricts = False
                        # non-probabilistic => p = 1.0
                        literal_score[lit_i] = 1.0    # deterministically true
                literal_restricts[ lit_i ] |= current_literal_restricts
        
        for lit_i, lit in enumerate(extensions) :
            if not literal_restricts[lit_i] and literal_isdet[lit_i] :
                literal_score[lit_i] = ph
            else :
                queries += literal_query[lit_i]
                
        return literal_score
        
    def valid_extension(self, literal) :
        if literal in self.body :
            return 1
        elif -literal in self.body :
            return -1
        else :
            return 0
        
        
    def evaluate_extensions(self, H, examples, extensions) :
        scores, queries = self._ground_extensions(H, examples, extensions )         
        queries.execute(scores, extensions)
        
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
            yield '#'
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
                    if arg0 == '#' :
                        yield [kb.newVar()] + argN
                    else :
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
        else :
            use_vars = None
        
        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, True, arg_info, [], use_vars) :
                yield Literal(kb, pred_name, args)

        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, False, arg_info, [], use_vars) :
                yield Literal(kb, pred_name, args, False)

    def __str__(self) :
        if self.comment != None :
            res = '% ' + self.comment + '\n'
        else :
            res = ''
        if self.body :
            res += str(self.head) + ' :- ' + ', '.join(map(str,self.body)) + '.'
        else :
            res += str(self.head) + '.'
        return res
        
    def __len__(self) :
        return len(self.body)
        
    def countNegated(self) :
        return sum( b.is_negated for b in self.body )    

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
                    with Log('learn', input=filename, target=target, _timer=True, **vars(SETTINGS)) :
                        H = learn(Hypothesis(Rule, Score, target, kb))   
    
                        print(H)
                        print(H.score, H.score.globalScore)
    
                        with Log('result') as l :
                            if l.file :
                                print(H, file=l.file)
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    